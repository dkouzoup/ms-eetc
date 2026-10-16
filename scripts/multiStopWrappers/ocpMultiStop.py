import casadi as ca

from math import floor
from pathlib import Path

import pandas as pd

from mseetc.ocp import casadiSolver
from mseetc.utils import get_power_loss_function, postProcessDataFrame

from runMetadata import createRunDirectory, describeTrain, describeTunnels, findRunWithSameInputs, \
    getGitCommit, hashInputs, removeRun, resolveDuplicateRun, writeMetadata
from tunnelAgnostic import estimateEnergyOnTrack, removeTunnels, reportForceLimitViolations
from wrapperConfig import loadConfig

INTERVALS_PER_METER = 400/4000

def loadPowerLossFunction(path):
    "Load a serialized casadi power-loss function (train-specific, measured)."

    return ca.Function.deserialize(Path(path).read_text())



if __name__ == '__main__':

    """
    Compute energy optimal trajectory for a multi-stop-journey
    Only make changes in config.json (copy config.example.json to get started).
    Results are saved in a new timestamped directory under "ocp" in the input folder,
    so that runs with different parameters can be compared instead of overwriting each other.
    Should a run with exactly the same inputs already exist, it can be replaced instead.
    Per journey section the resulting optimized trajectory is saved in a pickle file.
    All parameters of the run and the resulting energy consumption per section are saved
    in metadata.json (read by the dashboard).
    """

    from mseetc.train import Train
    from mseetc.track import Track
    from mseetc.journey import Journey


    #### INPUT #################################################################

    losses_ocp = 'perfect'
    losses_sim = 'dynamic'

    withEtcsBrakingCurves = True
    tunnelsInOcp = False
    withPnBrake = False
    roundTrip = True
    integrationMethod = 'RK'
    integrationOptions = {'numApproxSteps': 1}

    ############################################################################

    # edit config.json, see config.example.json
    config = loadConfig()

    directory = config['directory']
    trainDirectory = config['trainDirectory']
    trainId = config['trainId']
    trackDirectory = config['trackDirectory']
    trackId = config['trackId']
    journeyDirectory = config['journeyDirectory']
    journeys = config['journeys']
    lossesFile = config['lossesFile']  # measured power losses of the train, relative to directory

    # everything that determines the results, and therefore the identity of the run
    inputs = {
        "directory": directory,
        "trainDirectory": trainDirectory,
        "trainId": trainId,
        "trackDirectory": trackDirectory,
        "trackId": trackId,
        "journeyDirectory": journeyDirectory,
        "journeys": journeys,
        "losses": {
            "ocp": losses_ocp,
            "sim": losses_sim,
            "lossesFile": lossesFile if 'dynamic' in (losses_ocp, losses_sim) else None,
        },
        "discretization": {"intervalsPerMeter": INTERVALS_PER_METER},
        "solver": {
            "withEtcsBrakingCurves": withEtcsBrakingCurves,
            "integrationMethod": integrationMethod,
            "integrationOptions": integrationOptions,
            "energyOptimal": True,
        },
        "train": {"withPnBrake": withPnBrake},
        "roundTrip": roundTrip,
        "tunnelsInOcp": tunnelsInOcp,
    }

    inputsHash = hashInputs(inputs)

    # Results Folder: one timestamped directory per run
    runsDirectory = Path(directory) / "ocp"

    duplicate = findRunWithSameInputs(runsDirectory, inputsHash)
    supersededRun = duplicate if duplicate and resolveDuplicateRun(duplicate) == 'replace' else None

    ocpDirectory = createRunDirectory(runsDirectory)
    print(f"Results of this run are saved in {ocpDirectory}")

    # Train

    train = Train(config={'id':trainId}, pathJSON=trainDirectory)

    if not withPnBrake:

        train.forceMinPn = 0  # the solver reads the flag off this, see Train.exportModel

    lossFunctions = {
        'perfect': lambda: get_power_loss_function(train, 'perfect'),
        'static': lambda: get_power_loss_function(train, 'static'),
        'dynamic': lambda: loadPowerLossFunction(Path(directory) / lossesFile),
    }

    losses_fun_ocp = lossFunctions[losses_ocp]()
    losses_fun_sim = lossFunctions[losses_sim]()

    # Journey
    #
    # One journey per direction: the return service of a line is a timetable of its own,
    # in general not the mirror image of the outbound one.

    directions = ['forward', 'reverse'] if roundTrip else ['forward']

    for direction in directions:

        assert direction in journeys, (
            f"No journey given for the '{direction}' direction. "
            f"'journeys' in config.json needs one journey id per direction: {directions}.")

        print(f"{direction}: journey '{journeys[direction]}'")

    sectionMetadata = []  # parameters, stats and energies per journey section

    distanceOffset = 0  # distance covered since the start of the trip, to lay the sections out in a row

    def saveMetadata(complete):

        """
        Write the metadata of the run as it stands.

        Called after every section, so that a run that is interrupted or dies part way
        through still describes the results it produced.
        """

        writeMetadata(ocpDirectory, {
            "type": "ocp",
            "script": Path(__file__).name,
            "inputsHash": inputsHash,
            "complete": complete,
            "gitCommit": getGitCommit(),
            "inputs": inputs,
            "train": describeTrain(train),
            "sections": sectionMetadata,
        })

    saveMetadata(complete=False)

    for direction in directions:

        journeyId = journeys[direction]

        numOfJourneySections = len(Journey(config={'id': journeyId},
                                           pathJSON=journeyDirectory).journeySectionBounds)

        for sectionIdx in range(numOfJourneySections):

            # new ocp per journey section

            journey = Journey(config={'id': journeyId}, pathJSON=journeyDirectory)

            journey.selectJourneySection(sectionIdx)

            track = Track(config={'id': trackId}, pathJSON=trackDirectory)

            if direction == 'reverse':

                track.reverse()

            if not tunnelsInOcp:

                removeTunnels(track)  # before updateLimits, while the track is still complete

            track.updateTrainLengthDependentValues(train)
            track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

            train.powerLosses = losses_fun_ocp

            numOfIntervals = floor((journey.positionEnd - journey.positionStart) * INTERVALS_PER_METER)  # automatic assignment of shooting node count
            print(f"{direction} section {sectionIdx}, numOfIntervals: {numOfIntervals}")

            opts = {'numIntervals': numOfIntervals, 'integrationMethod': integrationMethod, 'integrationOptions': integrationOptions,
                    'energyOptimal': True, 'withEtcsBrakingCurves': withEtcsBrakingCurves}

            solver = casadiSolver(train, track, journey, opts)
            df, stats = solver.solve()

            name = f"{direction} {sectionIdx}"
            sectionStart = distanceOffset
            distanceOffset += journey.positionEnd - journey.positionStart

            # positions in the dataframes are relative to the start of the section. positionStart
            # maps them back onto the track the journey is defined on, which for the return trip
            # is the reversed one, and distanceOffset lays every section of the trip out on one
            # continuous axis.
            section = {
                "index": len(sectionMetadata),
                "direction": direction,
                "journeyId": journeyId,
                "sectionIdx": sectionIdx,
                "name": name,
                "file": f"df_{direction}_section_{sectionIdx}.pkl",
                "positionStart [m]": journey.positionStart,
                "positionEnd [m]": journey.positionEnd,
                "distanceOffset [m]": sectionStart,
                "timeOffset [s]": getattr(journey, 'timeOffset', 0),
                "numIntervals": numOfIntervals,
                "solverOptions": solver.opts.toDict(),
                "stats": stats,
                # the solver overwrites the speed limits of its discretization points with the
                # ETCS braking curves, so keep the ones of the track for reference
                "speedLimits": {
                    "Position [m]": track.speedLimits.index.tolist(),
                    "Speed limit [m/s]": track.speedLimits['Speed limit [m/s]'].tolist(),
                },
                # the stretches the section runs in a tunnel, empty when tunnelsInOcp is off
                "tunnels": describeTunnels(track),
            }

            if df is None:

                # solver did not converge: record the section as missing and move on

                print(f"Section '{name}' skipped, solver status '{stats['Solver status']}'.")

                section["file"] = None
                sectionMetadata.append(section)
                saveMetadata(complete=False)

                continue

            train.powerLosses = losses_fun_sim

            df.to_pickle(ocpDirectory / section["file"])

            section["Energy OCP [kWh]"] = df["Energy [kWh]"].sum()

            if tunnelsInOcp:

                df_sim = postProcessDataFrame(df, solver.points, train, solver.opts, integrateLosses=True)

                section["Energy SIM [kWh]"] = df_sim["Energy [kWh]"].sum()

                # split by the sign of the force, as the estimator does, so that the two can be priced apart
                traction = df_sim["Force (el) [N]"] > 0

                section["Used energy [kWh]"] = df_sim.loc[traction, "Energy [kWh]"].sum()
                section["Regenerated energy [kWh]"] = -df_sim.loc[~traction, "Energy [kWh]"].sum()

            else:

                # the ocp never saw the tunnels, so the forces in df are the ones that would hold
                # this speed profile on a tunnel free track. Back-calculate the forces that hold it
                # on the real one, and report the energy those cost.

                realTrack = Track(config={'id': trackId}, pathJSON=trackDirectory)

                if direction == 'reverse':

                    realTrack.reverse()

                dfEstimate, energyStats = estimateEnergyOnTrack(
                    df, journey.positionStart, train, realTrack, opts)

                section["estimateFile"] = f"df_estimate_{direction}_section_{sectionIdx}.pkl"
                dfEstimate.to_pickle(ocpDirectory / section["estimateFile"])

                section["Energy SIM [kWh]"] = energyStats["Net energy used [kWh]"]
                section["Used energy [kWh]"] = energyStats["Used energy [kWh]"]
                section["Regenerated energy [kWh]"] = energyStats["Regenerated energy [kWh]"]
                section["energyEstimate"] = energyStats

                overForce, overPower = reportForceLimitViolations(dfEstimate, train)

                if overForce or overPower:

                    # the advised profile is not actually holdable through the tunnels
                    print(f"Section '{name}': planned profile exceeds the train at "
                          f"{overForce} points (force) and {overPower} points (power).")

                section["forceLimitViolations"] = {"force": overForce, "power": overPower}
            sectionMetadata.append(section)
            saveMetadata(complete=False)


    saveMetadata(complete=True)

    print(f"Wrote {len(sectionMetadata)} sections and metadata.json to {ocpDirectory}")

    if supersededRun is not None:

        removeRun(supersededRun)  # only now that the new results are safely in place
