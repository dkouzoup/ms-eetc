import casadi as ca

from math import floor
from pathlib import Path

import pandas as pd

from mseetc.ocp import casadiSolver
from mseetc.utils import get_power_loss_function, postProcessDataFrame

from runMetadata import createRunDirectory, describeTrain, findRunWithSameInputs, getGitCommit, \
    hashInputs, removeRun, resolveDuplicateRun, writeMetadata
from wrapperConfig import loadConfig

INTERVALS_PER_METER = 300/4000

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

    losses_ocp = 'dynamic'
    losses_sim = 'dynamic'

    withEtcsBrakingCurves = True
    withPnBrake = False
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
    journeyId = config['journeyId']
    lossesFile = config['lossesFile']  # measured power losses of the train, relative to directory

    # everything that determines the results, and therefore the identity of the run
    inputs = {
        "directory": directory,
        "trainDirectory": trainDirectory,
        "trainId": trainId,
        "trackDirectory": trackDirectory,
        "trackId": trackId,
        "journeyDirectory": journeyDirectory,
        "journeyId": journeyId,
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

    journey = Journey(config={'id':journeyId}, pathJSON=journeyDirectory)
    numOfJourneySections = len(journey.journeySectionBounds)

    sectionMetadata = []  # parameters, stats and energies per journey section

    for sectionIdx in range(numOfJourneySections):

        # new ocp per journey section

        journey = Journey(config={'id': journeyId}, sectionIdx=sectionIdx, pathJSON=journeyDirectory)

        track = Track(config={'id': trackId}, pathJSON=trackDirectory)
        track.updateTrainLengthDependentValues(train)
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        train.powerLosses = losses_fun_ocp

        numOfIntervals = floor((journey.positionEnd - journey.positionStart) * INTERVALS_PER_METER)  # automatic assignment of shooting node count
        print(f"numOfIntervals: {numOfIntervals}")

        opts = {'numIntervals': numOfIntervals, 'integrationMethod': integrationMethod, 'integrationOptions': integrationOptions,
                'energyOptimal': True, 'withEtcsBrakingCurves': withEtcsBrakingCurves}

        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        # positions in the dataframes are relative to the start of the section,
        # the offsets below map them back onto the track for cross-section plots
        section = {
            "index": sectionIdx,
            "file": f"df_section_{sectionIdx}.pkl",
            "positionStart [m]": journey.positionStart,
            "positionEnd [m]": journey.positionEnd,
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
        }

        if df is None:

            # solver did not converge: record the section as missing and move on

            print(f"Section {sectionIdx} skipped, solver status '{stats['Solver status']}'.")

            section["file"] = None
            sectionMetadata.append(section)

            continue

        train.powerLosses = losses_fun_sim
        df_sim = postProcessDataFrame(df, solver.points, train, solver.opts, integrateLosses=True)

        sectionFile = ocpDirectory / f"df_section_{sectionIdx}.pkl"
        df.to_pickle(sectionFile)

        section["Energy OCP [kWh]"] = df["Energy [kWh]"].sum()
        section["Energy SIM [kWh]"] = df_sim["Energy [kWh]"].sum()
        sectionMetadata.append(section)


    # save all parameters and results of the run, so that they stay interpretable later on
    writeMetadata(ocpDirectory, {
        "type": "ocp",
        "script": Path(__file__).name,
        "inputsHash": inputsHash,
        "gitCommit": getGitCommit(),
        "inputs": inputs,
        "train": describeTrain(train),
        "sections": sectionMetadata,
    })

    print(f"Wrote {len(sectionMetadata)} sections and metadata.json to {ocpDirectory}")

    if supersededRun is not None:

        removeRun(supersededRun)  # only now that the new results are safely in place
