import pickle
from math import floor
from pathlib import Path

import pandas as pd

from mseetc.ocp import casadiSolver
from simulations.sim_launcher import get_power_loss_function


INTERVALS_PER_METER = 300/4000


if __name__ == '__main__':

    """
    Compute energy optimal trajectory for a multi-stop-journey
    Only make changes in the "Input" section.
    Results are saved in a new directory called "ocp" located in the input folder.
    Per journey section the resulting optimized trajectory is saved in a pickle file.
    For easy data access, energy consumption per section is saved in a csv file.
    """

    from mseetc.train import Train
    from mseetc.track import Track
    from mseetc.journey import Journey


    ####################################################################################################################
    ### Input
    directory = '../nightTests'
    trainId = 'trainNight'
    trackId = 'trackNight'
    journeyId = 'trackNight_journeyNight'
    ####################################################################################################################


    # Results Folder
    ocpDirectory = Path(directory) / "ocp"
    ocpDirectory.mkdir(exist_ok=True)

    # Train

    train = Train(config={'id':trainId}, pathJSON=directory)
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    # Journey

    journey = Journey(config={'id':journeyId}, pathJSON=directory)
    numOfJourneySections = len(journey.journeySectionBounds)


    energyResults = []  # result data container

    for sectionIdx in range(numOfJourneySections):

        # new ocp per journey section

        journey = Journey(config={'id': journeyId}, sectionIdx=sectionIdx, pathJSON=directory)

        track = Track(config={'id': trackId}, pathJSON=directory)
        track.updateTrainLengthDependentValues(train)
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        numOfIntervals = floor((journey.positionEnd - journey.positionStart) * INTERVALS_PER_METER)  # automatic assignment of shooting node count
        print(f"numOfIntervals: {numOfIntervals}")

        opts = {'numIntervals': numOfIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1},
                'energyOptimal': True, 'withEtcsBrakingCurves': True}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        sectionFile = ocpDirectory / f"df_section_{sectionIdx}.pkl"
        df.to_pickle(sectionFile)

        energyResults.append({
            "File": f"section_{sectionIdx}",
            "Energy [kWh]": stats["Cost"]
        })


    # save energy consumption per sections in a csv file
    energyFile = ocpDirectory / "energyStats.csv"
    energyResultsDf = pd.DataFrame(energyResults)
    energyResultsDf.to_csv(energyFile, index=False)