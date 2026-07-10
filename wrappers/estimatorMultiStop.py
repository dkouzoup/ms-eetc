from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from mseetc.estimator import forceEstimator, energyEstimator
from simulations.sim_launcher import get_power_loss_function


INTERVALS_PER_METER = 400/3000


def getTargetDf(file, trainLength):

    df = pd.read_csv(file)

    # todo:  add automatic preprocessing for odometry shift
    startPosition = 337  # match start of csv recordings with track position data
    positionMultiplier = 1.006  # account for odometry drift


    targetDf = df[["Time [s]", "Velocity [m/s]"]].copy()

    targetDf["Time [s]"] = targetDf["Time [s]"] - targetDf["Time [s]"].iloc[0]  # set start time to 0

    # linearly scale position measurements to account for integration drift in the measurement data
    positions = df['Odometry [m]'].to_numpy() + startPosition
    positionSteps = positionMultiplier * np.diff(positions)
    positionsScaled = np.zeros_like(positions)
    positionsScaled[0] = positions[0]
    positionsScaled[1:] = positions[0] + np.cumsum(positionSteps)

    # account for train-length-dependent values
    targetDf["Position [m]"] = positionsScaled + trainLength * 0.5

    targetDf = targetDf.dropna()
    targetDf = targetDf.sort_values("Time [s]")
    targetDf = targetDf.set_index("Time [s]")
    targetDf.index.name = None

    targetDf = targetDf[~targetDf.index.duplicated(keep="first")]  # needs strictly monotone increasing times
    targetDf = targetDf[targetDf["Position [m]"].diff().fillna(1) > 0]  # needs strictly monotone increasing positions

    return targetDf


if __name__ == '__main__':

    """
    Estimate energy consumption for multiple journey sections.
    
    Each journey section must be specified in a single csv file.
    Each csv file must have the following columns:
        - "Time [s]"
        - "Velocity [m/s]
        - "Odometry [m]
    All csv files must be saved in the same input folder (directory).
    
    Only make changes in the "Input" section of this script.
    
    Results are saved in a new directory called "estimator" located in the input folder (directory).
    Per journey section the resulting estimated trajectory is saved in a pickle file - 
    one for the force and one for the energy estimation.
    For easy data access, estimated energy consumption per section is saved in a csv file.
    """

    from mseetc.train import Train
    from mseetc.track import Track


    ####################################################################################################################
    ### Input
    directory = '../nightTests'
    trainId = 'trainNight'
    trackId = 'trackNight'
    ####################################################################################################################


    # Results Folder
    estimatorDirectory = Path(directory) / "estimator"
    estimatorDirectory.mkdir(exist_ok=True)

    # Train

    train = Train(config={'id':trainId}, pathJSON=directory)
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    csvFiles = list(Path(directory).glob("*.csv"))

    energyResults = []
    for sectionId in range(len(csvFiles)):

        targetDf = getTargetDf(csvFiles[sectionId], train.length)

        track = Track(config={'id': trackId}, pathJSON=directory)
        track.updateTrainLengthDependentValues(train)

        # automatic assignment of shooting node count
        numOfIntervals = floor((targetDf["Position [m]"].to_numpy()[-1] - targetDf["Position [m]"].to_numpy()[0]) * INTERVALS_PER_METER)
        print(f"numOfIntervals: {numOfIntervals}")

        optsDict = {'numIntervals': numOfIntervals, 'integrationMethod': 'RK','integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

        fEstimator = forceEstimator(targetDf, train, track, optsDict=optsDict, trainLengthDependentValues=True)
        dfEstimate = fEstimator.estimate()

        eEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsDict)
        energyStats = eEstimator.estimate()

        forceFile = estimatorDirectory / f"{csvFiles[sectionId].stem}_df_force_estimate.pkl"
        dfEstimate.to_pickle(forceFile)

        energyFile = estimatorDirectory / f"{csvFiles[sectionId].stem}_df_energy_estimate.pkl"
        pd.to_pickle(energyStats, energyFile)

        energyResults.append({
            "File": csvFiles[sectionId].stem,
            "Energy [kWh]": energyStats["Net energy used [kWh]"]
        })

    energyFile = estimatorDirectory / "energyStats.csv"
    energyResultsDf = pd.DataFrame(energyResults)
    energyResultsDf.to_csv(energyFile, index=False)