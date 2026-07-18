from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
import re

from mseetc.estimator import forceEstimator, energyEstimator
from mseetc.utils import get_power_loss_function

INTERVALS_PER_METER = 400/3000


def getTargetDf(file, trainLength, df_stationDict):

    pattern = re.compile(
        r"^odometry_(?P<id>[^_]+)_(?P<from_station>[^_]+)_(?P<to_station>[^_]+)\.csv$",
        re.IGNORECASE,
    )

    match = pattern.fullmatch(file.name)

    assert match, (
        f"Invalid odometry filename: {file.name}. "
        "Expected: odometry_<ID-or-Name>_<From-Station>_<To-Station>.csv"
    )

    fromStation = match.group("from_station")
    toStation = match.group("to_station")

    assert fromStation in df_stationDict.index.values, (
        f"Station '{fromStation}' not found in df_stationDict"
    )

    positionFromStation = df_stationDict.loc[fromStation, "Position [m]"]

    assert toStation in df_stationDict.index.values, (
        f"Station '{toStation}' not found in df_stationDict"
    )

    postionToStation = df_stationDict.loc[toStation, "Position [m]"]

    assert positionFromStation < postionToStation, f"From Station '{fromStation}' needs to be before To Station '{toStation}'"

    df = pd.read_csv(file)

    requiredColumns = {"Time [s]", "Velocity [m/s]", "Odometry [m]"}
    missingColumns = requiredColumns - set(df.columns)

    assert not missingColumns, (
        f"Input CSV ({file.name}) is missing the following required columns: "
        f"{sorted(missingColumns)}"
    )

    targetDf = df[["Time [s]", "Velocity [m/s]"]].copy()

    targetDf["Time [s]"] = targetDf["Time [s]"] - targetDf["Time [s]"].iloc[0]  # set start time to 0

    positions = df['Odometry [m]'].to_numpy()
    positions -= positions[0]

    # linearly scale position measurements to account for integration drift in the measurement data
    distanceReference = postionToStation - positionFromStation
    distanceMeasurement = positions[-1] - positions[0]
    positionMultiplier = distanceReference / distanceMeasurement  # account for odometry drift

    positionSteps = positionMultiplier * np.diff(positions)
    positionsScaled = np.zeros_like(positions)
    positionsScaled[0] = positions[0]
    positionsScaled[1:] = positions[0] + np.cumsum(positionSteps)

    startPosition = positionFromStation  # match start of csv recordings with track position data
    positionsScaled += startPosition

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
    The Title of each csv file must have the following form:
        odometry_<<ID/Name>>_<<From_Station_Name>>_<<To_Station_Name>>.csv
    Each csv file must have the following columns:
        - "Time [s]"
        - "Velocity [m/s]
        - "Odometry [m]
        
    All csv files must be saved in the same input folder (directory).
    
    The input folder (directory) must contain a station dict.
    It is a csv file with the following columns:
        - "Station_Name"
        - "Position [m]"
    The "Position [m]" is the midpoint position of the station, where the position reference comes from the track data used.
    
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
    directory = '../../nightTests'
    stationDict = "stationDict.csv"
    trainDirectory = '../../nightTests'
    trainId = 'trainNight'
    trackDirectory = '../../nightTests'
    trackId = 'trackNight'
    ####################################################################################################################


    # Results Folder
    estimatorDirectory = Path(directory) / "estimator"
    estimatorDirectory.mkdir(exist_ok=True)

    # Train
    train = Train(config={'id':trainId}, pathJSON=trainDirectory)
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    # Station Dict
    stationDictPath = Path(directory) / stationDict
    df_stationDict = pd.read_csv(stationDictPath)
    requiredColumns = {"Station_Name", "Position [m]"}
    missingColumns = requiredColumns - set(df_stationDict.columns)

    assert not missingColumns, (
        f"StationDict is missing the following required columns: "
        f"{sorted(missingColumns)}"
    )

    df_stationDict = df_stationDict.set_index("Station_Name")

    csvFiles = list(Path(directory).glob("odometry_*.csv"))

    energyResults = []
    for sectionId in range(len(csvFiles)):

        targetDf = getTargetDf(csvFiles[sectionId], train.length, df_stationDict)

        track = Track(config={'id': trackId}, pathJSON=trackDirectory)
        track.updateTrainLengthDependentValues(train)

        # automatic assignment of shooting node count
        numOfIntervals = floor((targetDf["Position [m]"].to_numpy()[-1] - targetDf["Position [m]"].to_numpy()[0]) * INTERVALS_PER_METER)
        print(f"numOfIntervals: {numOfIntervals}")

        optsDict = {'numIntervals': numOfIntervals, 'integrationMethod': 'RK','integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

        fEstimator = forceEstimator(targetDf, train, track, optsDict=optsDict, trainLengthDependentValues=True)
        dfEstimate = fEstimator.estimate()

        eEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsDict)
        energyStats = eEstimator.estimate()

        name = result = csvFiles[sectionId].stem.split("_", 1)[1]

        forceFile = estimatorDirectory / f"{name}_df_force_estimate.pkl"
        dfEstimate.to_pickle(forceFile)

        energyFile = estimatorDirectory / f"{name}_df_energy_estimate.pkl"
        pd.to_pickle(energyStats, energyFile)

        energyResults.append({
            "File": name,
            "Energy [kWh]": energyStats["Net energy used [kWh]"]
        })

    energyFile = estimatorDirectory / "energyStats.csv"
    energyResultsDf = pd.DataFrame(energyResults)
    energyResultsDf.to_csv(energyFile, index=False)