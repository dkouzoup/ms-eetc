import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.estimator import forceEstimator, plotVelocityComparison, plotForceComparison, plotTimeCoparison, \
    energyEstimator
from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train
from simulations.sim_launcher import get_power_loss_function, printStats


if __name__ == '__main__':
    """
    Analyze Night Data and Validate Estimation and Optimization Pipeline
    1. Import data
    2. Compare different position data sources (odometry, position and velocity integration)
    3. Alignment of track position data and train position measurement using gradient alignment
    4. Estimate force profile given odometry
    5. Estimate energy give force profile and odometry
    6. Compute optimal trajectory (with and without ETCS)
    7. Compare optimal trajectory with real trajectory (energy consumption and velocity profiles)
    """


    ### 1. Import Data

    df = pd.read_csv("../nightTests/journey1_odometry.csv")

    times = df['Time [s]'].to_numpy()
    positions = df['Position [m]'].to_numpy()
    odometry = df['Odometry [m]'].to_numpy()
    velocities = df['Velocity [m/s]'].to_numpy()
    forces = df['Force (el) [N]'].to_numpy()
    gradients = df['Gradient [-]'].to_numpy()

    # set startpoint of the journey to position = 0 m and start time = 0s

    times = times - times[0]
    positions = positions - positions[0]
    odometry = odometry - odometry[0]


    ### 2. Compare different position data sources (odometry, position and velocity integration)

    # compute position based on velocity

    positionBasedOnVelocity = np.zeros_like(positions)
    positionBasedOnVelocity[0] = positions[0]
    dt = np.diff(times)
    vMean = (velocities[:-1] + velocities[1:]) / 2
    positionBasedOnVelocity[1:] = positions[0] + np.cumsum(vMean * dt)

    # Fig 1.: Position Data Source Comparison

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.plot(times/60, positions, label="Position")
    ax.plot(times/60, odometry, label="Odometry")
    ax.plot(times / 60, positionBasedOnVelocity, label="Velocity Integration")
    ax.set_title("Fig 1.: Position Data Source Comparison")
    ax.set_ylabel("Position [m]")
    ax.set_xlabel("Time [min]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, times[-1]/60)
    ax.figure.tight_layout()

    plt.show()

    relativeEndPositionDifference = (odometry[-1] - positions[-1]) / positions[-1]
    print(f"Relative end position difference for Odometry vs Position: {relativeEndPositionDifference * 100:.2f} %")

    relativeEndPositionDifference = (positionBasedOnVelocity[-1] - positions[-1]) / positions[-1]
    print(f"Relative end position difference for Velocity Integration vs Position: {relativeEndPositionDifference * 100:.2f} %")

    relativeEndPositionDifference = (positionBasedOnVelocity[-1] - odometry[-1]) / odometry[-1]
    print(f"Relative end position difference for Velocity Integration vs Odometry: {relativeEndPositionDifference * 100:.2f} %")


    ### 3. Alignment of Track Position Data and Train Position Measurement using Gradient Alignment

    # manual tuning parameters
    startPosition = 337
    positionMultiplier = 1.006

    track = Track(config={'id': 'trackNight'}, pathJSON='../nightTests')
    gradTrackPositions = track.gradients.index.to_numpy()
    gradTrackValues = track.gradients["Gradient [permil]"].to_numpy()

    gradMeasurementPositions = df['Odometry [m]'].to_numpy() + startPosition
    positionSteps = positionMultiplier * np.diff(gradMeasurementPositions)
    gradMeasurementPositionsScaled = np.zeros_like(gradMeasurementPositions)
    gradMeasurementPositionsScaled[0] = gradMeasurementPositions[0]
    gradMeasurementPositionsScaled[1:] = gradMeasurementPositions[0] + np.cumsum(positionSteps)
    gradMeasurementPositions = gradMeasurementPositionsScaled
    gradMeasurementValues = gradients*1000

    # Fig 2.: Gradient Alignment

    fig2, ax = plt.subplots(figsize=(18, 12))
    ax.step(gradMeasurementPositions, gradMeasurementValues, where="post", label="Measurements")
    ax.step(gradTrackPositions, gradTrackValues, where="post", label="Track Data")
    ax.set_title("Fig 2.: Alignment of Track Position Data and Train Position Measurement using Gradient Alignment")
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Gradient [permil]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(gradMeasurementPositions[0], gradMeasurementPositions[-1])
    ax.figure.tight_layout()

    plt.show()


    ### 4. Estimate Force Profile given Odometry

    train = Train(config={'id':'trainNight'}, pathJSON='../nightTests')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    track = Track(config={'id':'trackNight'}, pathJSON='../nightTests')
    track.updateTrainLengthDependentValues(train)

    # Fig 3.: Using Gradient Alignment to align Track Position Data and Train Position Measurement for train-length-dependent values

    fig3, ax = plt.subplots(figsize=(18, 12))
    ax.step(gradMeasurementPositions + train.length*0.5, gradMeasurementValues, where="post", label="Measurements")
    ax.step(track.gradients.index.to_numpy(), track.gradients["Gradient [permil]"].to_numpy(), where="post", label="Track Data Train-length-dependent")
    ax.set_title("Fig 3.: Using Gradient Alignment to align Track Position Data and Train Position Measurement for train-length-dependent values")
    ax.set_xlabel("Position [m]")
    ax.set_ylabel("Gradient [permil]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(gradMeasurementPositions[0], gradMeasurementPositions[-1])
    ax.figure.tight_layout()

    plt.show()

    # prepare input data for forceEstimator: targetDf

    targetDf = df[["Time [s]", "Velocity [m/s]", "Force (el) [N]"]].copy()

    targetDf["Time [s]"] = pd.to_numeric(targetDf["Time [s]"], errors="coerce")
    targetDf["Velocity [m/s]"] = pd.to_numeric(targetDf["Velocity [m/s]"], errors="coerce")
    targetDf["Force (el) [N]"] = pd.to_numeric(targetDf["Force (el) [N]"], errors="coerce")

    targetDf["Time [s]"] = targetDf["Time [s]"] - targetDf["Time [s]"].iloc[0]  # set startTime to 0s
    targetDf["Position [m]"] = gradMeasurementPositions + train.length*0.5  # use the correct train-length-dependent position

    targetDf = targetDf.dropna()
    targetDf = targetDf.sort_values("Time [s]")
    targetDf = targetDf.set_index("Time [s]")
    targetDf.index.name = None

    targetDf = targetDf[~targetDf.index.duplicated(keep="first")]  # needs strictly monotone increasing times
    targetDf = targetDf[targetDf["Position [m]"].diff().fillna(1) > 0]  # needs strictly monotone increasing positions

    optsDict = {'numIntervals': 500, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

    fEstimator = forceEstimator(targetDf, train, track, optsDict=optsDict, trainLengthDependentValues=True)
    dfEstimate = fEstimator.estimate()

    # visual validation of estimation

    plotVelocityComparison(targetDf, dfEstimate)
    plotForceComparison(targetDf, dfEstimate)
    plotTimeCoparison(targetDf, dfEstimate)


    ### 5. Estimate Energy give Force Profile and Odometry

    eEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsDict)
    energyStats = eEstimator.estimate()
    eEstimator.printStats(energyStats)


    ### 6. Compute optimal Trajectory

    train.powerLosses = get_power_loss_function(train, "static")  # perfect

    start = targetDf["Position [m]"].iloc[0]
    end = targetDf["Position [m]"].iloc[-1]
    duration = df["Time [s]"].iloc[-1] - df["Time [s]"].iloc[0]

    print(f"Start position: {start:.1f} m")
    print(f"End position: {end:.1f} m")
    print(f"Duration: {duration:.1f} s")

    journey = Journey(config={'id':'trackNight_journey01'}, sectionIdx=0, pathJSON='../nightTests')  # journey file contains the previously printed journey data

    track = Track(config={'id':'trackNight'}, pathJSON='../nightTests')
    track.updateTrainLengthDependentValues(train)
    track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

    # non-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, journey, opts)
    df, stats = solver.solve()

    printStats(df, stats, solver, train)

    # ETCS-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True, 'withEtcsBrakingCurves': True}

    solverEtcs = casadiSolver(train, track, journey, opts)
    dfEtcs, statsEtcs = solverEtcs.solve()

    printStats(dfEtcs, statsEtcs, solverEtcs, train)


    ### 7. compare optimal Trajectory with real trajectory

    # compare energy use

    ocpEnergyNoETCS = stats["Cost"]
    ocpEnergyWithETCS = statsEtcs["Cost"]
    realEnergy = energyStats["Net energy used [kWh]"]
    print(f"ocpEnergy (no ETCS): {ocpEnergyNoETCS:.2f} kWh")
    print(f"ocpEnergy (with ETCS): {ocpEnergyWithETCS:.2f} kWh")
    print(f"realEnergy: {realEnergy:.2f} kWh")

    relativeEnergyDifference = (realEnergy - ocpEnergyNoETCS) / ocpEnergyNoETCS
    print(f"Relative energy difference (no ETCS): {relativeEnergyDifference * 100:.2f} %")
    relativeEnergyDifference = (realEnergy - ocpEnergyWithETCS) / ocpEnergyWithETCS
    print(f"Relative energy difference (with ETCS): {relativeEnergyDifference * 100:.2f} %")

    # compare velocity profile in space domain

    fig4, ax = plt.subplots(figsize=(18, 12))
    ax.plot((targetDf["Position [m]"].to_numpy()-targetDf["Position [m]"].to_numpy()[0]) * 0.001, targetDf["Velocity [m/s]"].to_numpy() * 3.6, label="real Trip")
    ax.plot(df["Position [m]"].to_numpy()*0.001, df["Velocity [m/s]"].to_numpy()*3.6, label="OCP (no Etcs)")
    ax.plot(dfEtcs["Position [m]"].to_numpy()*0.001, dfEtcs["Velocity [m/s]"].to_numpy()*3.6, label="OCP (with Etcs)")

    x = track.speedLimits.index.to_numpy(dtype=float)
    v = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    x_plot = np.append(x, track.length)
    v_plot = np.append(v, v[-1])

    ax.step(x_plot*0.001, v_plot*3.6, where="post", color="black", linestyle="-", label="Track Speed Limit")
    ax.plot(track.etcsPositions*0.001, track.etcsVelocities*3.6, color="red", linestyle="-", label="ETCS Speed Limit")

    ax.set_title("Fig 4.: Velocity Profile Comparison for Simulated and Real Trajectories w.r.t. Space")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    ax.set_xlim(df["Position [m]"].to_numpy()[0]*0.001, df["Position [m]"].to_numpy()[-1]*0.001)

    ax.figure.tight_layout()

    plt.show()

    # compare velocity profile in time domain

    fig5, ax = plt.subplots(figsize=(18, 12))
    ax.plot((targetDf.index.to_numpy()-targetDf.index.to_numpy()[0])/60, targetDf["Velocity [m/s]"].to_numpy() * 3.6, label="real Trip")
    ax.plot(df.index.to_numpy()/60, df["Velocity [m/s]"].to_numpy()*3.6, label="OCP (no Etcs)")
    ax.plot(dfEtcs.index.to_numpy()/60, dfEtcs["Velocity [m/s]"].to_numpy()*3.6, label="OCP (with Etcs)")
    ax.set_title("Fig 5.: Velocity Profile Comparison for Simulated and Real Trajectories w.r.t. Time")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    plt.show()