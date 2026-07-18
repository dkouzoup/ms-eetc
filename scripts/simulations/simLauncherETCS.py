import numpy as np
from matplotlib import pyplot as plt

from mseetc.ocp import casadiSolver
from mseetc.utils import get_power_loss_function, printStats


if __name__ == '__main__':

    """
    Run and compare energy-optimal OCP simulations with and without ETCS braking curves.

    The script prints solver statistics and plots the resulting speed profiles,
    track limits, ETCS limits, and optional shooting nodes.
    """


    from mseetc.train import Train
    from mseetc.track import Track
    from mseetc.journey import Journey


    ####################################################################################################################
    ### Input
    directoryTrain = '../../trains'
    trainId = 'CH_Stadler_FLIRT_TPF'

    directoryTrack = '../../tracks'
    trackId = 'CH_StGallen_Wil'

    directoryJourney = '../../journeys'
    journeyId = 'CH_StGallen_Wil_Journey_01'
    journeySectionIdx = 0

    efficiencyMode = "static"  # perfect, static, dynamic
    withTrainLengthDependentValues = True

    plotWithShootingNodes = True
    plotWithOCPSpeedLimit = True
    ####################################################################################################################


    train = Train(config={'id':trainId}, pathJSON=directoryTrain)
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, efficiencyMode)

    track = Track(config={'id':trackId}, pathJSON=directoryTrack)

    if withTrainLengthDependentValues:

        track.updateTrainLengthDependentValues(train)

    journey = Journey(config={'id':journeyId}, sectionIdx=journeySectionIdx, pathJSON=directoryJourney)
    track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')


    ### non-ETCS-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, journey, opts)
    df, stats = solver.solve()

    printStats(df, stats, solver, train)

    # df.to_pickle("../data/StGallenWilTrajectory01.pkl")


    ### ETCS-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True, 'withEtcsBrakingCurves': True}

    solverEtcs = casadiSolver(train, track, journey, opts)
    dfEtcs, statsEtcs = solverEtcs.solve()

    printStats(dfEtcs, statsEtcs, solverEtcs, train)


    ### Plot Trajectory

    fig, ax = plt.subplots(figsize=(24, 12))

    x = track.speedLimits.index.to_numpy(dtype=float)
    v = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    x_plot = np.append(x, track.length)
    v_plot = np.append(v, v[-1])

    ax.step(x_plot/1000, v_plot*3.6, where="post", color="black", linestyle="-", label="Track Speed Limit")
    ax.plot(track.etcsPositions/1000, track.etcsVelocities*3.6, color="red", linestyle="-", label="ETCS Speed Limit")

    ax.plot(df["Position [m]"] / 1000, df["Velocity [m/s]"] * 3.6, linestyle="--", label="non-adjusted speed profile")
    ax.plot(dfEtcs["Position [m]"] / 1000, dfEtcs["Velocity [m/s]"] * 3.6, linestyle="--", label="ETCS-adjusted speed profile")

    if plotWithShootingNodes:

        for pos in dfEtcs["Position [m]"].to_numpy():

            ax.vlines(pos / 1000, 0, 400, color='red', linewidth=0.1)

    if plotWithOCPSpeedLimit:

        positions = solver.points.index.to_numpy()
        speedLimits = solver.points["Speed limit [m/s]"].to_numpy()
        ax.step(positions / 1000, speedLimits * 3.6, where="post", color="green", linestyle="-", linewidth=1,label="OCP Speed Limit")

    ax.set_title("Speed Profile Comparison")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, df["Position [m]"].max() / 1000)
    ax.set_ylim(0, v_plot.max() * 3.6 * 1.3)
    ax.figure.tight_layout()

    plt.show()

    costRatio = (statsEtcs["Cost"] - stats["Cost"]) / stats["Cost"]

    print(f"Cost increase with ETCS: {costRatio:.2%}")