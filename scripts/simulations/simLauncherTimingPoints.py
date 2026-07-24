import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from mseetc.ocp import casadiSolver
from mseetc.utils import isSet, get_power_loss_function, printStats


def plotTpConstraintsTime(ax, journey):

    posShift = journey.positionEnd - journey.positionStart
    timeShift = journey.terminalTime - journey.initialTime

    for pos, tP in journey.timingPoints.iterrows():

        lowerTime = tP["Lower time constraint [s]"]
        upperTime = tP["Upper time constraint [s]"]

        if isSet(lowerTime):

            ax.add_patch(Rectangle((pos/1000, lowerTime-timeShift), posShift/1000, timeShift, facecolor='lightgrey', edgecolor='none'))

        if isSet(upperTime):

            ax.add_patch(Rectangle(((pos-posShift) / 1000, upperTime), posShift / 1000, timeShift, facecolor='lightgrey', edgecolor='none'))


def plotTpConstraintsVelocity(ax, journey):

    maxVel = 400  # [km/h]

    for pos, tP in journey.timingPoints.iterrows():

        lowerVel = tP["Lower speed constraint [m/s]"]
        upperVel = tP["Upper speed constraint [m/s]"]

        if isSet(lowerVel):

            ax.vlines(pos/1000, 0, lowerVel*3.6, color='red', linewidth=1)

        if isSet(upperVel):

            ax.vlines(pos/1000, upperVel*3.6, maxVel, color='red', linewidth=1)


if __name__ == '__main__':

    """
    Run an energy-optimal OCP simulation subject to journey timing-point constraints.

    The script prints solver statistics
    and plots the position-time profile with the specified timing constraints,
    as well as the position-velocity profile with timing-point velocity constraints and track speed limits.
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
    journeyId = 'CH_StGallen_Wil_Journey_02'
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


    ### Solve OCP problem

    opts = {'numIntervals':800, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, journey, opts)
    df, stats = solver.solve()

    printStats(df, stats, solver, train)


    ### Plot position-time-profile

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot((df["Position [m]"]+journey.positionStart)/ 1000, df.index.to_numpy(), linestyle="--")
    plotTpConstraintsTime(ax, journey)

    ax.set_title("Fig 1.: Position - Time - Profile")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Time [s]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlim(journey.positionStart / 1000 - 1, journey.positionEnd / 1000 + 1)
    ax.set_ylim(journey.initialTime - 100, journey.terminalTime + 100)
    ax.figure.tight_layout()

    plt.show()


    ### Plot position-velocity-profile

    x = track.speedLimits.index.to_numpy(dtype=float)
    v = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    x_plot = np.append(x, track.length)
    v_plot = np.append(v, v[-1])

    vMax = max(v.max(), df["Velocity [m/s]"].max()) * 3.6 + 20

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot((df["Position [m]"]+journey.positionStart) / 1000, df["Velocity [m/s]"] * 3.6, linestyle="--")
    ax.step((x_plot+journey.positionStart) / 1000, v_plot * 3.6, where="post", color="black", linestyle="-", label="Track Speed Limit")
    plotTpConstraintsVelocity(ax, journey)

    ax.set_title("Fig 2.: Position - Velocity - Profile")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xlim(journey.positionStart / 1000 - 1, journey.positionEnd / 1000 + 1)
    ax.set_ylim(0, vMax)
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    plt.show()