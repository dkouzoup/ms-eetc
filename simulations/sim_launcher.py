import numpy as np
from matplotlib import pyplot as plt

from mseetc.efficiency import totalLossesFunction
from mseetc.etcs import getEtcsSpeedLimits
from mseetc.ocp import casadiSolver


def get_power_loss_function(train, mode="perfect",* ,auxiliaries: float = 27_000, eta_gear: float = 0.96):

    if mode == "perfect":

        return lambda f, v: 0

    elif mode == "static":

        return lambda f, v: (f>0)*f*v*(1-train.etaTraction)/train.etaTraction - (f<0)*f*v*(1-train.etaRgBrake)

    elif mode == "dynamic":

        return totalLossesFunction(train, auxiliaries=auxiliaries, etaGear=eta_gear)

    else:

        raise ValueError("mode must be one of: 'perfect', 'static', 'dynamic'")


def printStats(df, stats, solver, train):

    if df is not None:

        print("")
        print("Objective value = {:.2f} {}".format(stats['Cost'], 'kWh' if solver.opts.energyOptimal else 's'))
        print("")
        print("Maximum acceleration: {:5.2f}, with bound {}".format(df.max()['Acceleration [m/s^2]'], train.accMax if train.accMax is not None else 'None'))
        print("Maximum deceleration: {:5.2f}, with bound {}".format(df.min()['Acceleration [m/s^2]'],train.accMin if train.accMin is not None else 'None'))

    else:

        print("Solver failed!")


def plotShootingNodes(ax, positions):

    for pos in positions:

        ax.vlines(pos / 1000, 0, 400, color='red', linewidth=0.1)


def plotOCPSpeedLimit(ax, solver):

    positions = solver.points.index.to_numpy()
    speedLimits = solver.points["Speed limit [m/s]"].to_numpy()


    ax.step(positions/1000, speedLimits*3.6, where="post", color="green", linestyle="-", linewidth=1, label="OCP Speed Limit")


if __name__ == '__main__':

    from mseetc.train import Train
    from mseetc.track import Track
    from mseetc.journey import Journey


    train = Train(config={'id':'CH_Stadler_FLIRT_TPF'}, pathJSON='../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    # track = Track(config={'id':'00_var_speed_limit_quick_change'}, pathJSON='../tracks')
    track = Track(config={'id':'CH_StGallen_Wil'}, pathJSON='../tracks')
    track.updateTrainLengthDependentValues(train)

    journey = Journey(config={'id':'CH_StGallen_Wil_Journey_01'}, pathJSON='../journeys')
    track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

    # non-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':5}, 'energyOptimal':True}

    solver = casadiSolver(train, track, journey, opts)
    df, stats = solver.solve(journey)

    printStats(df, stats, solver, train)


    # ETCS-adjusted speed profile
    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':5}, 'energyOptimal':True, 'withEtcsBrakingCurves': True}

    solverEtcs = casadiSolver(train, track, journey, opts)
    dfEtcs, statsEtcs = solverEtcs.solve(journey)

    printStats(dfEtcs, statsEtcs, solverEtcs, train)


    ### Plot Trajectory

    withShootingNodes = True
    withOCPSpeedLimit = True

    fig, ax = plt.subplots(figsize=(24, 12))

    x = track.speedLimits.index.to_numpy(dtype=float)
    v = track.speedLimits["Speed limit [m/s]"].to_numpy(dtype=float)
    x_plot = np.append(x, track.length)
    v_plot = np.append(v, v[-1])

    ax.step(x_plot/1000, v_plot*3.6, where="post", color="black", linestyle="-", label="Track Speed Limit")
    ax.plot(track.etcsPositions/1000, track.etcsVelocities*3.6, color="red", linestyle="-", label="ETCS Speed Limit")

    ax.plot(df["Position [m]"] / 1000, df["Velocity [m/s]"] * 3.6, linestyle="--", label="non-adjusted speed profile")
    ax.plot(dfEtcs["Position [m]"] / 1000, dfEtcs["Velocity [m/s]"] * 3.6, linestyle="--", label="ETCS-adjusted speed profile")

    if withShootingNodes:

        plotShootingNodes(ax, dfEtcs["Position [m]"].to_numpy())

    if withOCPSpeedLimit:

        plotOCPSpeedLimit(ax, solverEtcs)

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