import numpy as np
from matplotlib import pyplot as plt

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.utils import get_power_loss_function
from mseetc.track import Track
from mseetc.train import Train


if __name__ == '__main__':

    """
    Compare SBB reference and Swisstopo track data for the St. Gallen–Wil route.

    The script compares altitude profiles and speed limits for both track sources.
    It further solves an energy-optimal OCP for each dataset, and reports the resulting energy-cost difference.
    It also plots the resulting optimized speed profiles for direct comparison.
    """


    ### SBB Data

    SBB_track = Track(config={'id': 'CH_StGallen_Wil_Reference'}, pathJSON='../../tracks/swisstopo')  # Reference track with no curvature data

    SBB_positions = SBB_track.gradients.index.values
    SBB_gradients = SBB_track.gradients["Gradient [permil]"].to_numpy()

    initial_altitude = SBB_track.altitude
    delta_s = np.diff(SBB_positions)
    delta_h = SBB_gradients[:-1] / 1000 * delta_s

    SBB_altitude = np.insert(initial_altitude + np.cumsum(delta_h),0, initial_altitude)


    ### Swisstopo Data

    Topo_track = Track(config={'id': 'CH_StGallen_Wil_Swisstopo'}, pathJSON='../../tracks/swisstopo')

    Topo_positions = Topo_track.gradients.index.values
    Topo_gradients = Topo_track.gradients["Gradient [permil]"].to_numpy()

    initial_altitude = Topo_track.altitude
    delta_s = np.diff(Topo_positions)
    delta_h = Topo_gradients[:-1] / 1000 * delta_s

    Topo_altitude = np.insert(initial_altitude + np.cumsum(delta_h),0, initial_altitude)

    shift = 800


    ### Plot Altitude Comparison

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(SBB_positions / 1000, SBB_altitude, label="SBB")
    ax.plot((Topo_positions - shift) / 1000, Topo_altitude, label="Topo")
    ax.set_title("Fig 1.: Altitude Comparison")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Altitude [m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, SBB_track.length / 1000)
    ax.figure.tight_layout()

    plt.show()


    ### Plot Speed Limit Comparison

    fig2, ax2 = plt.subplots(figsize=(16, 8))

    ax2.step(SBB_track.speedLimits.index.values / 1000, SBB_track.speedLimits["Speed limit [m/s]"].to_numpy()*3.6, where="post", label="SBB")
    ax2.step((Topo_track.speedLimits.index.values-shift) / 1000, Topo_track.speedLimits["Speed limit [m/s]"].to_numpy()*3.6, where="post", label="Topo")
    ax2.set_title("Fig 2.: Speedlimit Comparison")
    ax2.set_xlabel("Position [km]")
    ax2.set_ylabel("Velocity [km/h]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, SBB_track.length / 1000)
    ax2.figure.tight_layout()

    plt.show()


    ### Compute Energy Comparison

    train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='../../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")


    ### OCP SBB Track

    SBB_track.updateTrainLengthDependentValues(train)

    journey_SBB = Journey(config={'id': 'CH_StGallen_Wil_Reference_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    SBB_track.updateLimits(positionStart=journey_SBB.positionStart, positionEnd=journey_SBB.positionEnd, unit='m')

    opts = {'numIntervals': 1000, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 2}, 'energyOptimal': True}

    solver = casadiSolver(train, SBB_track, journey_SBB, opts)
    dfSBB, statsSBB = solver.solve()


    ### OCP Swisstopo

    Topo_track.updateTrainLengthDependentValues(train)

    journey_Topo = Journey(config={'id': 'CH_StGallen_Wil_Swisstopo_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    Topo_track.updateLimits(positionStart=journey_Topo.positionStart, positionEnd=journey_Topo.positionEnd, unit='m')

    # Topo_track.updateLimits(positionStart=startPosition + shift, positionEnd=endPosition + shift, unit='m')

    solver = casadiSolver(train, Topo_track, journey_Topo, opts)
    dfTopo, statsTopo = solver.solve()


    ### Print Stats

    print(f"Cost SBB: {statsSBB['Cost']:.2f}")
    print(f"Cost Topo: {statsTopo['Cost']:.2f}")

    print(f"{abs(statsSBB['Cost'] - statsTopo['Cost']) / statsSBB['Cost'] * 100:.2f}%")


    ### Plot Trajectory

    fig3, ax3 = plt.subplots(figsize=(16, 8))

    ax3.plot(dfSBB["Position [m]"] / 1000, dfSBB["Velocity [m/s]"] * 3.6, label="SBB")
    ax3.plot(dfTopo["Position [m]"] / 1000, dfTopo["Velocity [m/s]"] * 3.6, label="Topo")
    ax3.set_title("Fig 3.: Speed Profile Comparison")
    ax3.set_xlabel("Position [km]")
    ax3.set_ylabel("Velocity [km/h]")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")
    ax3.set_xlim(0, dfSBB["Position [m]"].max() / 1000)
    ax3.figure.tight_layout()

    plt.show()


