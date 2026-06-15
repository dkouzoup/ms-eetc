import numpy as np
from matplotlib import pyplot as plt

from mseetc.ocp import casadiSolver
from simulations.sim_launcher import get_power_loss_function
from mseetc.track import Track
from mseetc.train import Train


if __name__ == '__main__':


    SBB_track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='')

    SBB_positions = SBB_track.gradients.index.values
    SBB_gradients = SBB_track.gradients["Gradient [permil]"].to_numpy()

    initial_altitude = SBB_track.altitude
    delta_s = np.diff(SBB_positions)
    delta_h = SBB_gradients[:-1] / 1000 * delta_s

    SBB_altitude = np.insert(initial_altitude + np.cumsum(delta_h),0, initial_altitude)

    Topo_track = Track(config={'id': 'CH_StGallen_Wil_Swisstopo'}, pathJSON='')

    Topo_positions = Topo_track.gradients.index.values
    Topo_gradients = Topo_track.gradients["Gradient [permil]"].to_numpy()

    initial_altitude = Topo_track.altitude
    delta_s = np.diff(Topo_positions)
    delta_h = Topo_gradients[:-1] / 1000 * delta_s

    Topo_altitude = np.insert(initial_altitude + np.cumsum(delta_h),0, initial_altitude)

    shift = 800 # 770


    ### Plot Altitude Comparison

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(SBB_positions / 1000, SBB_altitude, label="SBB")
    ax.plot((Topo_positions - shift) / 1000, Topo_altitude, label="Topo")
    ax.set_title("Altitude Comparison")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Altitude [m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, SBB_track.length / 1000)
    ax.figure.tight_layout()

    plt.show()


    ### Plot Speed Limit Comparison

    fig2, ax2 = plt.subplots(figsize=(16, 8))

    ax2.step(SBB_track.speedLimits.index.values / 1000, SBB_track.speedLimits["Speed limit [m/s]"].to_numpy()*3.6, label="SBB")
    ax2.step((Topo_track.speedLimits.index.values-shift) / 1000, Topo_track.speedLimits["Speed limit [m/s]"].to_numpy()*3.6, label="Topo")
    ax2.set_title("Altitude Comparison")
    ax2.set_xlabel("Position [km]")
    ax2.set_ylabel("Velocity [km/h]")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, SBB_track.length / 1000)
    ax2.figure.tight_layout()

    plt.show()


    ### Compute Energy Comparison

    # Timetable
    startPosition = 0           # [m]
    endPosition = 23000         # [m]
    duration = 23000/(80/3.6)   # [s]

    train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='../../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")
    opts = {'numIntervals': 1000, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 2}, 'energyOptimal': True}

    SBB_track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
    SBB_track.updateTrainLengthDependentValues(train)
    solver = casadiSolver(train, SBB_track, opts)
    dfSBB, statsSBB = solver.solve(duration)

    Topo_track.updateLimits(positionStart=startPosition + shift, positionEnd=endPosition + shift, unit='m')
    Topo_track.updateTrainLengthDependentValues(train)
    solver = casadiSolver(train, Topo_track, opts)
    dfTopo, statsTopo = solver.solve(duration)

    print(f"Cost SBB: {statsSBB['Cost']:.2f}")
    print(f"Cost Topo: {statsTopo['Cost']:.2f}")

    print(f"{abs(statsSBB['Cost'] - statsTopo['Cost']) / statsSBB['Cost'] * 100:.2f}%")


    ### Plot Trajectory

    fig3, ax3 = plt.subplots(figsize=(16, 8))

    ax3.plot(dfSBB["Position [m]"] / 1000, dfSBB["Velocity [m/s]"] * 3.6, label="SBB")
    ax3.plot(dfTopo["Position [m]"] / 1000, dfTopo["Velocity [m/s]"] * 3.6, label="Topo")
    ax3.set_title("Speed Profile Comparison")
    ax3.set_xlabel("Position [km]")
    ax3.set_ylabel("Velocity [km/h]")
    ax3.grid(True, which="both", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")
    ax3.set_xlim(0, dfSBB["Position [m]"].max() / 1000)
    ax3.figure.tight_layout()

    plt.show()


