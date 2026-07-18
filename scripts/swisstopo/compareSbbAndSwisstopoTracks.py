import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.utils import get_power_loss_function
from mseetc.track import Track
from mseetc.train import Train


def computeTrackPath(curvatureDf, initialHeadingDeg=0, signFlipped=False):

    positions = curvatureDf.index.to_numpy(dtype=float)
    curvatures = curvatureDf["Curvature [1/m]"].to_numpy(dtype=float)

    eastings = np.zeros(len(positions))
    northings = np.zeros(len(positions))
    headings = np.zeros(len(positions))

    headings[0] = np.radians(initialHeadingDeg)

    for i in range(len(positions) - 1):

        ds = positions[i + 1] - positions[i]
        curvature = -curvatures[i] if signFlipped else curvatures[i]

        if abs(curvature) > 1e-12:

            headingEnd = headings[i] + curvature * ds

            eastings[i + 1] = eastings[i] + (np.sin(headingEnd) - np.sin(headings[i])) / curvature
            northings[i + 1] = northings[i] - (np.cos(headingEnd) - np.cos(headings[i])) / curvature

            headings[i + 1] = headingEnd

        else:

            eastings[i + 1] = eastings[i] + ds * np.cos(headings[i])
            northings[i + 1] = northings[i] + ds * np.sin(headings[i])

            headings[i + 1] = headings[i]

    return eastings, northings


if __name__ == '__main__':

    """
    Compare SBB reference and Swisstopo track data for the St. Gallen–Wil route.

    The script compares altitude profiles and speed limits for both track sources.
    It further solves an energy-optimal OCP for each dataset, and reports the resulting energy-cost difference.
    It also plots the resulting optimized speed profiles for direct comparison.
    """


    ### SBB Data

    SBB_track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='../../tracks')  # Reference track
    zeroProfile = SBB_track.curvatures.iloc[[0]].copy()
    zeroProfile.loc[:, :] = 0.0
    SBB_track.curvatures = zeroProfile

    SBB_positions = SBB_track.gradients.index.values
    SBB_gradients = SBB_track.gradients["Gradient [permil]"].to_numpy()

    initial_altitude = SBB_track.altitude
    delta_s = np.diff(SBB_positions)
    delta_h = SBB_gradients[:-1] / 1000 * delta_s

    SBB_altitude = np.insert(initial_altitude + np.cumsum(delta_h),0, initial_altitude)


    ### Swisstopo Data

    Topo_track = Track(config={'id': 'CH_StGallen_Wil_Swisstopo'}, pathJSON='../../tracks/swisstopo')
    zeroProfile = Topo_track.curvatures.iloc[[0]].copy()
    zeroProfile.loc[:, :] = 0.0
    Topo_track.curvatures = zeroProfile

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

    journey_SBB = Journey(config={'id': 'CH_StGallen_Wil_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    SBB_track.updateLimits(positionStart=journey_SBB.positionStart, positionEnd=journey_SBB.positionEnd, unit='m')

    opts = {'numIntervals': 1000, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 2}, 'energyOptimal': True}

    solver = casadiSolver(train, SBB_track, journey_SBB, opts)
    dfSBB, statsSBB = solver.solve()


    ### OCP Swisstopo

    Topo_track.updateTrainLengthDependentValues(train)

    journey_Topo = Journey(config={'id': 'CH_StGallen_Wil_Swisstopo_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    Topo_track.updateLimits(positionStart=journey_Topo.positionStart, positionEnd=journey_Topo.positionEnd, unit='m')

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


    ### Curvature Analysis

    SBB_track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='../../tracks')  # Reference track
    zeroProfile = SBB_track.gradients.iloc[[0]].copy()
    zeroProfile.loc[:, :] = 0.0
    SBB_track.gradients = zeroProfile

    SBB_positions = SBB_track.curvatures.index.values
    SBB_curvatures = SBB_track.curvatures["Curvature [1/m]"].to_numpy()

    Topo_track = Track(config={'id': 'CH_StGallen_Wil_Swisstopo'}, pathJSON='../../tracks/swisstopo')
    zeroProfile = Topo_track.gradients.iloc[[0]].copy()
    zeroProfile.loc[:, :] = 0.0
    Topo_track.gradients = zeroProfile

    Topo_positions = Topo_track.curvatures.index.values
    Topo_curvatures = Topo_track.curvatures["Curvature [1/m]"].to_numpy()

    ### Plot Curvature Comparison

    fig4, ax = plt.subplots(figsize=(16, 8))

    ax.step(SBB_positions / 1000, SBB_curvatures, where='post', label="SBB")
    ax.step((Topo_positions - shift) / 1000, Topo_curvatures, where='post', label="Topo")
    ax.set_title("Fig 4.: Curvature Comparison")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Curvature [1/m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, SBB_track.length / 1000)
    ax.figure.tight_layout()

    plt.show()


    ### Plot track path

    csvInputfilePath = r"C:\Users\rolan\Documents\ms-eetc-innocheque\tracks\swisstopo\Track_StGallen_Wil.csv"
    df = pd.read_csv(csvInputfilePath, na_values=["<null>", "null", ""])
    eastingsOg = df["Easting"].to_numpy(dtype=float)
    northingsOg = df["Northing"].to_numpy(dtype=float)

    eastingsOg = eastingsOg - eastingsOg[0]
    northingsOg = northingsOg - northingsOg[0]

    fig5, ax = plt.subplots(figsize=(16, 8))

    ax.plot(eastingsOg, northingsOg, label="OG")
    eastings, northings = computeTrackPath(SBB_track.curvaturesSigned, initialHeadingDeg=-132, signFlipped=True)
    ax.plot(eastings, northings, label="SBB")
    eastings, northings = computeTrackPath(Topo_track.curvaturesSigned, initialHeadingDeg=-127)
    ax.plot(eastings, northings, label="Topo")

    ax.set_title("Fig 5.: Compare Track Path")
    ax.set_xlabel("Relative Easting [m]")
    ax.set_ylabel("Relative Northing [m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.axis("equal")
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    plt.show()


    ### OCP SBB Track

    SBB_track.updateTrainLengthDependentValues(train)

    journey_SBB = Journey(config={'id': 'CH_StGallen_Wil_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    SBB_track.updateLimits(positionStart=journey_SBB.positionStart, positionEnd=journey_SBB.positionEnd, unit='m')

    opts = {'numIntervals': 1000, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 2}, 'energyOptimal': True}

    solver = casadiSolver(train, SBB_track, journey_SBB, opts)
    dfSBB, statsSBB = solver.solve()


    ### OCP Swisstopo

    Topo_track.updateTrainLengthDependentValues(train)

    journey_Topo = Journey(config={'id': 'CH_StGallen_Wil_Swisstopo_Journey_SwisstopoComparison'}, sectionIdx=0, pathJSON='../../journeys')
    Topo_track.updateLimits(positionStart=journey_Topo.positionStart, positionEnd=journey_Topo.positionEnd, unit='m')

    solver = casadiSolver(train, Topo_track, journey_Topo, opts)
    dfTopo, statsTopo = solver.solve()


    ### Print Stats

    print(f"Cost SBB: {statsSBB['Cost']:.2f}")
    print(f"Cost Topo: {statsTopo['Cost']:.2f}")

    print(f"{abs(statsSBB['Cost'] - statsTopo['Cost']) / statsSBB['Cost'] * 100:.2f}%")