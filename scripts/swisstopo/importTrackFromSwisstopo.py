import json
from pathlib import Path
# from scipy.interpolate import splprep, splev   # todo: problems with np version!

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

DEFAULT_SPACING = 100  # [m]


def computeCurvature(df, spacing=DEFAULT_SPACING):

    positions  = df["Total_Distance"].to_numpy(dtype=float)
    eastings = df["Easting"].to_numpy(dtype=float)
    northings = df["Northing"].to_numpy(dtype=float)

    eastings = eastings - eastings[0]
    northings = northings - northings[0]

    assert np.all(np.diff(positions) > 0), (
        "Column 'Total_Distance' must be strictly increasing."
    )

    ### Fit smoothing spline

    plotSpacing = 1
    spline, _ = splprep([eastings, northings], u=positions, k=3, s=1000)

    ### Evaluate smoothing spline

    fittedPositions = np.arange(positions[0], positions[-1], plotSpacing)

    fittedPositions = np.append(fittedPositions, positions[-1])

    fittedEastings, fittedNorthings = splev(fittedPositions,spline)

    fittedEastings = np.asarray(fittedEastings)
    fittedNorthings = np.asarray(fittedNorthings)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(eastings, northings, label="OG")
    ax.plot(fittedEastings, fittedNorthings, label="Fitted")
    ax.set_title("Fig 1.: Visualize Track Path")
    ax.set_xlabel("Relative Easting [m]")
    ax.set_ylabel("Relative Northing [m]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.axis("equal")
    ax.legend(loc="upper right")
    ax.figure.tight_layout()
    plt.show()

    ### Define piecewise-constant curvature intervals

    curvatureBoundaries = np.arange(positions[0], positions[-1], spacing)
    curvatureBoundaries = np.append(curvatureBoundaries, positions[-1])

    ### Compute spline headings at interval boundaries

    dx, dy = splev(curvatureBoundaries, spline, der=1)

    headings = np.arctan2(dy, dx)
    headings = np.unwrap(headings)

    ### Compute piecewise-constant curvature

    ds = np.diff(curvatureBoundaries)
    deltaHeading = np.diff(headings)

    curvaturePositions = curvatureBoundaries[:-1]
    curvatures = deltaHeading / ds

    return curvaturePositions, curvatures


if __name__ == '__main__':

    """
    Convert Swisstopo CSV track data into a TTOBench-compatible JSON file.

    The script validates the required columns and initial speed limit,
    extracts the speed profile, computes smoothed gradients from altitude data,
    and saves the resulting track definition json.
    """


    ####################################################################################################################
    ### Input
    csvInputfilePath = r"C:\Users\rolan\Documents\ms-eetc-innocheque\tracks\swisstopo\Track_StGallen_Wil.csv"

    outputDirectory = Path(r"C:\Users\rolan\Documents\ms-eetc-innocheque\tracks\swisstopo")
    outputTrackId = "CH_StGallen_Wil_Swisstopo"
    author = "Roland Staerk"

    ####################################################################################################################


    ### Read CSV

    df = pd.read_csv(csvInputfilePath, na_values=["<null>", "null", ""])

    print(df.head())
    print(df.dtypes)

    requiredColumns = {
        "Total_Distance",
        "Distance",
        "Altitude",
        "Easting",
        "Northing",
        "Longitude",
        "Latitude",
        "V_max"
    }

    missingColumns = requiredColumns - set(df.columns)

    assert not missingColumns, (
        f"Input CSV is missing the following required columns: "
        f"{sorted(missingColumns)}"
    )


    ### Speed limits

    assert pd.notna(df["V_max"].iloc[0]) and df["V_max"].iloc[0] > 0, (
        "The first entry in column 'V_max' must be a valid value greater than 0."
    )

    speedProfile = df.loc[
        df["V_max"].notna(),
        ["Total_Distance", "V_max"]
    ].copy()


    ### Gradients

    totalDistance = df["Total_Distance"]
    altitude = df["Altitude"]

    window_size = 7
    altitude = altitude.rolling(window=window_size, center=True, min_periods=1).mean()

    spacing = 50  # [m]

    positions = np.arange(0, totalDistance.max(), spacing)
    altitude_interp = np.interp(positions, totalDistance, altitude)

    gradientPerMille = np.insert(1000 * np.diff(altitude_interp) / np.diff(positions),0,0 )
    gradientPerMille = np.round(gradientPerMille, 1)


    ### Curvature

    curvaturePositions, curvatures = computeCurvature(df)
    validCurvature = np.abs(curvatures) > 1e-12

    radii = np.full(curvatures.shape, "infinity", dtype=object)
    radii[validCurvature] = 1 / curvatures[validCurvature]


    ### Parse to Json

    output_path = outputDirectory / f"{outputTrackId}.json"

    stops = [
        0.0,
        float(totalDistance.iloc[-1])
    ]

    speed_limits = [
        [float(pos), float(vmax)]
        for pos, vmax in zip(
            speedProfile["Total_Distance"],
            speedProfile["V_max"]
        )
    ]

    gradients = [
        [float(pos), float(grad)]
        for pos, grad in zip(positions, gradientPerMille)
    ]

    curvatures = [
        [float(pos), float(radiusStart) if radiusStart != "infinity" else "infinity", float(radiusEnd) if radiusEnd != "infinity" else "infinity"]
        for pos, radiusStart, radiusEnd in zip(curvaturePositions, radii, radii)
    ]

    track_data = {
        "metadata": {
            "id": outputTrackId,
            "created by": author,
            "library version": "TTOBench v1.4",
            "license": "BSD 2-Clause License"
        },
        "altitude": {
            "unit": "m",
            "value": float(altitude_interp[0])
        },
        "stops": {
            "unit": "m",
            "values": stops
        },
        "speed limits": {
            "units": {
                "position": "m",
                "velocity": "km/h"
            },
            "values": speed_limits
        },
        "gradients": {
            "units": {
                "position": "m",
                "slope": "permil"
            },
            "values": gradients
        },
        "curvatures": {
            "units": {
                "position": "m",
                "radius at start": "m",
                "radius at end": "m"
            },
            "values": curvatures
        }
    }


    ### Save Json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=4)