import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SPACING = 10  # [m]
DEFAULT_SMOOTHINGWINDOW = 11  # [m]

def computeCurvature(df, spacing=DEFAULT_SPACING, smoothingWindow=DEFAULT_SMOOTHINGWINDOW):

    measuredPositions = df["Total_Distance"].to_numpy(dtype=float)
    eastings = df["Easting"].to_numpy(dtype=float)
    northings = df["Northing"].to_numpy(dtype=float)


    ### Interpolate to equally spaced positions

    positions = np.arange(measuredPositions[0], measuredPositions[-1] + spacing, spacing)

    eastings = np.interp(positions, measuredPositions, eastings)
    northings = np.interp(positions, measuredPositions, northings)


    ### Smooth coordinates

    eastings = pd.Series(eastings).rolling(window=smoothingWindow, center=True, min_periods=1).mean().to_numpy()
    northings = pd.Series(northings).rolling(window=smoothingWindow, center=True, min_periods=1).mean().to_numpy()


    ### Compute curvature

    dx = np.gradient(eastings, positions)
    dy = np.gradient(northings, positions)

    ddx = np.gradient(dx, positions)
    ddy = np.gradient(dy, positions)

    denominator = (dx**2 + dy**2)**1.5

    curvature = np.divide(dx * ddy - dy * ddx, denominator, out=np.zeros_like(denominator), where=denominator > 1e-12)


    ### Interpolate curvature to measured positions

    curvatures = np.interp(measuredPositions, positions, curvature)

    return curvatures


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

    curvatures = computeCurvature(df)
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
        for pos, radiusStart, radiusEnd in zip(positions, radii, radii)
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