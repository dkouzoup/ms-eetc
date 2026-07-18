import json
from pathlib import Path
from scipy.interpolate import splprep, splev   # todo: problems with np version!

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
    
    Only make changes in the "Input" section of this script.
    
    How to get track data from Swisstopo:
    
    1.  a)  Create an empty CSV File. This will eventually contain the whole track data.
        b)  In the first row, set these column titles:
            "Total_Distance", "Distance", "Altitude", "Easting", "Northing", "Longitude", "Latitude", "V_max", "Station"
    2.  a)  Go to: https://map.geo.admin.ch
        b)  In the searchbar, search for: "Railway swissTLM3D", select it under "add map".
            It should now be visible. On the left, it appears as an active map under "Maps displayed".
    3.  a)  Go to: https://openrailwaymap.org/
        b)  On the left, select Max speed.
    4.  Repeat step 4 for every segment of your track.
        a)  In Swisstopo, click on the track segment you want to export.
            A pop-up window will open. Then click on "Display profile".
            A second pop-up window appears displaying the altitude profile of the selected track segment.
            Click on the download symbol ("Get data as CSV file").
        b)  Open the downloaded CSV file in Excel.
            Because the altitude profile does not account for tunnels and bridges, one needs to manually add them to the altitude profile.
            For this, look at the displayed altitude profile in Swisstopo and correct the corresponding entries in Excel.
        c)  Copy all the data (including the modified altitude data) into the master CSV file below the last data entries there.
            Based on the "Distance" column of the new data, extend "Total_Distance" which serves as a continuous track distance measure. 
        d)  From Openrailway map, find all speed limit changes for the current track section and localize the changing points in Swisstopo.
            By hovering with the mouse over the altitude profile, one can precisely locate the speed limit changing points.
            Add the new speedlimit to the corresponding row in the master Excel.
            Not all entries in the "V_max" column need to be filled. Only enter values where the speed limit changes
        e)  If there is any station in the current track section, find the position of the middle point of the station on Swisstopo.
            Add the station name to the master Excel at the corresponding position entry.
    5.  Once all track segments have been completed, save the data as a CSV File.
        Set the corresponding paths in this script and then run it.
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
        "V_max",
        "Station",
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