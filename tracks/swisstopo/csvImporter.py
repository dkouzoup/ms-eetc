import json
from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == '__main__':


    ### Read CSV

    filePath = r"C:\Users\rolan\Documents\ms-eetc-innocheque\tracks\swisstopo\Track_StGallen_Wil.csv"

    df = pd.read_csv(filePath,na_values=["<null>", "null", ""])

    print(df.head())
    print(df.dtypes)


    ### Speed limits

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


    ### Parse to Json

    track_id = "CH_StGallen_Wil_Swisstopo"
    author = "Roland Staerk"

    name = "CH_StGallen_Wil_Swisstopo"

    output_dir = Path(r"C:\Users\rolan\Documents\ms-eetc-innocheque\tracks\swisstopo")
    output_path = output_dir / f"{name}.json"

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

    track_data = {
        "metadata": {
            "id": track_id,
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
        }
    }


    ### Save Json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(track_data, f, indent=4)