import json
import numpy as np
import pandas as pd

from pathlib import Path

from mseetc.utils import checkTTOBenchVersion, convertUnit


class Journey():

    def __init__(self, config, pathJSON=Path(__file__).parent.parent / 'journeys') -> None:
        """
        Constructor of Journey objects.
        """

        # check config
        if not isinstance(config, dict):

            raise ValueError("Journey configuration should be provided as a dictionary!")

        if 'id' not in config:

            raise ValueError("Journey ID must be specified in configuration!")

        # open json file
        filename = Path(pathJSON) / (config['id'] + '.json')

        with open(filename) as file:

            data = json.load(file)

        checkTTOBenchVersion(data, ['1.5'])

        # read data
        self.id = data['metadata']['id']

        self.associatedTrackID = data['metadata']['associated track']

        self.timingPoints = self.readTimingPoints(data['timing points'])

        self.checkFields()

        self.computeStartAndEndPoints()

        self.computeInitialAndTerminalStates()

    def readTimingPoints(self, timingPoints):

        units = timingPoints['units']

        values = {
            "Position [m]": [],
            "Lower time constraint [s]": [],
            "Upper time constraint [s]": [],
            "Lower speed constraint [m/s]": [],
            "Upper speed constraint [m/s]": []
        }

        for point in timingPoints['values']:
            pos, tMin, tMax, vMin, vMax = point

            values["Position [m]"].append(convertUnit(pos, units['position']))
            values["Lower time constraint [s]"].append(self.convertConstraint(tMin, units['lower time constraint']))
            values["Upper time constraint [s]"].append(self.convertConstraint(tMax, units['upper time constraint']))
            values["Lower speed constraint [m/s]"].append(self.convertConstraint(vMin, units['lower speed constraint']))
            values["Upper speed constraint [m/s]"].append(self.convertConstraint(vMax, units['upper speed constraint']))

        df = pd.DataFrame(values)
        df = df.set_index("Position [m]")

        return df


    def convertConstraint(self, value, unit):

        if value is None:

            return None

        return convertUnit(value, unit)


    def isSet(self, value):

        return value is not None and not pd.isna(value)


    def checkFields(self):

        if len(self.timingPoints) < 2:

            raise ValueError("Journey must contain at least two timing points!")

        positions = self.timingPoints.index.values

        if any(position < 0 or np.isinf(position) for position in positions):

            raise ValueError("Timing point positions must be positive finite numbers!")

        if any(pos2 <= pos1 for pos1, pos2 in zip(positions[:-1], positions[1:])):

            raise ValueError("Timing point positions must be strictly increasing!")

        firstPoint = self.timingPoints.iloc[0]
        lastPoint = self.timingPoints.iloc[-1]

        if firstPoint["Lower speed constraint [m/s]"] != 0 or firstPoint["Upper speed constraint [m/s]"] != 0:

            raise ValueError("First timing point must have both speed constraints set to zero!")

        if lastPoint["Lower speed constraint [m/s]"] != 0 or lastPoint["Upper speed constraint [m/s]"] != 0:

            raise ValueError("Last timing point must have both speed constraints set to zero!")

        for ii, point in self.timingPoints.iterrows():

            tMin = point["Lower time constraint [s]"]
            tMax = point["Upper time constraint [s]"]
            vMin = point["Lower speed constraint [m/s]"]
            vMax = point["Upper speed constraint [m/s]"]

            if self.isSet(tMin) and self.isSet(tMax) and tMin > tMax:

                raise ValueError("Lower time constraint must be smaller than or equal to upper time constraint!")

            if self.isSet(vMin) and self.isSet(vMax) and vMin > vMax:

                raise ValueError("Lower speed constraint must be smaller than or equal to upper speed constraint!")

            if any(self.isSet(value) and (value < 0 or np.isinf(value)) for value in [tMin, tMax, vMin, vMax]):

                raise ValueError("Timing point constraints must be positive finite numbers or None!")


    def computeStartAndEndPoints(self):

        self.positionStart = self.timingPoints.index.values[0]

        self.positionEnd = self.timingPoints.index.values[-1]


    def computeInitialAndTerminalStates(self):

        self.initialTime = self.timingPoints.iloc[0]["Lower time constraint [s]"]

        self.terminalTime = self.timingPoints.iloc[-1]["Upper time constraint [s]"]

        self.initialVelocity = self.timingPoints.iloc[0]["Lower speed constraint [m/s]"]

        self.terminalVelocity = self.timingPoints.iloc[-1]["Upper speed constraint [m/s]"]


    def getTimingPoint(self, position):

        idx = np.where(np.isclose(self.timingPoints.index.values, position))[0]

        if len(idx) == 0:
            return None

        return self.timingPoints.iloc[idx[0]]

