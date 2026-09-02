import json
import numpy as np
import pandas as pd

from pathlib import Path

from mseetc.utils import checkTTOBenchVersion, convertUnit


class Journey():

    def __init__(self, config, sectionIdx=0, pathJSON=Path(__file__).parent.parent / 'journeys') -> None:
        """
        Constructor of Journey objects.

        Parameters
        ----------
        config : dict
            Journey configuration. Must contain the key 'id'.

        sectionIdx : int, optional
            Zero-based index of the journey section to select.

            A journey file may contain timing points for multiple journey sections.
            The section index selects one section by cropping the
            complete timing point list to the timing points between the
            corresponding departure and arrival stopping points.

            After cropping, time constraints are shifted so that the selected
            departure time becomes 0.

        pathJSON : str or pathlib.Path, optional
            Path to the directory containing the journey JSON files.
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

        self.sectionIdx = sectionIdx

        self.completeTimingPoints = self.readTimingPoints(data['timing points'])

        self.journeySectionBounds = self.getJourneySectionBounds()

        self.selectJourneySection(sectionIdx)


    def selectJourneySection(self, sectionIdx):
        """
        Select one journey section from the complete timing point list.

        The selected section is defined by the departure and arrival stopping
        points corresponding to the given zero-based section index.
        """

        self.sectionIdx = sectionIdx

        self.timingPoints = self.cropTimingPoints(sectionIdx)

        self.shiftTimeConstraints()

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

    def isStoppingPoint(self, point):

        return point["Lower speed constraint [m/s]"] == 0 and point["Upper speed constraint [m/s]"] == 0

    def getJourneySectionBounds(self):

        positions = self.completeTimingPoints.index.values

        departures = [0]
        arrivals = []

        for ii in range(1, len(positions)):

            if positions[ii] == positions[ii - 1]:

                arrivals.append(ii - 1)
                departures.append(ii)

        arrivals.append(len(positions) - 1)

        return list(zip(departures, arrivals))


    def cropTimingPoints(self, sectionIdx):

        if type(sectionIdx) is not int:

            raise ValueError("Journey section index must be an integer!")

        if sectionIdx < 0 or sectionIdx >= len(self.journeySectionBounds):

            raise ValueError("Journey section index out of range!")

        idxStart, idxEnd = self.journeySectionBounds[sectionIdx]

        return self.completeTimingPoints.iloc[idxStart:idxEnd + 1].copy()


    def getTimeOffset(self):

        firstPoint = self.timingPoints.iloc[0]

        tMin = firstPoint["Lower time constraint [s]"]
        tMax = firstPoint["Upper time constraint [s]"]

        if self.isSet(tMin):

            return tMin

        if self.isSet(tMax):

            return tMax

        return 0


    def shiftTimeConstraints(self):

        self.timeOffset = self.getTimeOffset()

        for key in ["Lower time constraint [s]", "Upper time constraint [s]"]:
            self.timingPoints[key] = self.timingPoints[key].apply(
                lambda value: value - self.timeOffset if self.isSet(value) else value
            )


    def checkFields(self):

        if len(self.timingPoints) < 2:

            raise ValueError("Journey section must contain at least two timing points!")

        positions = self.timingPoints.index.values

        if any(position < 0 or np.isinf(position) for position in positions):

            raise ValueError("Timing point positions must be positive finite numbers!")

        if any(pos2 <= pos1 for pos1, pos2 in zip(positions[:-1], positions[1:])):

            raise ValueError("Timing point positions in selected journey section must be strictly increasing!")

        firstPoint = self.timingPoints.iloc[0]
        lastPoint = self.timingPoints.iloc[-1]

        if not self.isStoppingPoint(firstPoint):

            raise ValueError("First timing point of selected journey section must have both speed constraints set to zero!")

        if not self.isStoppingPoint(lastPoint):

            raise ValueError("Last timing point of selected journey section must have both speed constraints set to zero!")

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


    def checkAssociatedTrack(self, track):
        """
        Check that a track matches the one this journey was defined on.

        The timing point positions are only meaningful on the associated track,
        so pairing a journey with a different track silently yields results for
        stopping points that are not where the journey says they are.
        """

        if self.associatedTrackID != track.id:

            raise ValueError("Journey '{}' is defined on track '{}', but track '{}' was given!".format(
                self.id, self.associatedTrackID, track.id))


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

