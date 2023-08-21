import os
import json
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def importTuples(tuples, xLabel, yLabels):
    """
    Convert list of tuples (or lists) into pandas dataframe.
    """

    if not isinstance(yLabels, list):

        yLabels = [yLabels]

    if not isinstance(tuples, list):

        raise ValueError("Input must be a list (of tuples or lists)!")

    elementOk = lambda x: (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 1 + len(yLabels)

    if not all([elementOk(tup) for tup in tuples]):

        raise ValueError("Error in list!")

    index = np.array([tup[0] for tup in tuples])

    if any(index < 0):

        raise ValueError("Position data cannot be negative!")

    if any(np.diff(index) <= 0):

        raise ValueError("Position data must monotonically increase!")

    df = pd.DataFrame({xLabel:index}).set_index(xLabel)

    for ii, yLabel in enumerate(yLabels):

        data = [float(tup[1+ii]) for tup in tuples]
        df[yLabel] = data

    return df


def convertUnit(value, unit):
    """
    Convert from any known unit to internally used unit.
    """

    if unit in {'m', 'm/s', 'permil'}:

        pass

    elif unit == 'km':

        value /= 1000

    elif unit == 'km/h':

        value /= 3.6

    else:

        raise ValueError("Unknown unit!")

    return value


def checkDataFrame(df, trackLength):
    """
    Check validity of initial and end position in pandas dataframe.
    """

    if df.index[0] != 0:

        raise ValueError("Error in '{}': First track section must start at 0 m (beginning of track)!".format(df.columns[0]))

    if df.index[-1] > trackLength:

        raise ValueError("Error in '{}': Last track section must start before {} m (end of track)!".format(df.columns[0], trackLength))

    return True


def computeAltitude(gradients, length, altitudeStart=0):
    """
    Calculate altitude profile from gradients.
    """

    position = np.append(gradients.index.values, length)
    altitude = np.array([altitudeStart])

    for ii in range(1, len(gradients)+1):

        posStart = gradients.index[ii-1]
        posEnd = gradients.index[ii] if ii < len(gradients) else length
        gradient = gradients.iloc[ii-1][0] if isinstance(gradients, pd.DataFrame) else gradients.iloc[ii-1]
        height = (posEnd - posStart)*(gradient/1e3)
        altitude = np.append(altitude, altitude[-1] + height)

    df = pd.DataFrame({gradients.index.name:position, 'Altitude [m]':altitude}).set_index(gradients.index.name)

    return df


def computeDiscretizationPoints(track, numIntervals):
    """
    Compute the space discretization points based on track characteristics and horizon length.
    """

    df1 = track.mergeDataFrames()

    pos = np.linspace(0, track.length, numIntervals + 1 - (len(df1) - 1))

    df2 = pd.DataFrame({'position [m]':pos}).set_index('position [m]')
    df3 = df2.join(df1, how='outer').ffill()

    if len(df3) != numIntervals + 1:

        raise ValueError("Wrong number of computed discretization intervals!")

    return df3


class Track():

    def __init__(self, config, tUpper=None, pathJSON='tracks'):
        """
        Constructor of Track objects.
        """

        self.tUpper = tUpper  # maximum travel time [s] (minimum time + reserve)

        # check config
        if not isinstance(config, dict):

            raise ValueError("Track configuration should be provided as a dictionary!")

        if 'id' not in config:

            raise ValueError("Track ID must be specified in configuration!")

        # open json file
        filename = os.path.join(pathJSON, config['id']+'.json')

        with open(filename) as file:

            data = json.load(file)

        # check compatibility of TTOBench version
        versionTTOBench = '1.1'

        if 'metadata' not in data or 'library version' not in data['metadata']:

            raise ValueError("Library version not found in json file!")

        else:

            pattern = r'v([\d.]+)'
            match = re.search(pattern, data['metadata']['library version'])

            if match:

                version = match.group(1)

                if version != versionTTOBench:

                    raise ValueError("Import function works only for library version {}!".format(versionTTOBench))

            else:

                raise ValueError("Unexpected format of 'library version' in json file!")

        # read data
        self.length = convertUnit(data['stops']['values'][-1], data['stops']['unit'])
        self.altitude = convertUnit(data['altitude']['value'], data['altitude']['unit']) if 'altitude' in data else 0
        self.title = data['metadata']['id']

        self.importSpeedLimitTuples(data['speed limits']['values'], data['speed limits']['units']['velocity'])
        self.importGradientTuples(data['gradients']['values'], data['gradients']['units']['slope'])

        numStops = len(data['stops']['values'])
        indxDeparture = config['from'] if 'from' in config else 0
        indxDestination = config['to'] if 'to' in config else numStops-1

        if not 0 <= indxDeparture < numStops - 1:

            raise ValueError("Index of departure is out of bounds!")

        if not indxDeparture < indxDestination < numStops:

            raise ValueError("Index of destination is out of bounds!")

        posDeparture = convertUnit(data['stops']['values'][indxDeparture], data['stops']['unit'])
        posDestination = convertUnit(data['stops']['values'][indxDestination], data['stops']['unit'])

        self.updateLimits(posDeparture, posDestination)

        self.checkFields()


    def lengthOk(self):

        return True if self.length is not None and self.length > 0 and not np.isinf(self.length) else False


    def gradientsOk(self):

        return True if self.gradients.shape[0] > 0 and checkDataFrame(self.gradients, self.length) else False


    def speedLimitsOk(self):

        return True if self.speedLimits.shape[0] > 0 and checkDataFrame(self.speedLimits, self.length) else False


    def checkFields(self):

        if not self.lengthOk():

            raise ValueError("Track length must be a strictly positive number, not {}!".format(self.length))

        if self.altitude is None or np.isinf(self.altitude):

            raise ValueError("Altitude must be a number, not {}!".format(self.altitude))

        if self.tUpper is None or self.tUpper <= 0 or np.isinf(self.tUpper):

            raise ValueError("Maximum trip time must be a strictly positive number, not {}!".format(self.tUpper))

        if not self.gradientsOk():

            raise ValueError("Issue with track gradients!")

        if not self.speedLimitsOk():

            raise ValueError("Issue with track speed limits!")


    def importGradientTuples(self, tuples, unit='permil'):

        if not self.lengthOk():

            raise ValueError("Cannot import gradients without a valid track length!")

        if unit not in {'permil'}:

            raise ValueError("Specified gradient unit not supported!")

        self.gradients = importTuples(tuples, 'Position [m]', 'Gradient [permil]')

        checkDataFrame(self.gradients, self.length)


    def importSpeedLimitTuples(self, tuples, unit='km/h'):

        if not self.lengthOk():

            raise ValueError("Cannot import speed limits without a valid track length!")

        if unit not in {'km/h', 'm/s'}:

            raise ValueError("Specified speed unit not supported!")

        tuples = [(p, convertUnit(v, unit)) for p,v in tuples]
        self.speedLimits = importTuples(tuples, 'Position [m]', 'Speed limit [m/s]')

        checkDataFrame(self.speedLimits, self.length)


    def reverse(self):
        # switch to opposite trip

        try:

            self.checkFields()

        except ValueError as e:

            raise ValueError("Track cannot be reversed due to error: {}".format(str(e)))

        def flipData(df):

            newIndex = np.flip(self.length - np.append(df.index[1:], self.length))
            newValues = np.flip(df[df.keys()[0]].values)
            return pd.DataFrame({df.index.name:newIndex, df.keys()[0]:newValues}).set_index(df.index.name)

        self.gradients = -flipData(self.gradients)
        self.speedLimits = flipData(self.speedLimits)

        self.title = self.title + ' (reversed)'

        return self


    def mergeDataFrames(self):
        """
        Build dataframe with intervals of constant gradient and speed limit.
        """

        return self.gradients.join(self.speedLimits, how='outer').fillna(method='ffill')


    def print(self):
        """
        Basic printing functionality.
        """

        df = self.mergeDataFrames()
        print(df)


    def plot(self, figSize=[12, 6]):
        """
        Basic plotting functionality.
        """

        speedLimits = self.speedLimits
        speedLimits = pd.concat([speedLimits, pd.DataFrame({speedLimits.index.name:[self.length], speedLimits.keys()[0]:[None]}).set_index(speedLimits.index.name)])
        speedLimits.index = speedLimits.index.map(lambda x: x/1e3)  # convert m to km
        speedLimits['Speed limit [m/s]'] *= 3.6  # convert m/s to km/h
        speedLimits.rename(columns={'Speed limit [m/s]':'Speed limit'}, inplace=True)

        axV = speedLimits.plot(color='purple', drawstyle='steps-post', xlabel = 'Position [km]', ylabel='Velocity [km/h]', figsize=figSize)
        axV.legend(loc='lower left')

        axA = axV.twinx()

        altitude = computeAltitude(self.gradients, self.length)
        altitude.index = altitude.index.map(lambda x: x/1e3)  # convert m to km
        altitude.rename(columns={'Altitude [m]':'Track profile'}, inplace=True)

        axA = altitude.plot(ax=axA, color='gray', title='Visualization of ' + self.title + ' track', grid=True, ylabel='Altitude [m]')
        axA.legend(loc='upper right')

        plt.show()


    def updateLimits(self, positionStart=None, positionEnd=None, unit='m'):
        """
        Truncate track to given positions.
        """

        positionStart = 0 if positionStart is None else positionStart
        positionEnd = self.length if positionEnd is None else positionEnd

        if (not 0 <= positionStart < self.length) or (not 0 < positionEnd <= self.length) :

            raise ValueError("Given positions must be between limits of track!")

        positionStart = convertUnit(positionStart, unit)
        positionEnd = convertUnit(positionEnd, unit)

        newPos = pd.DataFrame({'Position [m]':[positionStart]}).set_index('Position [m]')

        def crop(dfIn):

            dfOut = newPos.join(dfIn, how='outer').ffill()
            dfOut = dfOut.loc[(dfOut.index >= positionStart)&(dfOut.index <= positionEnd)]
            dfOut['Position [m]'] = dfOut.index - dfOut.index[0]
            dfOut.set_index('Position [m]', inplace=True)

            return dfOut

        self.length -= positionStart + (self.length - positionEnd)

        self.speedLimits = crop(self.speedLimits)
        self.gradients = crop(self.gradients)


if __name__ == '__main__':

    # Example on how to load and plot a track

    track = Track(config={'id':'00_stationX_stationY'}, tUpper=1300)
    track.plot()
