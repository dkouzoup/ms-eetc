import numpy as np
import pandas as pd

from data import dataSwissTrack, dataRefSpeed100Track

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


def checkDataFrame(df, trackLength):

    if df.index[0] != 0:

        raise ValueError("Error in '{}': First track section must start at 0 m (beginning of track)!".format(df.columns[0]))

    if df.index[-1] > trackLength:

        raise ValueError("Error in '{}': Last track section must start before {} m (end of track)!".format(df.columns[0], trackLength))

    return True


def computeAltitude(gradients, length):

    position = np.append(gradients.index.values, length)
    altitude = np.array([0])

    for ii in range(1, len(gradients)+1):

        gradient = gradients.iloc[ii-1][0] if isinstance(gradients, pd.DataFrame) else gradients.iloc[ii-1]
        posStart = gradients.index[ii-1]
        posEnd = gradients.index[ii] if ii < len(gradients) else length
        height = (posEnd - posStart)*(gradient/1e3)
        altitude = np.append(altitude, altitude[-1] + height)

    altitude += abs(min(altitude))  # lowest point is 0 m
    df = pd.DataFrame({gradients.index.name:position, 'Altitude [m]':altitude}).set_index(gradients.index.name)

    return df


def computeDiscretizationPoints(track, numIntervals):
    "Compute the space discretization points based on track characteristics and horizon length."

    df1 = track.mergeDataFrames()

    pos = np.linspace(0, track.length, numIntervals + 1 - (len(df1) - 1))

    df2 = pd.DataFrame({'position [m]':pos}).set_index('position [m]')
    df3 = df2.join(df1, how='outer').ffill()

    if len(df3) != numIntervals + 1:

        raise ValueError("Wrong number of computed discretization intervals!")

    return df3


class Track():

    def __init__(self, track='RefSpeed100', tUpper=None):

        if track == 'RefSpeed100':

            data = dataRefSpeed100Track()

        elif track == 'Swiss':

            data = dataSwissTrack()

        else:

            raise ValueError("Unknown track specified!")

        self.length = data['length']  # track length [m]

        self.importGradientTuples(data['gradients'])  # gradients [promil]

        self.importSpeedLimitTuples(data['speedLimits'])  # speed limits [m/s]

        self.tUpper = tUpper  # maximum travel time [s] (minimum time + reserve)

        self.title = track

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

        if self.tUpper is not None and (self.tUpper <= 0 or np.isinf(self.tUpper)):

            raise ValueError("Maximum trip time must be a strictly positive number, not {}!".format(self.tUpper))

        if not self.gradientsOk():

            raise ValueError("Issue with track gradients!")

        if not self.speedLimitsOk():

            raise ValueError("Issue with track speed limits!")


    def importGradientTuples(self, tuples):

        if not self.lengthOk():

            raise ValueError("Cannot import gradients without a valid track length!")

        self.gradients = importTuples(tuples, 'Position [m]', 'Gradient [promil]')

        checkDataFrame(self.gradients, self.length)


    def importSpeedLimitTuples(self, tuples):

        if not self.lengthOk():

            raise ValueError("Cannot import speed limits without a valid track length!")

        self.speedLimits = importTuples(tuples, 'Position [m]', 'Speed limit [m/s]')

        checkDataFrame(self.speedLimits, self.length)


    def importGradientCsv(self, filename):

        if not self.lengthOk():

            raise ValueError("Cannot import gradients without a valid track length!")

        grad = pd.read_csv(filename)

        if len(grad.columns) != 2:

            raise ValueError("CSV file should have two columns (position in m and gradient in specified unit)!")

        grad.rename(columns={grad.columns[0]:'Position [m]', grad.columns[1]: 'Gradient [promil]'}, inplace=True)

        self.gradients = grad.set_index(grad.columns[0])

        checkDataFrame(self.gradients, self.length)


    def importSpeedLimitCsv(self, filename):

        if not self.lengthOk():

            raise ValueError("Cannot import speed limits without a valid track length!")

        speedLimits = pd.read_csv(filename)

        if len(speedLimits.columns) != 2:

            raise ValueError("CSV file should have two columns (position in m and velocity in specified unit)!")

        speedLimits.rename(columns={speedLimits.columns[0]:'Position [m]', speedLimits.columns[1]:'Speed limit [m/s]'}, inplace=True)

        self.speedLimits = speedLimits.set_index(speedLimits.columns[0])

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
        "Dataframe with intervals of constant gradient and speed limit."

        return self.gradients.join(self.speedLimits, how='outer').fillna(method='ffill')


    def print(self):

        df = self.mergeDataFrames()
        print(df)


    def plot(self, figSize=[12, 6]):

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


    def truncate(self, length=None):
        "Reduce length of track."

        if length == None:

            return

        if (not 0 < length <= self.length) :

            raise ValueError("New length must be smaller than current length!")

        def crop(dfIn):

            dfOut = dfIn.copy()
            dfOut = dfOut.loc[dfOut.index < length]

            return dfOut

        self.length = length

        self.speedLimits = crop(self.speedLimits)
        self.gradients = crop(self.gradients)


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Example on how to load and plot a track

    track = Track(track='Swiss')
    track.plot()

    plt.show()
