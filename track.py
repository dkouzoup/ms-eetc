import json
import math
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import checkTTOBenchVersion

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

    if any(np.isinf(index)):

        raise ValueError("Position data cannot be infinite!")

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

    CURVATURE_THRESHOLD = 1/150 # absolute value of maximum allowed cruvature [1/m]

    def __init__(self, config, pathJSON='tracks'):
        """
        Constructor of Track objects.
        """

        # check config
        if not isinstance(config, dict):

            raise ValueError("Track configuration should be provided as a dictionary!")

        if 'id' not in config:

            raise ValueError("Track ID must be specified in configuration!")

        # open json file
        filename = os.path.join(pathJSON, config['id']+'.json')

        with open(filename) as file:

            data = json.load(file)

        checkTTOBenchVersion(data, ['1.1', '1.2', '1.3'])

        # read data
        self.length = convertUnit(data['stops']['values'][-1], data['stops']['unit'])
        self.altitude = convertUnit(data['altitude']['value'], data['altitude']['unit']) if 'altitude' in data else 0
        self.title = data['metadata']['id']

        self.importSpeedLimitTuples(data['speed limits']['values'], data['speed limits']['units']['velocity'])

        self.importGradientTuples(data['gradients']['values'] if 'gradients' in data else [(0.0, 0.0)],
                                  data['gradients']['units']['slope'] if 'gradients' in data else 'permil')

        self.importCurvatureTuples(data['curvatures']['values'] if 'curvatures' in data else [(0.0, "infinity", "infinity")],
                                   data['curvatures']['units']['radius at start'] if 'curvatures' in data else "m",
                                   data['curvatures']['units']['radius at end'] if 'curvatures' in data else "m",
                                   config['clothoidSamplingInterval'] if 'clothoidSamplingInterval' in config else None)

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


    def curvaturesOk(self):

        if (abs(self.curvatures['Curvature [1/m]']) > Track.CURVATURE_THRESHOLD).values.any():

            return False

        return True if self.curvatures.shape[0] > 0 and checkDataFrame(self.curvatures, self.length) else False


    def checkFields(self):

        if not self.lengthOk():

            raise ValueError("Track length must be a strictly positive number, not {}!".format(self.length))

        if self.altitude is None or np.isinf(self.altitude):

            raise ValueError("Altitude must be a number, not {}!".format(self.altitude))

        if not self.gradientsOk():

            raise ValueError("Issue with track gradients!")

        if not self.speedLimitsOk():

            raise ValueError("Issue with track speed limits!")

        if not self.curvaturesOk():

            raise ValueError("Issue with track curvatures!")


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


    def importCurvatureTuples(self, tuples, unitRadiusStart='m', unitRadiusEnd='m', clothoidSamplingInterval=None):

        if not self.lengthOk():

            raise ValueError("Cannot import curvature without a valid track length!")

        if unitRadiusStart not in {'m', 'km'} or unitRadiusEnd not in {'m', 'km'}:

            raise ValueError("Specified curvature radius unit not supported!")

        # if radius is 'infinity', the casting to float produces the float inf
        tuples = [(p, convertUnit(float(radiusStart), unitRadiusStart), convertUnit(float(radiusEnd), unitRadiusEnd)) for p, radiusStart, radiusEnd in tuples]

        tuples = self.sampleClothoid(tuples, clothoidSamplingInterval)

        self.curvatures = importTuples(tuples, 'Position [m]', ['Curvature [1/m]'])

        checkDataFrame(self.curvatures, self.length)


    def sampleClothoid(self, tuples, ds=None):
        """
        Approximates clothoid transition curve with piecewise-constant function.

        Given an interval [s_i, s_i + ds] the approximation of K(s) on the interval
        is K_avg(s) = (K(s_i) + K(s_i + ds))/2. For the last interval [s_i, s_f] the
        approximation of K(s) is K_avg(s) = (K(s_i) + K(s_f))/2 where K(s_f) is the
        curvature at the end of the section. When ds is not specified, K(s) is
        approximated as K_approx(s) = (K(s_0) + K(s_f))/2.

        - param tuples: a list of triples of form (p, Rstart, Rend) where p is the
        coordinate [m] at the start of the track section; Rstart is the radius [m] at the
        start of the section and Rend the radius [m] at the end.
        - param ds: the step size [m] used to approximate the clothoid. Note that we cannot
        guarantee that all intervals have size ds. Hence, in
        general, the last interval has lenght L such that: ds <= L < 2*ds while all other
        intervals have size ds.
        - return: a list of pairs (p, K) where K [1/m] is the approximation
        of the clothoid curvature in the track section starting at position p.
        """

        if any([radiusValue == 0 for radiusValue in [trackSection[radiusType] for trackSection in tuples for radiusType in range(1,3)]]):

            raise ValueError("Curvature radius cannot be 0!")

        if any([tuples[sectionIndex][0] < 0 for sectionIndex in range(len(tuples))]):

            raise ValueError("Positions cannot be negative!")

        if any([tuples[sectionIndex][0] == tuples[sectionIndex+1][0] for sectionIndex in range(len(tuples)-1)]):

            raise ValueError("Positions must be monotonically increasing")

        if (ds != None and ds <= 0 ):

            raise ValueError("Discretization step must be greater than zero or None!")

        result = []

        epsilon = sys.float_info.epsilon

        for trackIndex, trackSection in enumerate(tuples):

            sectionStart = trackSection[0]
            curvatureStart = 1/trackSection[1]
            curvatureEnd = 1/trackSection[2]

            if abs(curvatureStart - curvatureEnd) <= epsilon:

                result.append((sectionStart, curvatureStart))

            else:

                sectionEnd = tuples[trackIndex+1][0] if trackIndex < len(tuples)-1 else self.length

                if ds == None or int((sectionEnd-sectionStart)/ds) == 0:

                    result.append((sectionStart, (curvatureStart + curvatureEnd)/2))

                else:

                    nIntervals = int((sectionEnd-sectionStart)/ds)

                    # the curvature of a clothoid is K(s) = K_0 + (s-s_0)/alpha
                    alpha = (sectionEnd-sectionStart)/(curvatureEnd-curvatureStart)

                    for intervalIndex in range(nIntervals):

                        discretizationPoint = sectionStart+intervalIndex*ds

                        curvatureAtDiscretizationPoint = curvatureStart + intervalIndex*ds/alpha

                        # remark that the last interval has lenght L such that: ds <= L < 2*ds.
                        avgCurvature = (curvatureAtDiscretizationPoint + curvatureEnd)/2 if intervalIndex==nIntervals-1 \
                              else curvatureAtDiscretizationPoint + ds/(2*alpha)

                        result.append((discretizationPoint, avgCurvature))

        return result


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
        self.curvatures = -flipData(self.curvatures)

        self.title = self.title + ' (reversed)'

        return self


    def mergeDataFrames(self):
        """
        Build dataframe with intervals of constant gradient, speed limit and curvature.
        """

        joinedGradientsAndSpeedLimits = self.gradients.join(self.speedLimits, how='outer').fillna(method='ffill')
        return self.curvatures.join(joinedGradientsAndSpeedLimits, how='outer').fillna(method='ffill')

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
        self.curvatures = crop(self.curvatures)


if __name__ == '__main__':

    # Example on how to load and plot a track

    track = Track(config={'id':'00_stationX_stationY'})
    track.plot()
