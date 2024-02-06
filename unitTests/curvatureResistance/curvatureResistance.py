import copy
import numpy as np
import sys
import unittest

sys.path.append('../..')

from efficiency import totalLossesFunction
from ocp import casadiSolver
from train import Train
from track import Track


class TestCurvatureResistance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        '''
        Creates two test tracks that are used in the tests.
        '''

        cls.track = ("../../tracks/", "00_var_speed_limit_100")

        cls.constantK = 1/300 # [m^-1]

        cls.finalPosition = 3475 # [m]

        cls.speedAtStart = 1 # [m/s]

        cls.speedAtEnd = 1 # [m/s]

        cls.thresholdVelocityError= 1e-3 # dimensionless quantity used in the minimum time test

        cls.thresholdEnergyError = 5e-2 # dimensionless quantity used in the minimum energy test

        try:

            cls.trackNoCurvature = Track(config={'id': cls.track[1]}, pathJSON=cls.track[0])

        except Exception as e:

            raise unittest.SkipTest("Impossible to create track object. Error was: ", e)

        cls.trackConstantCurvature = copy.deepcopy(cls.trackNoCurvature)

        cls.trackConstantCurvature.importCurvatureTuples(tuples=[[0.0, str(1/cls.constantK), str(1/cls.constantK)]])


    def specificCurvatureResistanceForce(self, g, rho):
        '''
        Computes specific curvature resistance force [m/s^2]

        - param g: constant of gravity [m/s^2]
        - param rho: dimensionless rotating mass factor
        '''

        return g*0.5*abs(self.constantK)/((1-30*abs(self.constantK))*rho)*(abs(self.constantK)<=1/300) + \
                g*0.65*abs(self.constantK)/((1-55*abs(self.constantK))*rho)*(abs(self.constantK)>1/300)


    def solveOptimalControlProblem(self, track, energyOptimal, lossFunction, terminalTime, train, finalPosition):
        '''
        Solves a minimum time or minimium energy optimal control problem.

        - param track: Track object
        - param energyOptimal: bool variable. If false the minimum time problem is solved
        - param lossFunction: a function taking two inputs (f,v) and returning the electrical
        losses of the train. Used for solving the minimum energy problem.
        - param terminalTime: time at which the trip must complete. For the minimum time problem this is just an upperbound of the problem.
        - param train: Train object
        - param finalPosition: position where the trip must end.
        - return: a dataframe containing the solution of the problem.
        '''

        v0 = self.speedAtStart
        vN = self.speedAtEnd

        track.updateLimits(positionEnd=finalPosition)

        solverOpts = { "maxIterations": 500,
                    "numIntervals": 300,
                    "integrationMethod": "RK",
                    "integrationOptions": {"order": 4, "numSteps": 1, "numApproxSteps": 1 },
                    "energyOptimal": energyOptimal,
                    "minimumVelocity": min(v0, vN)}

        train.powerLosses = lossFunction

        ocp0 = casadiSolver(train, track, solverOpts)

        df0, _ = ocp0.solve(terminalTime, terminalVelocity=vN, initialVelocity=v0)

        return df0


    def testMinimumTimeProblem(self):
        '''
        Two trains without power constraints are considered and each of them travels on one of the test tracks.
        There are only two differences between the two trains:
        1. The maximum traction force of the train travelling on the track with
        curvature is equal to the maximum traction force of the train on the straight
        track plus the value of curvature resistance.
        2. The maximum regenerative braking force of the train travelling on the track with
        curvature is equal to the maximum regenerative braking force of the train on the straight
        reduced by an amount equal to the value of curvature resistance.

        Under these conditions, the optimal speed profiles of both problems must coincide.
        '''

        minTimeUpperBound = 180 # [s]

        # # # Perfect efficiency
        lossFun = lambda f,v: 0

        train = Train(config={'id':'NL_Intercity_VIRM6'}, pathJSON='../../trains')
        train.forceMinPn = 0
        train.powerMax = None
        train.powerMin = None

        resultNoCurvature = self.solveOptimalControlProblem(track = self.trackNoCurvature,
                                                            energyOptimal = False,
                                                            lossFunction = lossFun,
                                                            terminalTime = minTimeUpperBound,
                                                            train = train,
                                                            finalPosition = self.finalPosition)

        train.forceMax = train.forceMax + self.specificCurvatureResistanceForce(train.g, train.rho)*train.mass*train.rho

        # remark that forceMin is negative; hence we sum the curvature resistance term.
        train.forceMin = train.forceMin + self.specificCurvatureResistanceForce(train.g, train.rho)*train.mass*train.rho

        resultConstantCurvature = self.solveOptimalControlProblem(track = self.trackConstantCurvature,
                                                                  energyOptimal = False,
                                                                  lossFunction = lossFun,
                                                                  terminalTime = minTimeUpperBound,
                                                                  train = train,
                                                                  finalPosition = self.finalPosition)

        resultNoCurvature, resultConstantCurvature = resultNoCurvature.reset_index(), resultConstantCurvature.reset_index()

        self.assertTrue(all(np.abs((resultNoCurvature['Velocity [m/s]'] - resultConstantCurvature['Velocity [m/s]'])/resultNoCurvature['Velocity [m/s]'])
                            <= self.thresholdVelocityError))


    def testMinimumEnergyProblem(self):
        '''
        Solves the minimum energy problem for both tracks considering
        different power losses models. The difference in
        mechanical energy between the two optimal trajectories
        must be equal to the energy originating from curvature resistance.
        '''

        tripTime = 200 # [s]

        finalPosition = 3475 # [m]

        train = Train(config={'id':'NL_Intercity_VIRM6'}, pathJSON='../../trains')
        train.forceMinPn = 0

        etaMax = 0.73
        auxiliaries = 27000
        etaGear = 0.96
        noLosses = lambda f,v: 0
        idealLosses = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
        realLosses = totalLossesFunction(train, auxiliaries=auxiliaries, etaGear=etaGear)

        lossesFunctions = [noLosses, idealLosses, realLosses]

        conversionWattToKWattHour = lambda w: w/(3600*1000)
        energyCurvatureResistance = conversionWattToKWattHour(self.specificCurvatureResistanceForce(train.g, train.rho)*train.rho*train.mass*finalPosition)

        # contains the metric for measuring the correctness of the result
        testsResults = []

        for lossFunction in lossesFunctions:

            resultNoCurvature = self.solveOptimalControlProblem(track = self.trackNoCurvature,
                                                                energyOptimal = True,
                                                                lossFunction = lossFunction,
                                                                terminalTime = tripTime,
                                                                train = train,
                                                                finalPosition = finalPosition)

            resultConstantCurvature = self.solveOptimalControlProblem(track = self.trackConstantCurvature,
                                                                      energyOptimal = True,
                                                                      lossFunction = lossFunction,
                                                                      terminalTime = tripTime,
                                                                      train = train,
                                                                      finalPosition = finalPosition)

            resultNoCurvature, resultConstantCurvature = resultNoCurvature.reset_index(),  resultConstantCurvature.reset_index()

            totalEnergyNoCurvature, totalEnergyConstantCurvature = round(resultNoCurvature['Energy [kWh]'].sum(),1), round(resultConstantCurvature['Energy [kWh]'].sum(),1)

            energyLossesNoCurvature, energyLossesConstantCurvature = round(resultNoCurvature['Losses [kWh]'].sum(),1), round(resultConstantCurvature['Losses [kWh]'].sum(),1)

            mechanicalEnergyTrackNoCurvature, mechanicalEnergyTrackConstantCurvature = totalEnergyNoCurvature - energyLossesNoCurvature, totalEnergyConstantCurvature - energyLossesConstantCurvature

            mechanicalEnergyDelta = mechanicalEnergyTrackConstantCurvature - mechanicalEnergyTrackNoCurvature

            testsResults.append(abs(energyCurvatureResistance - mechanicalEnergyDelta) / energyCurvatureResistance)

        self.assertTrue(all([result <= self.thresholdEnergyError for result in testsResults]))


    def testClothoidApproximation(self):

        testTrack = copy.deepcopy(self.trackNoCurvature)

        r0 = 1000
        rf = 500

        k0 = 1/r0
        kf = 1/rf

        # # # no sampling step specified. The result is the averge of the two curvatures.

        testTrack.importCurvatureTuples(tuples=[[0.0, r0, rf]])

        expectedResult = {0.0: (k0 + kf)/2}

        self.assertTrue(testTrack.curvatures['Curvature [1/m]'].to_dict() == expectedResult)

        # # # sampling with step size greater than section length

        testTrack.importCurvatureTuples(tuples=[[0.0, r0, rf]], clothoidSamplingInterval=testTrack.length+1)

        self.assertTrue(testTrack.curvatures['Curvature [1/m]'].to_dict() == expectedResult)

        # # # sampling with step size equal to 1/4 of the total length

        ds = testTrack.length/4

        testTrack.importCurvatureTuples(tuples=[[0.0, r0, rf]], clothoidSamplingInterval=ds)

        alpha = testTrack.length / (kf - k0)

        k1 = (k0 + (k0 + ds*1/alpha))/2

        k2 = ((k0 + ds*1/alpha) + (k0 + ds*2/alpha))/2

        k3 = ((k0 + ds*2/alpha) + (k0 + ds*3/alpha))/2

        k4 = ((k0 + ds*3/alpha) + kf)/2

        expectedResult = {0.0: k1, ds: k2, 2*ds: k3, 3*ds: k4}

        self.assertTrue(testTrack.curvatures['Curvature [1/m]'].to_dict() == expectedResult)

        # # # track length not divisible by ds.

        ds = testTrack.length/4 + 1

        testTrack.importCurvatureTuples(tuples=[[0.0, r0, rf]], clothoidSamplingInterval=ds)

        k1 = (k0 + (k0 + ds*1/alpha))/2

        k2 = ((k0 + ds*1/alpha) + (k0 + ds*2/alpha))/2

        k3 = ((k0 + ds*2/alpha) + kf)/2

        expectedResult = {0.0: k1, ds: k2, 2*ds: k3}

        self.assertTrue(testTrack.curvatures['Curvature [1/m]'].to_dict() == expectedResult)

        # # # radius equal to "infinity"

        testTrack.importCurvatureTuples(tuples=[[0.0, r0, "infinity"]])

        expectedResult = {0.0: k0/2}

        self.assertTrue(testTrack.curvatures['Curvature [1/m]'].to_dict() == expectedResult)

        # # # negative ds

        self.assertRaises(ValueError, testTrack.importCurvatureTuples, tuples=[[0.0, r0, rf]], clothoidSamplingInterval=-1)

        # # # radius equal to zero

        self.assertRaises(ValueError, testTrack.importCurvatureTuples, tuples=[[0.0, 0.0, rf]])

        # # # section of length zero

        self.assertRaises(ValueError, testTrack.importCurvatureTuples, tuples=[[500, r0, rf], [500, rf, 1+rf]])

        # # # section of length zero

        self.assertRaises(ValueError, testTrack.importCurvatureTuples, tuples=[[-1, r0, rf]])


if __name__ == '__main__':

    unittest.main()