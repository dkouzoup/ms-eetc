import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tempfile
import unittest

sys.path.append('../..')

from efficiency import totalLossesFunction
from ocp import casadiSolver
from train import Train
from track import Track
from utils import postProcessDataFrame


class TestCurvatureResistance(unittest.TestCase):

    TRACK = "../../tracks/00_stationX_stationY.json"

    CONSTANT_K = 1/300 # [m^-1]

    DEFAULT_TRIP_FINAL_POSITION = 3475 # [m]

    DEFAULT_SPEED_AT_START = 1 # [m/s]

    DEFAULT_SPEED_AT_END = 1 # [m/s]
    
    THRESHOLD_VELOCITY_ERROR_MINIMUM_TIME_PROBLEM = 1e-3 # adimensional quantity

    THRESHOLD_ENERGY_ERROR_MINIMUM_ENERGY_PROBLEM = 5e-2 # adimensional quantity

    @classmethod
    def setUpClass(cls):
        '''
        Based on the dataset contained in TRACK, two temporary track datasets are created.
        In each test we solve optimal control problems for both tracks and we compare their results.
        '''

        cls.fileNoCurvature = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w+')

        cls.fileConstantCurvature = tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w+')

        try:

            with open(cls.TRACK) as file:

                originalTrack = json.load(file)

            trackNoCurvature = originalTrack

            del trackNoCurvature['curvatures']

            json.dump(trackNoCurvature, cls.fileNoCurvature, indent=4)

            cls.fileNoCurvature.flush()

            trackConstantCurvature = trackNoCurvature

            trackConstantCurvature["curvatures"] = {"units": {"position": "m", "radius at start": "m", "radius at end": "m" }, 
                                                    "values": [[0.0, str(1/cls.CONSTANT_K), str(1/cls.CONSTANT_K)]]}
            
            json.dump(trackConstantCurvature, cls.fileConstantCurvature, indent=4)

            cls.fileConstantCurvature.flush()

        except Exception:

            cls.tearDownClass()

            raise unittest.SkipTest("Something went wrong when attempting to create test tracks. ABORTING ALL TESTS!")
        
        
    @classmethod
    def tearDownClass(cls):

        cls.fileNoCurvature.close(), cls.fileConstantCurvature.close()

        os.unlink(cls.fileNoCurvature.name), os.unlink(cls.fileConstantCurvature.name)

    

    def specificCurvatureResistanceForce(cls, g, rho):
        '''
        Computes specific curvature resistance force [m/s^2]

        - param g: constant of gravity [m/s^2]
        - param rho: dimensionless rotating mass factor 
        '''

        return g*0.5*abs(cls.CONSTANT_K)/((1-30*abs(cls.CONSTANT_K))*rho)*(abs(cls.CONSTANT_K)<=1/300) + \
                g*0.65*abs(cls.CONSTANT_K)/((1-55*abs(cls.CONSTANT_K))*rho)*(abs(cls.CONSTANT_K)>1/300)



    def solveOptimalControlProblem(cls, pathToTrackData, trackDataFilename, energyOptimal, lossFunction, 
                                   terminalTime, train=None, finalPosition=None, realLossFunction=None):
        '''
        Solves a minimum time or minimium energy optimal control problem.

        - param pathToTrackData: absolute path to the json file containing the track dataset
        - param trackDataFilename: name of json file without extension
        - param energyOptimal: bool variable. If false the minimum time problem is solved
        - param lossFunction: a function taking two inputs (f,v) and returning the electrical
        losses of the train. Used for solving the minimum energy problem.
        - param terminalTime: time at which the trip must complete. For the minimum time problem this is just an upperbound of the problem.
        - param finalPosition: position where the trip must end.
        - param realLossFunction: if provided this loss function is used to compute the real losses of the train. Note that it is not used to
        solve the minimum energy problem.
        - return: a dataframe containing the solution of the problem.
        '''

        v0 = cls.DEFAULT_SPEED_AT_START
        vN = cls.DEFAULT_SPEED_AT_END
        
        if (train == None):

            train = Train(train='Intercity')

            train.forceMinPn = 0  

        if (finalPosition == None):

            finalPosition = cls.DEFAULT_TRIP_FINAL_POSITION

        track = Track(config={'id': os.path.splitext(trackDataFilename)[0]}, pathJSON= pathToTrackData)

        track.updateLimits(positionEnd=finalPosition)

        solverOpts = { "maxIterations": 500, 
                    "numIntervals": 300, 
                    "integrationMethod": "RK", 
                    "integrationOptions": {"order": 4, "numSteps": 1, "numApproxSteps": 1 },
                    "energyOptimal": energyOptimal,
                    "minimumVelocity": min(v0, vN)}
        
        # # # solve problem with perfect efficiency
        train.powerLosses = lossFunction

        ocp0 = casadiSolver(train, track, solverOpts)

        df0, _ = ocp0.solve(terminalTime, terminalVelocity=vN, initialVelocity=v0)

        if (realLossFunction != None):

            train.powerLosses = realLossFunction

            df0 = postProcessDataFrame(df0, ocp0.points, train, integrateLosses=True)

        return df0


    def testMinimumTimeProblem(cls):
        '''
        Two trains without power constraints are considered and each of them travels on one of the test tracks.
        There are only two differences between the two trains:
        1. The maximum traction force of the train travelling on the track with 
        curvature is equal to the maximum traction force of the train on the straight
        track plus the value of curvature resistance.
        2. The maximum regenerative braking force of the train travelling on the track with 
        curvature is equal to the maximum regenerative braking force of the train on the straight
        track minus the value of curvature resistance.
        
        Under these conditions, the optimal speed profiles of both problems must coincide.
        '''
            
        minTimeInitialGuess = 180 # [s]

        # # # Perfect efficiency
        lossFun = lambda f,v: 0

        train = Train(train='Intercity')
        train.forceMinPn = 0

        train.powerMax = None

        train.powerMin = None
        
        pathNoCurvature, filenameNoCurvature = os.path.split(os.path.abspath(cls.fileNoCurvature.name))

        resultNoCurvature = cls.solveOptimalControlProblem(pathToTrackData=pathNoCurvature, 
                                                           trackDataFilename=os.path.splitext(filenameNoCurvature)[0], 
                                                           energyOptimal = False, 
                                                           lossFunction = lossFun,
                                                           terminalTime = minTimeInitialGuess,
                                                           train = train,)
        
        train.forceMax = train.forceMax + cls.specificCurvatureResistanceForce(train.g, train.rho)*train.mass*train.rho
        
        # remark that forceMin is negative; hence we sum the curvature resistance term.
        train.forceMin = train.forceMin + cls.specificCurvatureResistanceForce(train.g, train.rho)*train.mass*train.rho
        
        pathConstantCurvature, filenameConstantCurvature = os.path.split(os.path.abspath(cls.fileConstantCurvature.name))
        
        resultConstantCurvature = cls.solveOptimalControlProblem(pathToTrackData=pathConstantCurvature, 
                                                                 trackDataFilename=os.path.splitext(filenameConstantCurvature)[0], 
                                                                 energyOptimal = False, 
                                                                 lossFunction = lossFun, 
                                                                 terminalTime = minTimeInitialGuess,
                                                                 train = train)

        resultNoCurvature, resultConstantCurvature = resultNoCurvature.reset_index(), resultConstantCurvature.reset_index()
        
        # at each shooting node, the difference in speed between the two tracks must be less then the speed on one of the 
        # tracks times the threshold constant. 
        # If the reference speed on one track is 100m/s, a threshold of 1e-3 means that the resulting difference in position is 0.1m every 100m.
        cls.assertTrue(all(np.abs((resultNoCurvature['Velocity [m/s]'] - resultConstantCurvature['Velocity [m/s]'])/resultNoCurvature['Velocity [m/s]']) <= cls.THRESHOLD_VELOCITY_ERROR_MINIMUM_TIME_PROBLEM))


    
    def testMinimumEnergyProblem(cls):
        ''' 
        We solve the minimum energy problem for both tracks considering the same
        train traveling on the two tracks and different power losses models.
        
        We verify that the minimal energy solution is such that the difference in 
        mechanical energy between the two tracks is equal to the energy 
        originating due to curvature resistance.
        '''

        tripTime = 200 # [s]

        finalPosition = 3475 # [m]

        train = Train(train='Intercity')
        train.forceMinPn = 0

        etaMax = 0.73
        auxiliaries = 27000
        etaGear = 0.96
        noLosses = lambda f,v: 0
        idealLosses = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
        realLosses = totalLossesFunction(train, auxiliaries=auxiliaries, etaGear=etaGear)

        lossesFunctions = [noLosses, idealLosses, realLosses]
        
        pathNoCurvature, filenameNoCurvature = os.path.split(os.path.abspath(cls.fileNoCurvature.name))

        pathConstantCurvature, filenameConstantCurvature = os.path.split(os.path.abspath(cls.fileConstantCurvature.name))

        conversionWattToKWattHour = lambda w: w/(3600*1000)
        energyCurvatureResistance = conversionWattToKWattHour(cls.specificCurvatureResistanceForce(train.g, train.rho)*train.rho*train.mass*finalPosition)

        # contains the metric for measuring the correctness of the result
        testsResults = []

        for lossFunction in lossesFunctions:

            resultNoCurvature = cls.solveOptimalControlProblem(pathToTrackData = pathNoCurvature,
                                                                trackDataFilename = os.path.splitext(filenameNoCurvature)[0],
                                                                energyOptimal = True,
                                                                lossFunction = lossFunction,
                                                                terminalTime = tripTime,
                                                                train = train,
                                                                finalPosition = finalPosition,
                                                                realLossFunction = realLosses)
            
            resultConstantCurvature = cls.solveOptimalControlProblem(pathToTrackData = pathConstantCurvature,
                                                                    trackDataFilename = os.path.splitext(filenameConstantCurvature)[0],
                                                                    energyOptimal = True,
                                                                    lossFunction = lossFunction,
                                                                    terminalTime = tripTime,
                                                                    train = train,
                                                                    finalPosition = finalPosition,
                                                                    realLossFunction = realLosses)

            resultNoCurvature, resultConstantCurvature = resultNoCurvature.reset_index(),  resultConstantCurvature.reset_index()

            totalEnergyNoCurvature, totalEnergyConstantCurvature = round(resultNoCurvature['Energy [kWh]'].sum(),1), round(resultConstantCurvature['Energy [kWh]'].sum(),1)
            
            energyLossesNoCurvature, energyLossesConstantCurvature = round(resultNoCurvature['Losses [kWh]'].sum(),1), round(resultConstantCurvature['Losses [kWh]'].sum(),1)

            mechanicalEnergyTrackNoCurvature, mechanicalEnergyTrackConstantCurvature = totalEnergyNoCurvature - energyLossesNoCurvature, totalEnergyConstantCurvature - energyLossesConstantCurvature

            mechanicalEnergyDelta = mechanicalEnergyTrackConstantCurvature - mechanicalEnergyTrackNoCurvature

            testsResults.append(abs(energyCurvatureResistance - mechanicalEnergyDelta) / energyCurvatureResistance)
        
        cls.assertTrue(all([result <= cls.THRESHOLD_ENERGY_ERROR_MINIMUM_ENERGY_PROBLEM for result in testsResults]))
    

if __name__ == '__main__':

    unittest.main()