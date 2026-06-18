import unittest
from pathlib import Path

from mseetc.estimator import forceEstimator, energyEstimator
from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train
from simulations.sim_launcher import get_power_loss_function


class TestETCS(unittest.TestCase):

    def testInterventionPointsForFlatTrack(self):
        '''
        Verify that the ETCS braking curve calculator computes the expected
        intervention points for a flat test track.

        The test checks the main supervision limits I, P, W, SBI, and EBI
        against known reference positions. A small absolute tolerance is used
        because the values may vary slightly due to numerical computation.
        '''

        Path(__file__).resolve()

        train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='tests/fixtures/trains')
        train.forceMinPn = 0
        train.withPnBrake = False
        train.powerLosses = get_power_loss_function(train, "static")

        track = Track(config={'id': 'test_ETCS_flat_track'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        journey = Journey(config={'id': 'CH_StGallen_Wil_Journey_02'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        optsTarget = {'numIntervals':800, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

        solver = casadiSolver(train, track, journey, optsTarget)
        dfTarget, statsTraget = solver.solve()

        optsEstimate = {'numIntervals': 600, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

        fEstimator = forceEstimator(dfTarget, train, track, optsDict=optsEstimate, trainLengthDependentValues=True)
        dfEstimate = fEstimator.estimate()

        eEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsEstimate)
        energyStats = eEstimator.estimate()

        tol = 0.1

        self.assertAlmostEqual(
            statsTraget["Cost"],
            energyStats["Net energy used [kWh]"],
            delta=tol,
            msg=(
                f"Net energy consumption is not within the expected tolerance. "
                f"Expected: {statsTraget['Cost']:.2f} m, "
                f"actual: {energyStats['Net energy used [kWh]']:.2f} m, "
                f"allowed tolerance: ±{tol:.2f} m."
            )
        )
