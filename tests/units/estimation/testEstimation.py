import unittest
from pathlib import Path

from mseetc.estimator import forceEstimator, energyEstimator
from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train
from mseetc.utils import get_power_loss_function


class TestETCS(unittest.TestCase):

    def testInterventionPointsForFlatTrack(self):
        '''
        Verify that the force and energy estimation pipeline reproduces the
        optimized energy consumption on a flat test track.

        The test first computes an energy-optimal reference trajectory with the
        CasADi solver. Then, the force estimator reconstructs the traction and
        braking forces from this trajectory, and the energy estimator computes
        the resulting net energy consumption.

        The estimated net energy consumption is compared against the original net enrgy consumption.
        A small absolute tolerance is used because the values may vary slightly due to numerical approximation.
        '''

        Path(__file__).resolve()

        train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='tests/fixtures/trains')
        train.forceMinPn = 0
        train.withPnBrake = False
        train.powerLosses = get_power_loss_function(train, "static")

        track = Track(config={'id': 'test_ETCS_flat_track'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        journey = Journey(config={'id': 'test_ETCS_flat_track_Journey_01'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        optsTarget = {'numIntervals':800, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

        solver = casadiSolver(train, track, journey, optsTarget)
        dfTarget, statsTarget = solver.solve()

        optsEstimate = {'numIntervals': 600, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

        fEstimator = forceEstimator(dfTarget, train, track, optsDict=optsEstimate, trainLengthDependentValues=True)
        dfEstimate = fEstimator.estimate()

        eEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsEstimate)
        energyStats = eEstimator.estimate()

        tol = 0.1

        self.assertAlmostEqual(
            statsTarget["Cost"],
            energyStats["Net energy used [kWh]"],
            delta=tol,
            msg=(
                f"Net energy consumption is not within the expected tolerance. "
                f"Expected: {statsTarget['Cost']:.2f} kWh, "
                f"actual: {energyStats['Net energy used [kWh]']:.2f} kWh, "
                f"allowed tolerance: ±{tol:.2f} kWh."
            )
        )