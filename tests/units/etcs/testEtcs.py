import unittest
from pathlib import Path

from mseetc.etcs import BrakingTarget, EtcsBrakingCurveCalculator
from mseetc.track import Track
from mseetc.train import Train


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

        track = Track(config={'id': 'test_ETCS_flat_track'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        target = BrakingTarget(
            position=5000,
            overlap=100,
            permittedVelocity=140/3.6,
            targetVelocity=0
        )


        calculator = EtcsBrakingCurveCalculator(train, track, distancePre=5000, distancePost=1000)
        _, interventionPoints = calculator.computeTarget(target)

        tol = 1.6

        expectedInterventionPoints = {
            "I": 2098.10,
            "P": 1748.10,
            "W": 1670.33,
            "SBI": 1592.55,
            "EBI": 1397.00,
        }

        for pointName, expectedValue in expectedInterventionPoints.items():

            actualValue = interventionPoints[pointName]

            self.assertAlmostEqual(
                actualValue,
                expectedValue,
                delta=tol,
                msg=(
                    f"Intervention point {pointName} is not within the expected tolerance. "
                    f"Expected: {expectedValue:.2f} m, "
                    f"actual: {actualValue:.2f} m, "
                    f"allowed tolerance: ±{tol:.2f} m."
                )
            )


    def testInterventionPointsForNonFlatTrack(self):
        '''
        Verify that the ETCS braking curve calculator computes the expected
        intervention points for a non-flat test track.

        The test checks the main supervision limits I, P, W, SBI, and EBI
        against known reference positions. A small absolute tolerance is used
        because the values may vary slightly due to numerical computation.
        '''

        Path(__file__).resolve()

        train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='tests/fixtures/trains')

        track = Track(config={'id': 'test_ETCS_non_flat_track'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        target = BrakingTarget(
            position=5000,
            overlap=100,
            permittedVelocity=140/3.6,
            targetVelocity=0
        )


        calculator = EtcsBrakingCurveCalculator(train, track, distancePre=5000, distancePost=1000)
        _, interventionPoints = calculator.computeTarget(target)

        tol = 1

        expectedInterventionPoints = {
            "I": 2139.76,
            "P": 1789.76,
            "W": 1711.98,
            "SBI": 1634.21
        }

        for pointName, expectedValue in expectedInterventionPoints.items():

            actualValue = interventionPoints[pointName]

            self.assertAlmostEqual(
                actualValue,
                expectedValue,
                delta=tol,
                msg=(
                    f"Intervention point {pointName} is not within the expected tolerance. "
                    f"Expected: {expectedValue:.2f} m, "
                    f"actual: {actualValue:.2f} m, "
                    f"allowed tolerance: ±{tol:.2f} m."
                )
            )

            tol = 8.5

            expectedInterventionPoints = {
                "EBI": 1491.69
            }

            for pointName, expectedValue in expectedInterventionPoints.items():
                actualValue = interventionPoints[pointName]

                self.assertAlmostEqual(
                    actualValue,
                    expectedValue,
                    delta=tol,
                    msg=(
                        f"Intervention point {pointName} is not within the expected tolerance. "
                        f"Expected: {expectedValue:.2f} m, "
                        f"actual: {actualValue:.2f} m, "
                        f"allowed tolerance: ±{tol:.2f} m."
                    )
                )

