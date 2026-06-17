import unittest

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train


class TestSpeedLimit(unittest.TestCase):

    def test_train_length_dependent_speed_limit_delays_acceleration(self):
        '''
        Speed limit increases from 22 m/s to 40 m/s at position 1000 m.

        Without train-length-dependent values, the optimizer may accelerate too early.
        With train-length-dependent values, the front of the train may only exceed
        22 m/s after the whole train has passed the speed-increase position.
        '''

        speedIncreasePosition = 1000  # [m]
        speedLimitBeforeIncrease = 22  # [m/s]
        speedTolerance = 0.001  # [m/s]

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='tests/fixtures/trains')

        track = Track(config={'id': 'test_one_speed_increase'}, pathJSON='tests/fixtures/tracks')

        journey = Journey(config={'id': 'test_one_speed_increase_Journey_01'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, journey, opts)
        dfWithoutTrainLength, statsWithoutTrainLength = solver.solve()

        speedWithoutTrainLengthAfterIncrease = dfWithoutTrainLength[dfWithoutTrainLength['Position [m]'] > speedIncreasePosition].iloc[0]['Velocity [m/s]']

        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, journey, opts)
        dfWithTrainLength, statsWithTrainLength = solver.solve()

        speedWithTrainLengthAfterIncrease = dfWithTrainLength[dfWithTrainLength['Position [m]'] > speedIncreasePosition].iloc[0]['Velocity [m/s]']
        speedWithTrainLengthAfterTrainPassedIncrease = dfWithTrainLength[dfWithTrainLength['Position [m]'] > (speedIncreasePosition+train.length)].iloc[0]['Velocity [m/s]']

        self.assertGreater(
            speedWithoutTrainLengthAfterIncrease,
            speedLimitBeforeIncrease,
            msg="Without train-length-dependent values, the train should accelerate immediately after the speed limit increase."
        )

        self.assertLessEqual(
            speedWithTrainLengthAfterIncrease,
            speedLimitBeforeIncrease + speedTolerance,
            msg="With train-length-dependent values, the train should not accelerate before the whole train has passed the speed limit increase."
        )

        self.assertGreater(
            speedWithTrainLengthAfterTrainPassedIncrease,
            speedLimitBeforeIncrease,
            msg="With train-length-dependent values, the train should accelerate after the whole train has passed the speed limit increase."
        )