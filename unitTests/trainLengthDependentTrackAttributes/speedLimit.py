import unittest

from ocp import casadiSolver
from track import Track
from train import Train


class TestSpeedLimit(unittest.TestCase):

    def testTrainDoesNotAccelerateToEarly(self):
        '''
        Speed limit increases from 22 m/s to 40 m/s at position 1000 m.

        Without train-length-dependent values, the optimizer may accelerate too early.
        With train-length-dependent values, the front of the train may only exceed
        22 m/s after the whole train has passed the speed-increase position.
        '''

        startPosition = 0 # [m]
        endPosition = 12000 # [m]
        duration = 12000/(115/3.6) # [s]

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='trains')

        track = Track(config={'id': 'test_speed_increase'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df1, stats1 = solver.solve(duration)

        speedAfterSpeedIncrease1 = df1[df1['Position [m]'] > 1000].iloc[0]['Velocity [m/s]']

        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df2, stats2 = solver.solve(duration)

        speedAfterSpeedIncrease2 = df2[df2['Position [m]'] > 1000].iloc[0]['Velocity [m/s]']
        speedAfterSpeedIncrease3 = df2[df2['Position [m]'] > (1000+train.length)].iloc[0]['Velocity [m/s]']

        epsilon = 0.001

        self.assertGreater(
            speedAfterSpeedIncrease1,
            22,
            msg="Without train-length-dependent values, the train should accelerate immediately after the speed limit increase."
        )

        self.assertLessEqual(
            speedAfterSpeedIncrease2,
            22 + epsilon,
            msg="With train-length-dependent values, the train should not accelerate before the whole train has passed the speed limit increase."
        )

        self.assertGreater(
            speedAfterSpeedIncrease3,
            22,
            msg="With train-length-dependent values, the train should accelerate after the whole train has passed the speed limit increase."
        )