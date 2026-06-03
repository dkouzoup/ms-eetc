import unittest

from ocp import casadiSolver
from track import Track
from train import Train


class TestGradient(unittest.TestCase):

    def testAllIntegratorTypesWork(self):
        '''
        Verify that all supported integration methods produce consistent results
        for the same train, track, and optimization setup.

        The test compares RK, IRK, and CVODES, including the approximate time
        integration option for RK and IRK. The resulting energy costs should only
        differ by a small relative tolerance.
        '''

        startPosition = 0  # [m]
        endPosition = 5000  # [m]
        duration = 5000 / (60 / 3.6)  # [s]

        tol = 0.1
        numIntervals = 200

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_RK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0}}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_RK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 1}}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_IRK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 0}}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_IRK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES'}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_CVODES = stats['Cost']

        relDiff_RKApprox_IRKApprox = abs(energy_RK_Approx - energy_IRK_Approx) / energy_RK_Approx
        relDiff_RKApprox_CVODES = abs(energy_RK_Approx - energy_CVODES) / energy_RK_Approx
        relDiff_RK_IRK = abs(energy_RK - energy_IRK) / energy_RK

        self.assertLess(
            relDiff_RKApprox_IRKApprox,
            tol,
            msg=(
                "RK and IRK with numApproxSteps=1 should give similar costs. "
                f"RK approx: {energy_RK_Approx:.6f}, "
                f"IRK approx: {energy_IRK_Approx:.6f}, "
                f"relative difference: {relDiff_RKApprox_IRKApprox:.6f}."
            )
        )

        self.assertLess(
            relDiff_RKApprox_CVODES,
            tol,
            msg=(
                "RK with numApproxSteps=1 and CVODES should give similar costs. "
                f"RK approx: {energy_RK_Approx:.6f}, "
                f"CVODES: {energy_CVODES:.6f}, "
                f"relative difference: {relDiff_RKApprox_CVODES:.6f}."
            )
        )

        self.assertLess(
            relDiff_RK_IRK,
            tol,
            msg=(
                "RK and IRK with numApproxSteps=0 should give similar costs. "
                f"RK: {energy_RK:.6f}, "
                f"IRK: {energy_IRK:.6f}, "
                f"relative difference: {relDiff_RK_IRK:.6f}."
            )
        )