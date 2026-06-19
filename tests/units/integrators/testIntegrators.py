import unittest

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train


class TestGradient(unittest.TestCase):

    def testAllIntegratorTypesWork(self):
        '''
        Verify that all supported integration methods produce consistent results
        for the same train, track, and optimization setup.

        The test compares RK, IRK, and CVODES, including the approximate time
        integration option for RK and IRK. The resulting energy costs should only
        differ by a small relative tolerance.
        '''

        tol = 0.1
        numIntervals = 200

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='tests/fixtures/trains')
        train.length = 600

        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        journey = Journey(config={'id': 'CH_StGallen_Wil_Journey_01'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energy_RK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0}}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energy_RK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 1}}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energy_IRK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 0}}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energy_IRK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES'}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

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

    def testIntegratedLossesMatchMidpointApproximation(self):
        '''
        Verify that integrated drivetrain losses produce a similar energy cost as
        the midpoint loss approximation.

        The test solves the same energy-optimal problem twice: once with losses
        integrated along each interval and once with the midpoint approximation.
        The resulting energy costs should differ only by a small absolute tolerance.
        '''

        tol = 0.1
        numIntervals = 200

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='tests/fixtures/trains')

        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tests/fixtures/tracks')
        track.updateTrainLengthDependentValues(train)

        journey = Journey(config={'id': 'CH_StGallen_Wil_Journey_01'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'integrateLosses': True}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energyWithLossIntegration = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}}
        solver = casadiSolver(train, track, journey, opts)
        df, stats = solver.solve()

        energyWithMidpointApproximation = stats['Cost']

        self.assertAlmostEqual(
            energyWithLossIntegration,
            energyWithMidpointApproximation,
            delta=tol,
            msg=(
                f"Energy cost with integrated drivetrain losses is not within the expected tolerance of the midpoint loss approximation. "
                f"Integrated losses: {energyWithLossIntegration:.2f} kWh, "
                f"midpoint approximation: {energyWithMidpointApproximation:.2f} kWh, "
                f"allowed tolerance: ±{tol:.2f} kWh."
            )
        )