import unittest

from matplotlib import pyplot as plt

from ocp import casadiSolver
from track import Track, computeAltitude
from train import Train


class TestCurvature(unittest.TestCase):

    def testLinearCurvature(self):
        '''
        Track with right turn from 1000 m to 2000 m and a left turn from 3000 m to 4000 m.

        Energy consumption should be roughly equal if computed using piecewise
        linear curvatures or equivalent piecewise constant curvatures.
        '''

        startPosition = 0 # [m]
        endPosition = 5000 # [m]
        duration = 5000/(60/3.6) # [s]

        energyRelativeTolerance = 0.004
        numIntervals = 100

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_two_radii'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df1, stats1 = solver.solve(duration)

        energyConsumptionWithLinearTerms = stats1['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True, 'pwcLengthDependentTrackAttributes': True}
        solver = casadiSolver(train, track, opts)
        df2, stats2 = solver.solve(duration)

        energyConsumptionWithPwcTerms= stats2['Cost']

        relativeEnergyDifference = (abs(energyConsumptionWithLinearTerms - energyConsumptionWithPwcTerms) / energyConsumptionWithLinearTerms)

        self.assertLess(
            relativeEnergyDifference,
            energyRelativeTolerance,
            msg=(
                "Energy consumption differs too much between piecewise linear and "
                f"piecewise constant curvatures. Relative difference: "
                f"{relativeEnergyDifference:.6f}, tolerance: {energyRelativeTolerance:.6f}. "
                f"PWL cost: {energyConsumptionWithLinearTerms:.6f}, "
                f"PWC cost: {energyConsumptionWithPwcTerms:.6f}."
            )
        )


        plotDebug = True

        if plotDebug:

            fig, ax = plt.subplots(figsize=(16, 8))

            ax.plot(df1["Position [m]"] / 1000, df1["Curvature [1/m]"], label="pwl curvatures")
            ax.step(df2["Position [m]"] / 1000, df2["Curvature [1/m]"], "--", where="post", label="pwc curvatures")
            ax.set_title("Curvatures")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Curvature [1/m]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()

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

        tol = 0.02
        numIntervals = 100

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_two_radii'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1},'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_RK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0},'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_RK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 1},'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_IRK_Approx = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'IRK', 'integrationOptions': {'numApproxSteps': 0},'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df, stats = solver.solve(duration)

        energy_IRK = stats['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES', 'energyOptimal': True}
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