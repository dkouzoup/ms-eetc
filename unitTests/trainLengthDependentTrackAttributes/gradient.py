import unittest

from matplotlib import pyplot as plt

from ocp import casadiSolver
from track import Track, computeAltitude
from train import Train


class TestGradient(unittest.TestCase):

    def testLinearGradient(self):
        '''
        Track with 20 permil increase from 1000 m to 2000 m and 20 permil
        decrease from 3000 m to 4000 m.

        Energy consumption should be roughly equal if computed using piecewise
        linear gradients or equivalent piecewise constant gradients.

        Altitude should be 0 m at the end.
        '''

        startPosition = 0 # [m]
        endPosition = 5000 # [m]
        duration = 5000/(60/3.6) # [s]

        altitudeTolerance = 1e-4
        energyRelativeTolerance = 0.004
        numIntervals = 100

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_one_hill'}, pathJSON='tracks')
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

        df_grads_2 = df2.set_index("Position [m]")[["Gradient [permil]"]]
        df_alt_2 = computeAltitude(df_grads_2, track.length)
        finalAltitude = df_alt_2.iloc[-1]["Altitude [m]"]

        self.assertLess(
            relativeEnergyDifference,
            energyRelativeTolerance,
            msg=(
                "Energy consumption differs too much between piecewise linear and "
                f"piecewise constant gradients. Relative difference: "
                f"{relativeEnergyDifference:.6f}, tolerance: {energyRelativeTolerance:.6f}. "
                f"PWL cost: {energyConsumptionWithLinearTerms:.6f}, "
                f"PWC cost: {energyConsumptionWithPwcTerms:.6f}."
            )
        )

        self.assertLess(
            abs(finalAltitude),
            altitudeTolerance,
            msg=(
                "Final altitude should be close to 0 m. "
                f"Final altitude: {finalAltitude:.8f} m, "
                f"tolerance: {altitudeTolerance:.8f} m."
            )
        )

        plotDebug = True

        if plotDebug:

            fig, ax = plt.subplots(figsize=(16, 8))

            df_grads_1 = df1.set_index("Position [m]")[["Gradient [permil]"]]
            ax.plot(df_grads_1.index.values / 1000, df_grads_1["Gradient [permil]"],label="pwl gradients")
            ax.step(df_grads_2.index.values / 1000, df_grads_2["Gradient [permil]"], '--', where='post', label="pwc gradients")
            ax.set_title("Gradients")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Gradient [‰]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()

    def testAllIntegratorsWork(self):
        startPosition = 0
        endPosition = 12000