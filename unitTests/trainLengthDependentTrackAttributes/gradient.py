import unittest

import numpy as np
from matplotlib import pyplot as plt

from ocp import casadiSolver, OptionsCasadiSolver
from track import Track, computeAltitude
from train import Train, TrainIntegrator


class TestGradient(unittest.TestCase):

    def test_integrator_CVODES(self):
        '''
        Track with a linearly increasing gradient over 1000 m.

        The result obtained using the piecewise linear gradient model is compared
        against a piecewise constant midpoint approximation with increasing numbers
        of intervals.

        The piecewise constant approximation should converge to the piecewise linear
        result for both duration and final velocity.
        '''

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')

        optsDict = {'integrationMethod': 'CVODES'}
        opts = OptionsCasadiSolver(optsDict)

        trainModel = train.exportModel()
        trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

        # Scenario
        time0 = 0
        velSq0 = 1
        ds = 1000
        traction = 0.8 * (train.forceMax / train.mass)

        initialGradient = 0
        finalGradient = 0.07

        maxIntervals = 50
        relativeTolerance = 1e-3
        plotDebug = True

        # PWL gradient reference
        out = trainIntegrator.solve(
            time=time0,
            velocitySquared=velSq0,
            ds=ds,
            traction=traction,
            pnBrake=0,
            gradient=initialGradient,
            gradientLinearTerm=(finalGradient - initialGradient) / ds,
            curvature=0,
            curvatureLinearTerm=0,
            tunnelFactor=0
        )

        pwlDuration = float(out['time'])
        pwlVelocity = np.sqrt(float(out['velSquared']))

        # PWC gradients using midpoint rule
        times = []
        velocities = []
        intervalCounts = []

        for numIntervals in range(1, maxIntervals + 1):

            time = time0
            velSq = velSq0

            for idx in range(numIntervals):

                gradient = (initialGradient + (idx + 0.5) * (finalGradient - initialGradient) / numIntervals)

                out = trainIntegrator.solve(
                    time=time,
                    velocitySquared=velSq,
                    ds=ds / numIntervals,
                    traction=traction,
                    pnBrake=0,
                    gradient=gradient,
                    gradientLinearTerm=0,
                    curvature=0,
                    curvatureLinearTerm=0,
                    tunnelFactor=0
                )

                time = out['time']
                velSq = out['velSquared']

            intervalCounts.append(numIntervals)
            times.append(float(time))
            velocities.append(np.sqrt(float(velSq)))

        finalPwcDuration = times[-1]
        finalPwcVelocity = velocities[-1]

        relativeDurationError = abs(finalPwcDuration - pwlDuration) / pwlDuration
        relativeVelocityError = abs(finalPwcVelocity - pwlVelocity) / pwlVelocity

        self.assertLess(
            relativeDurationError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently "
                "to the PWL duration. "
                f"PWL duration: {pwlDuration:.6f}, "
                f"PWC duration: {finalPwcDuration:.6f}, "
                f"relative error: {relativeDurationError:.6e}."
            )
        )

        self.assertLess(
            relativeVelocityError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently "
                "to the PWL final velocity. "
                f"PWL velocity: {pwlVelocity:.6f}, "
                f"PWC velocity: {finalPwcVelocity:.6f}, "
                f"relative error: {relativeVelocityError:.6e}."
            )
        )

        if plotDebug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            ax1.axhline(pwlDuration, label="pwl")
            ax1.plot(intervalCounts, times, marker="o", color="orange", label="pwc midpoint")
            ax1.set_xlabel("Number of intervals")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.axhline(pwlVelocity, label="pwl")
            ax2.plot(intervalCounts, velocities, marker="o", color="orange", label="pwc midpoint")
            ax2.set_xlabel("Number of intervals")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            fig.tight_layout()
            plt.show()


    def test_integrator_RK(self):
        '''
        Track with a linearly increasing gradient over 1000 m.

        The result obtained using the piecewise linear gradient model is compared
        against a piecewise constant midpoint approximation with increasing numbers
        of intervals.

        The piecewise constant approximation should converge to the piecewise linear
        result for both duration and final velocity.
        '''

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')

        optsDict = {'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':0, 'numSteps': 1}}
        opts = OptionsCasadiSolver(optsDict)

        trainModel = train.exportModel()
        trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

        # Scenario
        time0 = 0
        velSq0 = 1
        ds = 1000
        traction = 0.8 * (train.forceMax / train.mass)

        initialGradient = 0
        finalGradient = 0.07

        maxIntervals = 50
        relativeTolerance = 1e-3
        plotDebug = True

        # PWL gradient reference
        pwlTimes = []
        pwlVelocities = []
        pwlIntervalCounts = []

        for numStep in range(1, maxIntervals + 1):

            optsDict = {'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1, 'numSteps': numStep}}
            opts = OptionsCasadiSolver(optsDict)
            pwlTrainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

            out = pwlTrainIntegrator.solve(
                time=time0,
                velocitySquared=velSq0,
                ds=ds,
                traction=traction,
                pnBrake=0,
                gradient=initialGradient,
                gradientLinearTerm=(finalGradient - initialGradient) / ds,
                curvature=0,
                curvatureLinearTerm=0,
                tunnelFactor=0
            )

            pwlTimes.append(float(out['time']))
            pwlVelocities.append(np.sqrt(float(out['velSquared'])))
            pwlIntervalCounts.append(numStep)

        finalPwlDuration = pwlTimes[-1]
        finalPwlVelocity = pwlVelocities[-1]

        # PWC gradients using midpoint rule
        times = []
        velocities = []
        intervalCounts = []

        for numIntervals in range(1, maxIntervals + 1):

            time = time0
            velSq = velSq0

            for idx in range(numIntervals):

                gradient = (initialGradient + (idx + 0.5) * (finalGradient - initialGradient) / numIntervals)

                out = trainIntegrator.solve(
                    time=time,
                    velocitySquared=velSq,
                    ds=ds / numIntervals,
                    traction=traction,
                    pnBrake=0,
                    gradient=gradient,
                    gradientLinearTerm=0,
                    curvature=0,
                    curvatureLinearTerm=0,
                    tunnelFactor=0
                )

                time = out['time']
                velSq = out['velSquared']

            intervalCounts.append(numIntervals)
            times.append(float(time))
            velocities.append(np.sqrt(float(velSq)))

        finalPwcDuration = times[-1]
        finalPwcVelocity = velocities[-1]

        relativeDurationError = abs(finalPwcDuration - finalPwlDuration) / finalPwlDuration
        relativeVelocityError = abs(finalPwcVelocity - finalPwlVelocity) / finalPwlVelocity

        self.assertLess(
            relativeDurationError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently "
                "to the PWL duration. "
                f"PWL duration: {finalPwlDuration:.6f}, "
                f"PWC duration: {finalPwcDuration:.6f}, "
                f"relative error: {relativeDurationError:.6e}."
            )
        )

        self.assertLess(
            relativeVelocityError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently "
                "to the PWL final velocity. "
                f"PWL velocity: {finalPwlVelocity:.6f}, "
                f"PWC velocity: {finalPwcVelocity:.6f}, "
                f"relative error: {relativeVelocityError:.6e}."
            )
        )

        if plotDebug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            ax1.plot(pwlIntervalCounts, pwlTimes, marker="o", label="pwl")
            ax1.plot(intervalCounts, times, marker="o", color="orange", label="pwc midpoint")
            ax1.set_xlabel("Number of intervals")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(pwlIntervalCounts, pwlVelocities, marker="o", label="pwl")
            ax2.plot(intervalCounts, velocities, marker="o", color="orange", label="pwc midpoint")
            ax2.set_xlabel("Number of intervals")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            fig.tight_layout()
            plt.show()


    def testPWLProfile(self):
        '''
        Should result in same target altitude
        '''


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
        energyRelativeTolerance = 1e-4
        numIntervals = 100

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_one_hill'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df1, stats1 = solver.solve(duration)

        energyConsumptionWithLinearTerms = stats1['Cost']

        opts = {'numIntervals': numIntervals, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0}, 'energyOptimal': True, 'pwcLengthDependentTrackAttributes': True}
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