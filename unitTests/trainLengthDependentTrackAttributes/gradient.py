import unittest
from time import perf_counter_ns

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
        against a piecewise constant midpoint approximation of the gradient with increasing numbers of intervals.

        The piecewise constant approximation should converge to the piecewise linear
        result for both duration and final velocity.

        CVODES is used as the integrator.
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

            fig.suptitle("CVODES: PWC midpoint gradient approximation compared to PWL gradient", fontsize=14)

            ax1.axhline(pwlDuration, label="pwl")
            ax1.plot(intervalCounts, times, marker="o", color="orange", label="pwc midpoint")
            ax1.set_xlabel("Number of intervals of pwc gradient approximation")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.axhline(pwlVelocity, label="pwl")
            ax2.plot(intervalCounts, velocities, marker="o", color="orange", label="pwc midpoint")
            ax2.set_xlabel("Number of intervals of pwc gradient approximation")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            fig.tight_layout()
            plt.show()


    def test_integrator_RK(self):
        '''
        Track with a linearly increasing gradient over 1000 m.

        The result obtained using the piecewise linear gradient model is compared
        against a piecewise constant midpoint approximation of the gradient with increasing numbers of intervals.

        The piecewise constant approximation should converge to the piecewise linear
        result for both duration and final velocity.

        RK without time approximation is used as the integrator.
        RK substeps are increased until convergence
        '''

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')

        optsDict = {'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps': 0, 'numSteps': 1}}
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

            optsDict = {'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 0, 'numSteps': numStep}}
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
        pwcTimes = []
        pwcVelocities = []
        pwcIntervalCounts = []

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

            pwcIntervalCounts.append(numIntervals)
            pwcTimes.append(float(time))
            pwcVelocities.append(np.sqrt(float(velSq)))

        finalPwcDuration = pwcTimes[-1]
        finalPwcVelocity = pwcVelocities[-1]

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

            fig.suptitle("Explicit RK: PWL gradient with RK substeps vs PWC midpoint approximation", fontsize=14)

            ax1.plot(pwlIntervalCounts, pwlTimes, marker="o", label="PWL: RK substeps")
            ax1.plot(pwcIntervalCounts, pwcTimes, marker="o", color="orange", label="PWC: intervals")
            ax1.set_xlabel("Refinement level: PWC intervals and RK substeps")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(pwlIntervalCounts, pwlVelocities, marker="o", label="PWL: RK substeps")
            ax2.plot(pwcIntervalCounts, pwcVelocities, marker="o", color="orange", label="PWC: intervals")
            ax2.set_xlabel("Refinement level: PWC intervals and RK substeps")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            fig.tight_layout()
            plt.show()


    def test_integrator_RK_with_Time_Approx(self):
        '''
        Track with a linearly increasing gradient over 1000 m.

        The result obtained using the piecewise linear gradient model is compared
        against a piecewise constant midpoint approximation of the gradient.

        The piecewise constant approximation should converge to the piecewise linear
        result for both duration and final velocity.

        RK with time approximation is used as the integrator.
        RK uses 50 substeps.
        Time approx steps are increased until convergence
        '''

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        trainModel = train.exportModel()

        # Scenario
        time0 = 0
        velSq0 = 1
        ds = 1000
        traction = 0.8 * (train.forceMax / train.mass)

        initialGradient = 0
        finalGradient = 0.07

        numIntervals = 50
        timeApproxSteps = 30
        relativeTolerance = 1e-3
        plotDebug = True

        # PWL gradient reference
        pwlTimes = []
        pwlVelocities = []
        timeApproxStepCounts  = []

        for timeSteps in range(1, timeApproxSteps + 1):

            optsDict = {'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': timeSteps, 'numSteps': 50}}
            opts = OptionsCasadiSolver(optsDict)
            trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

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

            pwlTimes.append(float(out['time']))
            pwlVelocities.append(np.sqrt(float(out['velSquared'])))
            timeApproxStepCounts.append(timeSteps)

        finalPwlDuration = pwlTimes[-1]
        finalPwlVelocity = pwlVelocities[-1]

        # PWC gradients using midpoint rule
        pwcTimes = []
        pwcVelocities = []

        for timeSteps in range(1, timeApproxSteps + 1):

            optsDict = {'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': timeSteps, 'numSteps': 50}}
            opts = OptionsCasadiSolver(optsDict)
            trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

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

            pwcTimes.append(float(time))
            pwcVelocities.append(np.sqrt(float(velSq)))

        finalPwcDuration = pwcTimes[-1]
        finalPwcVelocity = pwcVelocities[-1]

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

            fig.suptitle("RK with Time Approx: PWC midpoint gradient approximation compared to PWL gradient", fontsize=14)

            ax1.plot(timeApproxStepCounts , pwlTimes, marker="o", label="pwl")
            ax1.plot(timeApproxStepCounts, pwcTimes, marker="o", color="orange", label="pwc midpoint")
            ax1.set_xlabel("Number of time approx steps")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.plot(timeApproxStepCounts , pwlVelocities, marker="o", label="pwl")
            ax2.plot(timeApproxStepCounts, pwcVelocities, marker="o", color="orange", label="pwc midpoint")
            ax2.set_xlabel("Number of time approx steps")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")
            ax2.ticklabel_format(axis="y", style="plain", useOffset=False)

            fig.tight_layout()
            plt.show()


    def testPWLProfile(self):
        '''
        Compare the final altitude of the original length-independent gradient profile
        with the train-length-dependent piecewise linear profile.

        Both profiles should start from the same altitude and end at the same target altitude.
        '''

        plotDebug = True
        altitudeTolerance = 1e-6

        trainLength = 800   # [m]
        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tracks')

        # track needs to be flat at least train length meters before the end of the track
        track.gradients = track.gradients[track.gradients.index < track.length - trainLength]
        track.gradients.loc[track.length - trainLength] = {"Gradient [permil]": 0.0, "Gradient linear term [permil/m]": 0.0}

        df_alt = computeAltitude(track.gradients, track.length)

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = trainLength
        track.updateTrainLengthDependentValues(train)

        positions = track.gradients.index.to_numpy(dtype=float)
        positions = np.r_[positions, track.length]

        gradient = track.gradients["Gradient [permil]"].to_numpy(dtype=float)
        gradientLinear = track.gradients["Gradient linear term [permil/m]"].to_numpy(dtype=float)

        ds = positions[1:] - positions[:-1]

        pwlAltitude = np.zeros(len(positions))
        pwlAltitude[1:] = np.cumsum((gradient * ds + 0.5 * gradientLinear * ds ** 2) / 1000)

        self.assertAlmostEqual(
            df_alt["Altitude [m]"].iloc[0],
            pwlAltitude[0],
            delta=altitudeTolerance,
            msg="Length-independent and train-length-dependent altitude profiles should start at the same altitude."
        )

        lengthIndependentFinalAltitude = df_alt["Altitude [m]"].iloc[-1]
        lengthDependentFinalAltitude = pwlAltitude[-1]

        self.assertAlmostEqual(
            lengthIndependentFinalAltitude,
            lengthDependentFinalAltitude,
            delta=altitudeTolerance,
            msg=(
                "Length-independent and train-length-dependent altitude profiles "
                "should end at the same altitude. "
                f"Length-independent final altitude: {lengthIndependentFinalAltitude:.8f} m, "
                f"train-length-dependent final altitude: {lengthDependentFinalAltitude:.8f} m."
            )
        )

        if plotDebug:

            fig, ax = plt.subplots(figsize=(16, 8))

            ax.plot(df_alt.index.values / 1000, df_alt["Altitude [m]"].to_numpy(), label="length-independent altitude")
            ax.plot(positions / 1000, pwlAltitude, label="length-dependent altitude")
            ax.set_title("Altitude Comparison")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Altitude [m]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()


    def testLinearGradient(self):
        '''
        Track with 20 permil increase from 1000 m to 2000 m and 20 permil
        decrease from 3000 m to 4000 m.

        Energy consumption should be roughly equal if computed using piecewise
        linear gradients or equivalent piecewise constant gradients due to a high number of shooting intervals.

        Altitude should be 0 m at the end.
        '''

        startPosition = 0 # [m]
        endPosition = 5000 # [m]
        duration = 5000/(60/3.6) # [s]

        altitudeTolerance = 1e-4
        energyRelativeTolerance = 1e-3
        numIntervals = 50

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_one_hill'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        # PWL Gradients
        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES'}
        solver = casadiSolver(train, track, opts)
        pwl_df, pwl_stats = solver.solve(duration)

        energyConsumptionWithLinearTerms = pwl_stats['Cost']

        # PWC Gradients
        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES', 'pwcLengthDependentTrackAttributes': True}
        solver = casadiSolver(train, track, opts)
        pwc_df, pwc_stats = solver.solve(duration)

        energyConsumptionWithPwcTerms= pwc_stats['Cost']

        relativeEnergyDifference = (abs(energyConsumptionWithLinearTerms - energyConsumptionWithPwcTerms) / energyConsumptionWithLinearTerms)

        pwc_df_grads = pwc_df.set_index("Position [m]")[["Gradient [permil]"]]
        pwc_df_alt = computeAltitude(pwc_df_grads, track.length)
        finalAltitude = pwc_df_alt.iloc[-1]["Altitude [m]"]

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

            df_grads_1 = pwl_df.set_index("Position [m]")[["Gradient [permil]"]]
            ax.plot(df_grads_1.index.values / 1000, df_grads_1["Gradient [permil]"],label="pwl gradients")
            ax.step(pwc_df_grads.index.values / 1000, pwc_df_grads["Gradient [permil]"], '--', where='post', label="pwc gradients")
            ax.set_title("Gradients")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Gradient [‰]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()


    def testSmallTrainLengthNotAffectingEnergyConsumption(self):
        '''
        Use an artificially short train on a real track profile.

        For a very small train length, the train-length-dependent gradient profile
        should be almost identical to the original gradient profile. Therefore, the
        energy consumption should remain within a small relative tolerance.
        '''

        relativeTolerance = 0.01
        trainLength = 10  # [m]
        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = trainLength

        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tracks')

        # track needs to be flat at least train length meters before the end of the track
        track.gradients = track.gradients[track.gradients.index < track.length - trainLength]
        track.gradients.loc[track.length - trainLength] = {"Gradient [permil]": 0.0, "Gradient linear term [permil/m]": 0.0}

        duration = track.length / (80/3.6)

        # train-length-independent
        opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':0}, 'energyOptimal':True}
        solver = casadiSolver(train, track, opts)
        indep_df, indep_stats = solver.solve(duration)

        track.updateTrainLengthDependentValues(train)
        solver = casadiSolver(train, track, opts)
        dep_df, dep_stats = solver.solve(duration)

        energyConsumptionIndependentOfTrainLength = indep_stats['Cost']
        energyConsumptionDependentOfTrainLength = dep_stats['Cost']

        relativeDifference = (abs(energyConsumptionDependentOfTrainLength - energyConsumptionIndependentOfTrainLength) / energyConsumptionIndependentOfTrainLength)

        self.assertLess(
            relativeDifference,
            relativeTolerance,
            msg=(
                "Energy consumption with and without train-length-dependent gradients should be roughly equal. "
                f"Independent: {energyConsumptionIndependentOfTrainLength:.6f}, "
                f"dependent: {energyConsumptionDependentOfTrainLength:.6f}, "
                f"relative difference: {relativeDifference:.6e}."
            )
        )