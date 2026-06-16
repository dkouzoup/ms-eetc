import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.ocp import casadiSolver, OptionsCasadiSolver
from mseetc.track import Track
from mseetc.train import Train, TrainIntegrator


plotDebug = False


def computeHeadingFromCurvature(curvatures, trackLength):

    positions = curvatures.index.to_numpy(dtype=float)

    if positions[-1] < trackLength:
        positions = np.r_[positions, trackLength]

    curvature = curvatures["Curvature [1/m]"].to_numpy(dtype=float)

    if "Curvature linear term [1/m^2]" in curvatures.columns:

        curvatureLinear = curvatures["Curvature linear term [1/m^2]"].to_numpy(dtype=float)

    else:

        curvatureLinear = np.zeros(len(curvature))

    ds = positions[1:] - positions[:-1]

    heading = np.zeros(len(positions))
    heading[1:] = np.cumsum(curvature * ds + 0.5 * curvatureLinear * ds**2)

    return pd.DataFrame(
        {"Heading [rad]": heading},
        index=positions
    )


class TestCurvature(unittest.TestCase):

    def test_cvodes_pwc_midpoint_curvature_converges_to_pwl_curvature(self):
        '''
        Track with a linearly increasing curvature over 1000 m.

        The result obtained using the piecewise linear curvature model is compared
        against a piecewise constant midpoint approximation of the curvature with increasing numbers of intervals.

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

        initialCurvature = 0
        finalCurvature = 0.005

        maxIntervals = 50
        relativeTolerance = 1e-4

        # PWL curvature reference
        out = trainIntegrator.solve(
            time=time0,
            velocitySquared=velSq0,
            ds=ds,
            traction=traction,
            pnBrake=0,
            gradient=0,
            gradientLinearTerm=0,
            curvature=initialCurvature,
            curvatureLinearTerm=(finalCurvature - initialCurvature) / ds,
            tunnelFactor=0
        )

        pwlDuration = float(out['time'])
        pwlVelocity = np.sqrt(float(out['velSquared']))

        # PWC curvature using midpoint rule
        times = []
        velocities = []
        intervalCounts = []

        for numIntervals in range(1, maxIntervals + 1):

            time = time0
            velSq = velSq0

            for idx in range(numIntervals):

                curvature = (initialCurvature + (idx + 0.5) * (finalCurvature - initialCurvature) / numIntervals)

                out = trainIntegrator.solve(
                    time=time,
                    velocitySquared=velSq,
                    ds=ds / numIntervals,
                    traction=traction,
                    pnBrake=0,
                    gradient=0,
                    gradientLinearTerm=0,
                    curvature=curvature,
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
                "PWC midpoint approximation did not converge sufficiently to the PWL duration. "
                f"PWL duration: {pwlDuration:.6f}, "
                f"PWC duration: {finalPwcDuration:.6f}, "
                f"relative error: {relativeDurationError:.6e}."
            )
        )

        self.assertLess(
            relativeVelocityError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently to the PWL final velocity. "
                f"PWL velocity: {pwlVelocity:.6f}, "
                f"PWC velocity: {finalPwcVelocity:.6f}, "
                f"relative error: {relativeVelocityError:.6e}."
            )
        )

        if plotDebug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            fig.suptitle("CVODES: PWC midpoint curvature approximation compared to PWL curvature", fontsize=14)

            ax1.axhline(pwlDuration, label="pwl")
            ax1.plot(intervalCounts, times, marker="o", color="orange", label="pwc midpoint")
            ax1.set_xlabel("Number of intervals of pwc curvature approximation")
            ax1.set_ylabel("Duration [s]")
            ax1.grid(True, which="both", linestyle="--", alpha=0.5)
            ax1.legend(loc="upper right")

            ax2.axhline(pwlVelocity, label="pwl")
            ax2.plot(intervalCounts, velocities, marker="o", color="orange", label="pwc midpoint")
            ax2.set_xlabel("Number of intervals of pwc curvature approximation")
            ax2.set_ylabel("Velocity [m/s]")
            ax2.grid(True, which="both", linestyle="--", alpha=0.5)
            ax2.legend(loc="upper right")

            fig.tight_layout()
            plt.show()


    def test_rk_pwc_midpoint_curvature_converges_to_pwl_curvature(self):
        '''
        Track with a linearly increasing curvature over 1000 m.

        The result obtained using the piecewise linear curvature model is compared
        against a piecewise constant midpoint approximation of the curvature with increasing numbers of intervals.

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

        initialCurvature = 0
        finalCurvature = 0.005

        maxIntervals = 50
        relativeTolerance = 1e-5

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
                gradient=0,
                gradientLinearTerm=0,
                curvature=initialCurvature,
                curvatureLinearTerm=(finalCurvature - initialCurvature) / ds,
                tunnelFactor=0
            )

            pwlTimes.append(float(out['time']))
            pwlVelocities.append(np.sqrt(float(out['velSquared'])))
            pwlIntervalCounts.append(numStep)

        finalPwlDuration = pwlTimes[-1]
        finalPwlVelocity = pwlVelocities[-1]

        # PWC curvature using midpoint rule
        pwcTimes = []
        pwcVelocities = []
        pwcIntervalCounts = []

        for numIntervals in range(1, maxIntervals + 1):

            time = time0
            velSq = velSq0

            for idx in range(numIntervals):

                curvature = (initialCurvature + (idx + 0.5) * (finalCurvature - initialCurvature) / numIntervals)

                out = trainIntegrator.solve(
                    time=time,
                    velocitySquared=velSq,
                    ds=ds / numIntervals,
                    traction=traction,
                    pnBrake=0,
                    gradient=0,
                    gradientLinearTerm=0,
                    curvature=curvature,
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
                "PWC midpoint approximation did not converge sufficiently to the PWL duration. "
                f"PWL duration: {finalPwlDuration:.6f}, "
                f"PWC duration: {finalPwcDuration:.6f}, "
                f"relative error: {relativeDurationError:.6e}."
            )
        )

        self.assertLess(
            relativeVelocityError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently to the PWL final velocity. "
                f"PWL velocity: {finalPwlVelocity:.6f}, "
                f"PWC velocity: {finalPwcVelocity:.6f}, "
                f"relative error: {relativeVelocityError:.6e}."
            )
        )

        if plotDebug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            fig.suptitle("Explicit RK: PWL curvature with RK substeps vs PWC midpoint approximation", fontsize=14)

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


    def test_rk_time_approx_pwc_midpoint_curvature_converges_to_pwl_curvature(self):
        '''
        Track with a linearly increasing curvature over 1000 m.

        The result obtained using the piecewise linear curvature model is compared
        against a piecewise constant midpoint approximation of the curvature.

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

        initialCurvature = 0
        finalCurvature = 0.005

        numIntervals = 50
        timeApproxSteps = 30
        relativeTolerance = 1e-3

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
                gradient=0,
                gradientLinearTerm=0,
                curvature=initialCurvature,
                curvatureLinearTerm=(finalCurvature - initialCurvature) / ds,
                tunnelFactor=0
            )

            pwlTimes.append(float(out['time']))
            pwlVelocities.append(np.sqrt(float(out['velSquared'])))
            timeApproxStepCounts.append(timeSteps)

        finalPwlDuration = pwlTimes[-1]
        finalPwlVelocity = pwlVelocities[-1]

        # PWC curvature using midpoint rule
        pwcTimes = []
        pwcVelocities = []

        for timeSteps in range(1, timeApproxSteps + 1):

            optsDict = {'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': timeSteps, 'numSteps': 50}}
            opts = OptionsCasadiSolver(optsDict)
            trainIntegrator = TrainIntegrator(trainModel, opts.integrationMethod, opts.integrationOptions.toDict())

            time = time0
            velSq = velSq0

            for idx in range(numIntervals):

                curvature = (initialCurvature + (idx + 0.5) * (finalCurvature - initialCurvature) / numIntervals)

                out = trainIntegrator.solve(
                    time=time,
                    velocitySquared=velSq,
                    ds=ds / numIntervals,
                    traction=traction,
                    pnBrake=0,
                    gradient=0,
                    gradientLinearTerm=0,
                    curvature=curvature,
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
                "PWC midpoint approximation did not converge sufficiently to the PWL duration. "
                f"PWL duration: {finalPwlDuration:.6f}, "
                f"PWC duration: {finalPwcDuration:.6f}, "
                f"relative error: {relativeDurationError:.6e}."
            )
        )

        self.assertLess(
            relativeVelocityError,
            relativeTolerance,
            msg=(
                "PWC midpoint approximation did not converge sufficiently to the PWL final velocity. "
                f"PWL velocity: {finalPwlVelocity:.6f}, "
                f"PWC velocity: {finalPwcVelocity:.6f}, "
                f"relative error: {relativeVelocityError:.6e}."
            )
        )

        if plotDebug:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

            fig.suptitle("RK with Time Approx: PWC midpoint curvature approximation compared to PWL curvature", fontsize=14)

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


    def test_train_length_dependent_curvature_preserves_target_heading(self):
        '''
        Compare the final heading of the original length-independent curvature profile
        with the train-length-dependent piecewise linear curvature profile.

        Both profiles should start from the same heading and end at the same target heading.
        '''

        headingTolerance  = 1e-6

        trainLength = 800   # [m]
        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tracks')

        # track needs to be straight at least train length meters before the end of the track
        track.curvatures = track.curvatures[track.curvatures.index < track.length - trainLength]
        track.curvatures.loc[track.length - trainLength] = {"Curvature [1/m]": 0.0}

        df_heading_indep = computeHeadingFromCurvature(track.curvatures, track.length)

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = trainLength
        track.updateTrainLengthDependentValues(train)

        df_heading_dep = computeHeadingFromCurvature(track.curvatures, track.length)

        self.assertAlmostEqual(
            df_heading_indep["Heading [rad]"].iloc[0],
            df_heading_dep["Heading [rad]"].iloc[0],
            delta=headingTolerance,
            msg="Length-independent and train-length-dependent curvature profiles should start with the same heading."
        )

        self.assertAlmostEqual(
            df_heading_indep["Heading [rad]"].iloc[-1],
            df_heading_dep["Heading [rad]"].iloc[-1],
            delta=headingTolerance,
            msg=(
                "Length-independent and train-length-dependent curvature profiles "
                "should end with the same heading. "
                f"Length-independent final heading: {df_heading_indep['Heading [rad]'].iloc[-1]:.12f} rad, "
                f"train-length-dependent final heading: {df_heading_dep['Heading [rad]'].iloc[-1]:.12f} rad."
            )
        )

        if plotDebug:

            fig, ax = plt.subplots(figsize=(16, 8))

            ax.plot(df_heading_indep.index.values / 1000, df_heading_indep["Heading [rad]"].to_numpy(), label="length-independent heading")
            ax.plot(df_heading_dep.index.values / 1000, df_heading_dep["Heading [rad]"].to_numpy(), label="length-dependent heading")
            ax.set_title("Heading  Comparison")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Heading [rad]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()


    def test_pwc_curvature_approximation_matches_pwl_curvature_energy(self):
        '''
        Track with right turn from 1000 m to 2000 m and a left turn from 3000 m to 4000 m.

        Energy consumption should be roughly equal if computed using piecewise
        linear curvatures or equivalent piecewise constant curvatures.
        '''

        startPosition = 0  # [m]
        endPosition = 5000  # [m]
        duration = 5000/(60/3.6)  # [s]

        energyRelativeTolerance = 1e-4
        numIntervals = 100

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = 600

        track = Track(config={'id': 'test_two_radii'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
        track.updateTrainLengthDependentValues(train)

        # PWL Curvatures
        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES'}
        solver = casadiSolver(train, track, opts)
        pwl_df, pwl_stats = solver.solve(duration)

        energyConsumptionWithLinearTerms = pwl_stats['Cost']

        # PWC Curvatures
        opts = {'numIntervals': numIntervals, 'integrationMethod': 'CVODES', 'pwcLengthDependentTrackAttributes': True}
        solver = casadiSolver(train, track, opts)
        pwc_df, pwc_stats = solver.solve(duration)

        energyConsumptionWithPwcTerms= pwc_stats['Cost']

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

        if plotDebug:

            fig, ax = plt.subplots(figsize=(16, 8))

            ax.plot(pwl_df["Position [m]"] / 1000, pwl_df["Curvature [1/m]"], label="pwl curvatures")
            ax.step(pwc_df["Position [m]"] / 1000, pwc_df["Curvature [1/m]"], "--", where="post", label="pwc curvatures")
            ax.set_title("Curvatures")
            ax.set_xlabel("Position [km]")
            ax.set_ylabel("Curvature [1/m]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlim(0, track.length / 1000)
            ax.figure.tight_layout()

            plt.show()


    def test_short_train_length_has_negligible_effect_on_curvature_energy_consumption(self):
        '''
        Use an artificially short train on a real track profile.

        For a very small train length, the train-length-dependent curvature profile
        should be almost identical to the original curvature profile. Therefore, the
        energy consumption should remain within a small relative tolerance.
        '''

        relativeTolerance = 1e-2
        trainLength = 10  # [m]
        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')
        train.length = trainLength

        track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='tracks')
        track.gradients = track.gradients.iloc[[0]]
        track.gradients["Gradient [permil]"].iloc[0] = 0

        # track needs to be straight at least train length meters before the end of the track
        track.curvatures = track.curvatures[track.curvatures.index < track.length - trainLength]
        track.curvatures.loc[track.length - trainLength] = {"Curvature [1/m]": 0.0}

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
                "Energy consumption with and without train-length-dependent curvature should be roughly equal. "
                f"Independent: {energyConsumptionIndependentOfTrainLength:.6f}, "
                f"dependent: {energyConsumptionDependentOfTrainLength:.6f}, "
                f"relative difference: {relativeDifference:.6e}."
            )
        )