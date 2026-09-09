import unittest

import numpy as np

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train, TrainIntegrator
from mseetc.utils import IVP, vecToNum


class TestIVP(unittest.TestCase):

    """
    The IVP class is the reference integration the optimized trajectories are checked
    against: postProcessDataFrame runs it over the solution and reports the discrepancy
    as 'Error position [m]' and 'Error velocity [m/s]'.

    That check is only worth anything while the IVP integrates the same model as the
    solver, so these tests pin the two against each other. Nothing else does: the tests
    in testIntegrators.py compare the solver's own integration methods, which all share
    the model and therefore agree even when the IVP does not.
    """

    def testIvpMatchesTheIntegratorOfTheSolver(self):
        '''
        Integrate one interval with the integrator of the solver and with the IVP, and
        verify that they end up in the same place at the same speed.

        Every track attribute is switched on, so that a term missing from either model
        shows up. The two integrate in different domains, the solver over distance and
        the IVP over time, so the time the solver reports for the interval is what the
        IVP is given. Converting between the two leaves a residual of about 3 cm that no
        amount of integration accuracy removes, hence the tolerances below.
        '''

        tolPosition = 0.1  # [m]
        tolVelocity = 0.02  # [m/s]

        ds = 200.0  # [m]
        initialVelocity = 22.0  # [m/s]
        traction = 0.25  # [m/s^2], specific force as used inside the solver

        attributes = {
            'gradient': 0.012,
            'gradientLinearTerm': 2.0e-4,
            'curvature': 1/400,
            'curvatureLinearTerm': 4.0e-6,
            'tunnelFactor': 1.5e-5,
        }

        train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='tests/fixtures/trains')
        train.forceMinPn = 0

        model = train.exportModel()

        integrator = TrainIntegrator(model, 'RK', {'numSteps': 20, 'order': 4, 'numApproxSteps': 0})
        out = integrator.solve(time=0, velocitySquared=initialVelocity**2, ds=ds,
                               traction=traction, **attributes)

        dt = vecToNum(out['time'])
        velocityReference = float(np.sqrt(vecToNum(out['velSquared'])))

        position, velocity = IVP(model).solve(
            tf=dt, f=traction, sRef=0, v0=initialVelocity, s0=0,
            grd=attributes['gradient'], grdLin=attributes['gradientLinearTerm'],
            cr=attributes['curvature'], crLin=attributes['curvatureLinearTerm'],
            tunnelFactor=attributes['tunnelFactor'])

        self.assertLess(
            abs(position - ds),
            tolPosition,
            msg=(
                "The IVP does not cover the same distance as the integrator of the solver, "
                "which means the two integrate different models. "
                f"Solver: {ds:.4f} m, IVP: {position:.4f} m, "
                f"difference: {abs(position - ds):.4f} m, allowed: {tolPosition} m."
            )
        )

        self.assertLess(
            abs(velocity - velocityReference),
            tolVelocity,
            msg=(
                "The IVP does not reach the same speed as the integrator of the solver, "
                "which means the two integrate different models. "
                f"Solver: {velocityReference:.4f} m/s, IVP: {velocity:.4f} m/s, "
                f"difference: {abs(velocity - velocityReference):.4f} m/s, allowed: {tolVelocity} m/s."
            )
        )


    def testCvodesCrossCheckStaysSmallInsideATunnel(self):
        '''
        Verify that the reference integration follows the optimized trajectory along a
        track with a tunnel.

        This covers the way from the solution to the IVP as well as the IVP itself: the
        tunnel factors are computed in postProcessDataFrame and have to reach
        simulateCVODES, which is a step that has been missed before. Without them the
        reference integration drifts off by kilometers rather than meters.
        '''

        tolPosition = 50  # [m], against roughly 7 m of honest integration error

        train = Train(config={'id': 'CH_Stadler_FLIRT_TPF'}, pathJSON='tests/fixtures/trains')

        track = Track(config={'id': 'test_flat_with_tunnel'}, pathJSON='tests/fixtures/tracks')

        journey = Journey(config={'id': 'test_flat_with_tunnel_Journey_01'}, pathJSON='tests/fixtures/journeys')
        track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': 200, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1},
                'energyOptimal': True}

        df, _ = casadiSolver(train, track, journey, opts).solve()

        maxError = df['Error position [m]'].max()

        self.assertLess(
            maxError,
            tolPosition,
            msg=(
                "The reference integration drifted away from the optimized trajectory, which "
                "means it is not being given the same track attributes as the solver. "
                f"Largest deviation: {maxError:.2f} m, allowed: {tolPosition} m."
            )
        )


if __name__ == '__main__':

    unittest.main()
