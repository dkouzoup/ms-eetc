import unittest

from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train


class TestTunnelResistance(unittest.TestCase):

    def test_tunnel_resistance_increases_energy_consumption(self):
        '''
        26 km long small tunnel with cross section of 24 m^2 on a track of 28 km results in significant higher energy consumption.
        '''

        startPosition = 0  # [m]
        endPosition = 28000  # [m]
        duration = 28000/(145/3.6)  # [s]

        minEnergyRatio = 1.5

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='trains')

        trackWithoutTunnel = Track(config={'id': 'test_flat_no_tunnel'}, pathJSON='tracks')
        trackWithoutTunnel.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, trackWithoutTunnel, opts)
        dfWithoutTunnel, statsWithoutTunnel = solver.solve(duration)

        energyConsumptionWithoutTunnel = statsWithoutTunnel['Cost']

        trackWithTunnel = Track(config={'id': 'test_flat_with_tunnel'}, pathJSON='tracks')
        trackWithTunnel.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, trackWithTunnel, opts)
        dfWithTunnel, statsWithTunnel = solver.solve(duration)

        energyConsumptionWithTunnel = statsWithTunnel['Cost']

        self.assertGreater(
            energyConsumptionWithTunnel,
            energyConsumptionWithoutTunnel,
            msg="Energy consumption with tunnel should be higher than without tunnel."
        )

        self.assertGreater(
            energyConsumptionWithTunnel / energyConsumptionWithoutTunnel,
            minEnergyRatio,
            msg=(
                "Energy consumption with tunnel should be significantly higher. "
                f"Expected ratio > {minEnergyRatio}, got "
                f"{energyConsumptionWithTunnel / energyConsumptionWithoutTunnel:.3f}."
            )

        )