import unittest

from mseetc.journey import Journey
from mseetc.ocp import casadiSolver
from mseetc.track import Track
from mseetc.train import Train


class TestTunnelResistance(unittest.TestCase):

    def test_tunnel_resistance_increases_energy_consumption(self):
        '''
        26 km long small tunnel with cross section of 24 m^2 on a track of 28 km results in significant higher energy consumption.
        '''

        minEnergyRatio = 1.5

        train = Train(config={'id': 'CH_Stadler_Flirt_TPF'}, pathJSON='tests/fixtures/trains')

        trackWithoutTunnel = Track(config={'id': 'test_flat_no_tunnel'}, pathJSON='tests/fixtures/tracks')

        journey = Journey(config={'id': 'test_flat_no_tunnel_Journey_01'}, pathJSON='tests/fixtures/journeys')
        trackWithoutTunnel.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, trackWithoutTunnel, journey, opts)
        dfWithoutTunnel, statsWithoutTunnel = solver.solve()

        energyConsumptionWithoutTunnel = statsWithoutTunnel['Cost']

        trackWithTunnel = Track(config={'id': 'test_flat_with_tunnel'}, pathJSON='tests/fixtures/tracks')

        journey = Journey(config={'id': 'test_flat_with_tunnel_Journey_01'}, pathJSON='tests/fixtures/journeys')
        trackWithTunnel.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, trackWithTunnel, journey, opts)
        dfWithTunnel, statsWithTunnel = solver.solve()

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