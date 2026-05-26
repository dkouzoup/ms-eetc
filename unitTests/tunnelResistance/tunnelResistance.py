import os
import unittest
from pathlib import Path

from ocp import casadiSolver
from track import Track
from train import Train


class TestTunnelResistance(unittest.TestCase):

    def testHigherEnergyConsumptionInTunnels(self):
        '''
        26 km long small tunnel with cross section of 24 m^2 on a track of 28 km results in significant higher energy consumption.
        '''

        startPosition = 0  # [m]
        endPosition = 28000  # [m]
        duration = 28000/(145/3.6)  # [s]

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='trains')

        track = Track(config={'id': 'test_flat_no_tunnel'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df1, stats1 = solver.solve(duration)

        energyConsuptionWithoutTunnel = stats1['Cost']

        track = Track(config={'id': 'test_flat_with_tunnel'}, pathJSON='tracks')
        track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

        opts = {'numIntervals': 300, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}
        solver = casadiSolver(train, track, opts)
        df2, stats2 = solver.solve(duration)

        energyConsuptionWithTunnel = stats2['Cost']

        self.assertTrue(energyConsuptionWithoutTunnel < energyConsuptionWithTunnel)