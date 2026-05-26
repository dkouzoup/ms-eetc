import unittest

from track import Track
from train import Train


class TestTunnelResistance(unittest.TestCase):

    def testHigherEnergyConsumptionInTunnels(self):
        '''

        '''

        train = Train(config={'id': 'Flirt_Tpf'}, pathJSON='../trains')

        track = Track(config={'id': 'CH_ZH_LU'}, pathJSON='../tracks')

        self.assertTrue()