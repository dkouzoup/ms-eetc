import sys
sys.path.append('..')

import json

import pandas as pd

from utils import postProcessDataFrame
from train import Train
from track import Track
from ocp import casadiSolver
from efficiency import totalLossesFunction


if __name__ == '__main__':

    v0 = 1
    vN = 1

    train = Train(train='Intercity')
    train.forceMinPn = 0

    fun = totalLossesFunction(train, auxiliaries=27000, etaGear=0.96)

    train.powerLosses = fun

    track = Track(config={'id':'00_var_speed_limit_100'}, tUpper=1541, pathJSON='../tracks')

    with open('config.json') as file:

        solverOpts = json.load(file)

    solverOpts['minimumVelocity'] = min(v0, vN)

    nRuns = 5
    N_table = [50, 100, 200, 300, 400, 1000, 5000]

    iters = []
    intervals = []
    cpuTimes = []
    expectedEnergies = []
    actualEnergies = []

    for numIntervals in N_table:

        solverOpts['numIntervals'] = numIntervals

        # # # solve problem with varying efficiency

        ocp = casadiSolver(train, track, solverOpts)

        cpuTmp = []
        iterTmp = []

        for ii in range(nRuns):

            df, stats, _ = ocp.solve(initialVelocity=v0, terminalVelocity=vN)

            cpuTmp += [stats['CPU time [s]']]
            iterTmp += [stats['IP iterations']]

        if not all([it==iterTmp[0] for it in iterTmp]):

            raise ValueError("Different number of iterations detected between runs!")

        interval, numIter, cpuTime = round(df['Position [m]'].diff().median()), iterTmp[0], min(cpuTmp)

        intervals += [interval]
        iters += [numIter]
        cpuTimes += [cpuTime]

        expectedEnergies.append(df['Energy [kWh]'].sum())

        # calculate actual losses
        dfb = postProcessDataFrame(df, ocp.points, train, integrateLosses=True)

        actualEnergies.append(dfb['Energy [kWh]'].sum())

    table = pd.DataFrame({'N':N_table, 'Interval size':intervals, 'Iterations':iters, 'CPU time [s]':cpuTimes}).set_index('N')
    table['Energy (expected [kWh])'] = expectedEnergies
    table['Energy (actual [kWh])'] = actualEnergies

    table = table.apply(lambda x: round(x,1))

    print(table)
