import sys
sys.path.append('..')

import json

import numpy as np
import matplotlib.pyplot as plt

from utils import postProcessDataFrame, latexify, show, saveFig
from train import Train
from track import Track, computeAltitude
from ocp import casadiSolver
from efficiency import totalLossesFunction


def plotThreeTrajectories(df0, df1, df2, withSpeedLimits=False, withAltitude=False, figSize=None, \
    filename=None, losses0=None, losses1=None, losses2=None, energy0=None, energy1=None, energy2=None):

    latexify()

    fig, ax = plt.subplots(2, 1)

    def formatLegend(txt, e, l):

        withParenthesis = l is not None or e is not None

        leg = txt + (' (' if withParenthesis else '')
        leg += '{}'.format(round(e,1) if e is not None else '')
        leg += '/' if e is not None  and l is not None else ''
        leg += '{}'.format(round(l,1) if l is not None else '')
        leg += ' kWh)' if withParenthesis else ''

        return leg

    l0=ax[0].plot(df0['Position [m]']*1e-3, df0['Velocity [m/s]']*3.6, '-.', color='tab:blue', label=formatLegend('Perfect efficiency', energy0, losses0))
    ax[0].plot(df0['Position - cvodes [m]']*1e-3, df0['Velocity - cvodes [m/s]']*3.6, '-.', color='tab:blue')

    if df1 is not None:

        l1=ax[0].plot(df1['Position [m]']*1e-3, df1['Velocity [m/s]']*3.6, '--', color='tab:red', label=formatLegend('Static efficiency', energy1, losses1))
        ax[0].plot(df1['Position - cvodes [m]']*1e-3, df1['Velocity - cvodes [m/s]']*3.6, '--', color='tab:red')

    if df2 is not None:

        l2=ax[0].plot(df2['Position [m]']*1e-3, df2['Velocity [m/s]']*3.6, '-', color='tab:green', label=formatLegend('Dynamic efficiency', energy2, losses2))
        ax[0].plot(df2['Position - cvodes [m]']*1e-3, df2['Velocity - cvodes [m/s]']*3.6, '-', color='tab:green')

    if withSpeedLimits and withAltitude:

        le = ax[0].plot(np.NaN, np.NaN, '-', color='none', label=' ')  # phantom line to format legends properly

    if withSpeedLimits:

        lv = ax[0].step(df0['Position [m]']*1e-3, df0['Speed limit [m/s]']*3.6, '-', color='tab:purple', label='Speed limit', where='post')

    ax[0].set_ylabel('Velocity [km/h]')

    if withAltitude:

        axr = ax[0].twinx()

        gradients = df0.set_index('Position [m]')['Gradient [permil]']
        altitude = computeAltitude(gradients.iloc[:-1], gradients.index[-1])
        altitude -= altitude.min()[0]

        lg = axr.plot(gradients.index*1e-3, altitude.values, '-', color='tab:gray', label='Normalized altitude')
        axr.set_ylabel('Height [m]')

    l3=ax[1].step(df0['Position [m]']*1e-3, df0['Force [N]']*1e-3, '-.', color='tab:blue', label='Perfect efficiency', where='post')

    if df1 is not None:

        l4=ax[1].step(df1['Position [m]']*1e-3, df1['Force [N]']*1e-3, '--', color='tab:red', label='Static efficiency', where='post')

    if df2 is not None:

        l5=ax[1].step(df2['Position [m]']*1e-3, df2['Force [N]']*1e-3, '-', color='tab:green', label='Dynamic efficiency', where='post')

    ax[1].set_xlabel('Position [km]')
    ax[1].set_ylabel('Force [kN]')

    ax[0].grid(visible=True)
    ax[1].grid(visible=True)

    ax[0].set_xlim([0, df0['Position [m]'].iloc[-1]*1e-3])
    ax[1].set_xlim([0, df0['Position [m]'].iloc[-1]*1e-3])

    legends = l0 + (l1 if df1 is not None else []) + (l2 if df2 is not None else []) + (le if withSpeedLimits and withAltitude else []) + (lv if withSpeedLimits else []) + (lg if withAltitude else [])
    ax[0].legend(handles=legends, loc='lower left', ncol=2 if withAltitude and withSpeedLimits else 1)

    if figSize is not None:

        fig.set_size_inches(figSize[0], figSize[1])

    fig.tight_layout()

    saveFig(fig, ax, filename)

    show()


def runSimulation(trackID='00_var_speed_limit_100', nRuns=1, brakeType='rg'):

    v0 = 1
    vN = 1

    train = Train(train='Intercity')

    if brakeType == 'rg':

        train.forceMinPn = 0

    elif brakeType == 'pn':

        train.forceMin = 0

    else:

        raise ValueError("Unknown brake type!")

    etaMax = 0.73

    fun0 = lambda f,v: 0
    fun1 = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
    fun2 = totalLossesFunction(train, auxiliaries=27000, etaGear=0.96)

    tUpper = 1541 if trackID == '00_var_speed_limit_100' else 1242
    track = Track(config={'id':trackID}, tUpper=tUpper, pathJSON='../tracks')

    with open('config.json') as file:

        solverOpts = json.load(file)

    solverOpts['minimumVelocity'] = min(v0, vN)

    # # # solve problem with perfect efficiency
    train.powerLosses = fun0

    ocp0 = casadiSolver(train, track, solverOpts)

    cpuTmp0 = []
    iterTmp0 = []

    for ii in range(nRuns):

        df0, stats0, _ = ocp0.solve(initialVelocity=v0, terminalVelocity=vN)

        cpuTmp0 += [stats0['CPU time [s]']]
        iterTmp0 += [stats0['IP iterations']]

    numIter0, cpuTime0 = iterTmp0[0], min(cpuTmp0)

    # calculate actual losses
    train.powerLosses = fun2
    df0b = postProcessDataFrame(df0, ocp0.points, train, integrateLosses=True)

    # # # solve problem with constant efficiency
    train.powerLosses = fun1

    ocp1 = casadiSolver(train, track, solverOpts)

    cpuTmp1 = []
    iterTmp1 = []

    for _ in range(nRuns):

        df1, stats1, _ = ocp1.solve(initialVelocity=v0, terminalVelocity=vN)

        cpuTmp1 += [stats1['CPU time [s]']]
        iterTmp1 += [stats1['IP iterations']]

    numIter1, cpuTime1 = iterTmp1[0], min(cpuTmp1)

    # calculate actual losses
    train.powerLosses = fun2
    df1b = postProcessDataFrame(df1, ocp1.points, train, integrateLosses=True)

    # # # solve problem with varying efficiency
    train.powerLosses = fun2

    # solverOpts['integrateLosses'] = True
    ocp2 = casadiSolver(train, track, solverOpts)

    cpuTmp2 = []
    iterTmp2 = []

    for ii in range(nRuns):

        df2, stats2, _ = ocp2.solve(initialVelocity=v0, terminalVelocity=vN)

        cpuTmp2 += [stats2['CPU time [s]']]
        iterTmp2 += [stats2['IP iterations']]

    if not all([it==iterTmp0[0] for it in iterTmp0]) or not all([it==iterTmp1[0] for it in iterTmp1]) or not all([it==iterTmp2[0] for it in iterTmp2]):

        raise ValueError("Different number of iterations detected!")

    numIter2, cpuTime2 = iterTmp2[0], min(cpuTmp2)

    # calculate actual losses
    train.powerLosses = fun2
    df2b = postProcessDataFrame(df2, ocp2.points, train, integrateLosses=True)

    stats = {'numIter0':numIter0, 'numIter1':numIter1, 'numIter2':numIter2, \
        'cpuTime0':cpuTime0, 'cpuTime1':cpuTime1, 'cpuTime2':cpuTime2}

    return df0, df0b, df1, df1b, df2, df2b, stats, train


if __name__ == '__main__':

    df0, df0b, df1, df1b, df2, df2b, _, _ = runSimulation()

    actualLosses0 =  df0b['Losses [kWh]'].sum()
    actualLosses1 = df1b['Losses [kWh]'].sum()
    actualLosses2 = df2b['Losses [kWh]'].sum()

    actualEnergy0 =  df0b['Energy [kWh]'].sum()
    actualEnergy1 = df1b['Energy [kWh]'].sum()
    actualEnergy2 = df2b['Energy [kWh]'].sum()

    plotThreeTrajectories(df0, df1, df2, figSize=[8, 5], filename='figure6.pdf', withSpeedLimits=False, withAltitude=False, \
        losses0=actualLosses0, losses1=actualLosses1, losses2=actualLosses2, \
        energy0=actualEnergy0, energy1=actualEnergy1, energy2=actualEnergy2)
