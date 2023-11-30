import sys
sys.path.append('..')

import json

import matplotlib.pyplot as plt

from ocp import casadiSolver
from train import Train
from track import Track
from utils import latexify, show, saveFig, postProcessDataFrame
from efficiency import totalLossesFunction


def plot(df0b_list, df1b_list, df2b_list, tp, figSize=None, filename=None):

    latexFound = latexify()

    fig, ax = plt.subplots(2, 2)

    for indx,t in zip(range(len(df0b_list)),tp):

        ii = 0 if indx in {0,1} else 1
        jj = 0 if indx in {0,2} else 1

        energy0 = round(df0b_list[indx]['Energy [kWh]'].sum(),1)
        losses0 = round(df0b_list[indx]['Losses [kWh]'].sum(),1)

        l0=ax[ii][jj].plot(df0b_list[indx]['Position [m]']*1e-3, df0b_list[indx]['Velocity [m/s]']*3.6, '-.', color='tab:blue', label='Perfect efficiency ({}/{} kWh)'.format(energy0, losses0))
        ax[ii][jj].plot(df0b_list[indx]['Position - cvodes [m]']*1e-3, df0b_list[indx]['Velocity - cvodes [m/s]']*3.6, '-.', color='tab:blue')

        if df1b_list is not None:

            energy1 = round(df1b_list[indx]['Energy [kWh]'].sum(),1)
            losses1 = round(df1b_list[indx]['Losses [kWh]'].sum(),1)

            l1=ax[ii][jj].plot(df1b_list[indx]['Position [m]']*1e-3, df1b_list[indx]['Velocity [m/s]']*3.6, '--', color='tab:red', label='Static efficiency ({}/{} kWh)'.format(energy1, losses1))
            ax[ii][jj].plot(df1b_list[indx]['Position - cvodes [m]']*1e-3, df1b_list[indx]['Velocity - cvodes [m/s]']*3.6, '--', color='tab:red')

        else:

            l1 = []

        if df2b_list is not None:

            energy2 = round(df2b_list[indx]['Energy [kWh]'].sum(),1)
            losses2 = round(df2b_list[indx]['Losses [kWh]'].sum(),1)

            l2=ax[ii][jj].plot(df2b_list[indx]['Position [m]']*1e-3, df2b_list[indx]['Velocity [m/s]']*3.6, '-', color='tab:green', label='Dynamic efficiency ({}/{} kWh)'.format(energy2, losses2))
            ax[ii][jj].plot(df2b_list[indx]['Position - cvodes [m]']*1e-3, df2b_list[indx]['Velocity - cvodes [m/s]']*3.6, '-', color='tab:green')

        else:

            l2 = []

        if ii == 1:

            ax[ii][jj].set_xlabel('Position [km]')

        if jj == 0:

            ax[ii][jj].set_ylabel('Velocity [km/h]')

        ax[ii][jj].grid(visible=True)

        ax[ii][jj].legend(handles=l0+l1+l2, loc='lower right')

        percentSymbol = '\%' if latexFound else '%'
        ax[ii][jj].set_title(r'Time reserve: {}{}'.format(t, percentSymbol))

        ax[ii][jj].set_xlim([0, 8.5])
        ax[ii][jj].set_ylim([0, 145])

    if figSize is not None:

        fig.set_size_inches(figSize[0], figSize[1])

    fig.tight_layout()

    saveFig(fig, ax, filename)

    show()


if __name__ == '__main__':

    v0 = 1
    vN = 100/3.6

    train = Train(train='Intercity')
    train.forceMinPn = 0

    etaMax = 0.73

    fun0 = lambda f,v: 0
    fun1 = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
    fun2 = totalLossesFunction(train, auxiliaries=27000, etaGear=0.96)

    minimumTime = 272.4726

    df0b_list = []
    df1b_list = []
    df2b_list = []

    timeReserves = [0, 10, 20, 30]

    for tp in timeReserves:

        tripTime = minimumTime*(1 + tp/100)

        track = Track(config={'id':'00_var_speed_limit_100'}, tUpper=tripTime, pathJSON='../tracks')
        track.updateLimits(positionEnd=8500)

        with open('config.json') as file:

            solverOpts = json.load(file)

        solverOpts['minimumVelocity'] = min(v0, vN)

        # # # solve problem with perfect efficiency
        train.powerLosses = fun0

        ocp0 = casadiSolver(train, track, solverOpts)

        df0, _ = ocp0.solve(initialVelocity=v0, terminalVelocity=vN)

        # calculate actual losses
        train.powerLosses = fun2
        df0b = postProcessDataFrame(df0, ocp0.points, train, integrateLosses=True)
        df0b_list.append(df0b)

        # # # solve problem with constant efficiency
        train.powerLosses = fun1

        ocp1 = casadiSolver(train, track, solverOpts)

        df1, _ = ocp1.solve(initialVelocity=v0, terminalVelocity=vN)

        # calculate actual losses
        train.powerLosses = fun2
        df1b = postProcessDataFrame(df1, ocp1.points, train, integrateLosses=True)
        df1b_list.append(df1b)

        # # # solve problem with varying efficiency
        train.powerLosses = fun2

        ocp2 = casadiSolver(train, track, solverOpts)

        df2, _ = ocp2.solve(initialVelocity=v0, terminalVelocity=vN)

        # calculate actual losses
        train.powerLosses = fun2
        df2b = postProcessDataFrame(df2, ocp2.points, train, integrateLosses=True)
        df2b_list.append(df2b)

    plot(df0b_list, df1b_list, df2b_list, timeReserves, figSize=[7,5], filename='figure5.pdf')
