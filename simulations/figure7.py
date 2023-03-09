
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from figure6 import runSimulation

from utils import latexify, show, saveFig


def plotColormapZV(df_perfect, df_static, df_dynamic, train, figSize=None, filename=None):

    latexify()

    vMin = 0  # km/h
    vMax = train.velocityMax*3.6  # km/h

    velocities = np.linspace(vMin, vMax, 2*int(vMax-vMin))

    fMin = train.forceMin
    fMax = train.forceMax

    forces = np.linspace(fMin, fMax, int((fMax-fMin)/1000))

    velocities3D, forces3D = np.meshgrid(velocities, forces, indexing='ij')

    losses3D = np.zeros(velocities3D.shape)
    eta3D = np.zeros(velocities3D.shape)

    numRows, numCols = losses3D.shape

    for ii in range(numRows):

        for jj in range(numCols):

            vel = velocities3D[ii,jj]/3.6

            losses3D[ii,jj] = train.powerLosses(forces3D[ii,jj], vel)

            p1 = forces3D[ii,jj]*vel
            p2 = p1 + losses3D[ii,jj]

            eta3D[ii,jj] = (p1/p2) if forces3D[ii,jj]>=0 else p2/p1

    eta3D[eta3D>=1] = None
    eta3D[eta3D<0] = None  # regenerated power lower than losses

    tmpDf_static = pd.DataFrame({'Force (acc) [N]':df_static['Force (acc) [N]'], 'Force (rgb) [N]':df_static['Force (rgb) [N]'], 'Velocity [km/h]':df_static['Velocity [m/s]']*3.6}).set_index('Velocity [km/h]')
    tmpDf_dynamic = pd.DataFrame({'Force (acc) [N]':df_dynamic['Force (acc) [N]'], 'Force (rgb) [N]':df_dynamic['Force (rgb) [N]'], 'Velocity [km/h]':df_dynamic['Velocity [m/s]']*3.6}).set_index('Velocity [km/h]')
    tmpDf_perfect = pd.DataFrame({'Force (acc) [N]':df_perfect['Force (acc) [N]'], 'Force (rgb) [N]':df_perfect['Force (rgb) [N]'], 'Velocity [km/h]':df_perfect['Velocity [m/s]']*3.6}).set_index('Velocity [km/h]') if df_perfect is not None else 0*df_static

    threshold = 0  # N
    tmpDf_static['Force (acc) [N]'][tmpDf_static['Force (acc) [N]'] < threshold] = None
    tmpDf_dynamic['Force (acc) [N]'][tmpDf_dynamic['Force (acc) [N]'] < threshold] = None
    tmpDf_perfect['Force (acc) [N]'][tmpDf_perfect['Force (acc) [N]'] < threshold] = None

    tmpDf_static['Force (rgb) [N]'][tmpDf_static['Force (rgb) [N]'] > -threshold] = None
    tmpDf_dynamic['Force (rgb) [N]'][tmpDf_dynamic['Force (rgb) [N]'] > -threshold] = None
    tmpDf_perfect['Force (rgb) [N]'][tmpDf_perfect['Force (rgb) [N]'] > -threshold] = None

    fig, ax = plt.subplots(1,2)

    etaMax = np.ceil(np.nanmax(eta3D)*10)/10
    barrier = np.floor(0.7*etaMax*10)/10

    levels = np.linspace(0, barrier, round((barrier+0.1)/0.1)).tolist() + np.linspace(barrier+0.1, etaMax, 11).tolist()

    pcm0 = ax[0].contourf(velocities3D, 1e-3*forces3D, eta3D, levels)

    ax[0].plot(tmpDf_perfect.index, 1e-3*tmpDf_perfect['Force (acc) [N]'], zorder=1, marker='>', linestyle='dashed', linewidth=0.5, label='Perfect efficiency', color='tab:blue')
    ax[0].plot(tmpDf_static.index, 1e-3*tmpDf_static['Force (acc) [N]'], zorder=1, marker='s', linestyle='dashed', linewidth=0.5, label='Static efficiency', color='tab:red')
    ax[0].plot(tmpDf_dynamic.index, 1e-3*tmpDf_dynamic['Force (acc) [N]'], zorder=1, marker='o', linestyle='dashed', linewidth=0.5, label='Dynamic efficiency', color='tab:green')
    ax[0].set_title('Traction')
    ax[0].set_ylim([0, 1e-3*fMax])
    ax[0].legend(loc='upper right')

    ax[0].set_xlabel('Velocity [km/h]')
    ax[0].set_ylabel('Force [kN]')

    pcm1 = ax[1].contourf(velocities3D, 1e-3*forces3D, eta3D, levels)

    if df_perfect is not None:

        ax[1].plot(tmpDf_perfect.index, 1e-3*tmpDf_perfect['Force (rgb) [N]'], zorder=1, marker='>', linestyle='dashed', linewidth=0.5, label='Perfect efficiency', color='tab:blue')

    ax[1].plot(tmpDf_static.index, 1e-3*tmpDf_static['Force (rgb) [N]'], zorder=1, marker='s', linestyle='dashed', linewidth=0.5, label='Static efficiency', color='tab:red')
    ax[1].plot(tmpDf_dynamic.index, 1e-3*tmpDf_dynamic['Force (rgb) [N]'], zorder=1, marker='o', linestyle='dashed', linewidth=0.5, label='Dynamic efficiency', color='tab:green')
    ax[1].set_title('Regenerative braking')
    ax[1].set_ylim([1e-3*fMin, 0])

    ax[1].set_ylabel('Force [kN]')
    ax[1].set_xlabel('Velocity [km/h]')

    offset = 0.02
    box = ax[1].get_position()
    box.x0 = box.x0 + offset
    box.x1 = box.x1 + offset
    ax[1].set_position(box)

    box = ax[1].get_position()
    cb = fig.add_axes([box.x1 + 0.03, box.y0, 0.02, box.y1-box.y0])
    fig.colorbar(pcm1,cax=cb)

    if figSize is not None:

        fig.set_size_inches(figSize[0], figSize[1])

    saveFig(fig, ax, filename)

    show()


if __name__ == '__main__':

    df0, df0b, df1, df1b, df2, df2b, _, train = runSimulation()

    plotColormapZV(df0, df1, df2, train, figSize=[8, 4], filename='figure7.pdf')
