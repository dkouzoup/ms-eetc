import sys
sys.path.append('..')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

from utils import latexify, show, saveFig
from efficiency import loadToForce, totalLossesFunction


def plotSpline(loads, velocities, splineFun, forceMax, powerMax, axObj=None, type='3d'):

    loads3D, velocities3D = np.meshgrid(loads, velocities, indexing='ij')

    forces3D = loadToForce(loads3D, velocities3D, forceMax, powerMax)

    losses3D = np.zeros(loads3D.shape)

    rows, cols = losses3D.shape

    for ii in range(rows):

        for jj in range(cols):

            losses3D[ii,jj] = splineFun(forces3D[ii,jj], velocities3D[ii,jj])

    if axObj is None:

        axObj = plt.axes(projection='3d')

    if type == '3d':

        handle = axObj.plot_surface(1e-3*forces3D, velocities3D*3.6, 1e-3*losses3D, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        axObj.set_zlabel('Losses [kW]')
        axObj.set_xlabel('Force [kN]')
        axObj.set_ylabel('Velocity [km/h]')

    elif type == 'heatmap':

        handle = axObj.contourf(velocities3D*3.6, 1e-3*forces3D, 1e-3*losses3D)
        axObj.set_xlabel('Velocity [km/h]')
        axObj.set_ylabel('Force [kN]')
        axObj.set_title('Losses [kW]')

    else:

        raise ValueError("Unknown type!")

    if axObj is None:

        show()

    return np.amax(losses3D), handle


def plotSplines(loads, velocities, splineFun1, splineFun2, forceMax, powerMax, plotEta=False, figSize=None, filename=None):

    latexify()

    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3,projection='3d')
    ax4 = fig.add_subplot(2,2,4,projection='3d')

    lossesMax1, h1 = plotSpline(loads, velocities, splineFun1, forceMax, powerMax, axObj=ax1, type='heatmap')
    lossesMax2, h2 = plotSpline(loads, velocities, splineFun2, forceMax, powerMax, axObj=ax2, type='heatmap')
    plotSpline(loads, velocities, splineFun1, forceMax, powerMax, axObj=ax3, type='3d')
    plotSpline(loads, velocities, splineFun2, forceMax, powerMax, axObj=ax4, type='3d')

    if figSize is not None:

        fig.set_size_inches(figSize[0], figSize[1])

    fig.tight_layout()

    width=0.34
    height=0.34
    boxOld = ax1.get_position()
    boxNew = Bbox.from_bounds(boxOld.x0+0.05, boxOld.y0, width, height)
    ax1.set_position(boxNew)
    boxOld = ax2.get_position()
    boxNew = Bbox.from_bounds(boxOld.x0, boxOld.y0, width, height)
    ax2.set_position(boxNew)

    # move colorbar next to ax2
    box = ax2.get_position()
    cb = fig.add_axes([box.x1 + 0.03, box.y0, 0.02, box.y1-box.y0])
    fig.colorbar(h2,cax=cb)

    saveFig(fig, [ax1, ax2], filename)

    show()

    return lossesMax1, lossesMax2


if __name__ == '__main__':

    from train import Train

    train = Train(config={'id':'NL_intercity_VIRM6'}, pathJSON='../trains')

    fun1 = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
    fun2 = totalLossesFunction(train, auxiliaries=27000, etaGear=0.96)

    loadsEval = np.linspace(-100, 100, 200)
    velocitiesEval = np.linspace(1, 170, 170)/3.6

    etaMax = 0.73

    lossesMax1, lossesMax2 = plotSplines(loadsEval, velocitiesEval, fun1, fun2, train.forceMax, train.powerMax, figSize=[8, 7], filename='figure3.pdf')

    if not 0.99 <= lossesMax1/lossesMax2 <= 1.01:

        raise ValueError("Losses not close enough, tune etaMax!")
