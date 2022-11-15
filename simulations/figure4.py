import sys
sys.path.append('..')

import numpy as np

import matplotlib.pyplot as plt

from utils import latexify, show, saveFig
from efficiency import loadToForce, totalLossesFunction


def plotSpline(loads, velocities, splineFun, forceMax, powerMax, axObj=None):

    loads3D, velocities3D = np.meshgrid(loads, velocities, indexing='ij')

    forces3D = loadToForce(loads3D, velocities3D, forceMax, powerMax)

    losses3D = np.zeros(loads3D.shape)
    eta3D = np.zeros(loads3D.shape)

    rows, cols = losses3D.shape

    for ii in range(rows):

        for jj in range(cols):

            losses3D[ii,jj] = splineFun(forces3D[ii,jj], velocities3D[ii,jj])

    if axObj is None:

        axObj = plt.axes(projection='3d')

    axObj.plot_surface(1e-3*forces3D, velocities3D*3.6, 1e-3*losses3D, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    axObj.set_xlabel('Force [kN]')
    axObj.set_ylabel('Velocity [km/h]')
    axObj.set_zlabel('Losses [kW]')

    if axObj is None:

        show()

    return np.amax(losses3D)


def plotSplines(loads, velocities, splineFun1, splineFun2, forceMax, powerMax, plotEta=False, figSize=None, filename=None):

    latexify()

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')

    lossesMax1 = plotSpline(loads, velocities, splineFun1, forceMax, powerMax, axObj=ax1)
    lossesMax2 = plotSpline(loads, velocities, splineFun2, forceMax, powerMax, axObj=ax2)

    if figSize is not None:

        fig.set_size_inches(figSize[0], figSize[1])

    fig.tight_layout()

    saveFig(fig, [ax1, ax2], filename)

    show()

    return lossesMax1, lossesMax2


if __name__ == '__main__':

    from train import Train

    train = Train(train='Intercity')

    fun1 = lambda f,v: f*v*(f>0)*(1 - etaMax)/etaMax - (1-etaMax)*f*v*(f<0)
    fun2 = totalLossesFunction(train, auxiliaries=27000, etaGear=0.96)

    loadsEval = np.linspace(-100, 100, 200)
    velocitiesEval = np.linspace(1, 170, 170)/3.6

    etaMax = 0.73

    lossesMax1, lossesMax2 = plotSplines(loadsEval, velocitiesEval, fun1, fun2, train.forceMax, train.powerMax, figSize=[8, 4], filename='figure4.pdf')

    if not 0.99 <= lossesMax1/lossesMax2 <= 1.01:

        raise ValueError("Losses not close enough, tune etaMax!")
