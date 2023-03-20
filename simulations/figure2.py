import sys
sys.path.append('..')

import numpy as np

import matplotlib.pyplot as plt

from utils import latexify, show, saveFig
from train import Train
from efficiency import motorLossesFunction, loadToForce


def plotSplines(train, data):

    latexFound = latexify()

    dfA = data['dfA']
    dfB = data['dfB']
    fun = data['fun']

    fig, ax = plt.subplots(1, 1)

    for l in dfA.keys():

        l1 = ax.plot(dfA.index*3.6, dfA[l]*1e-3, ':o', color='tab:blue', fillstyle='none', label='Measurements, configuration A')
        l2 = ax.plot(dfB.index*3.6, dfB[l]*1e-3, ':x', color='tab:red', fillstyle='none', label='Measurements, configuration B')

    velHighRes = np.linspace(dfA.index[0], dfA.index[-1], 200)

    loadHighRes = [0, 25, 50, 75, 90, 100]
    posText = [20, 60, 84, 117, 143, 168]

    for l,p in zip(loadHighRes, posText):

        l3 = ax.plot(velHighRes*3.6, np.array([fun(loadToForce(l, v, train.forceMax, train.powerMax), v) for v in velHighRes])*1e-3, '-', color='tab:green', label='Spline fit, minimum losses')

        percentSymbol = '\%' if latexFound else '%'
        ax.text(140, p, r'{}{} load'.format(l, percentSymbol), fontsize=10)

    ax.set_ylabel('Power losses [kW]')
    ax.set_xlabel('Velocity [km/h]')

    ax.grid(visible=True)

    ax.set_title('Losses of converters and motors')
    ax.legend(handles=l1+l2+l3, loc='upper center')

    fig.set_size_inches(7, 5)
    fig.tight_layout()

    saveFig(fig, ax, 'figure2.pdf')
    show()


if __name__ == '__main__':

    train = Train(train='Intercity')
    data = motorLossesFunction(train, detailedOutput=True)

    plotSplines(train, data)
