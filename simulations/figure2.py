import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import casadi as ca

import matplotlib.pyplot as plt

from utils import vecToNum, IVP, latexify, show, saveFig
from train import Train, TrainIntegrator


if __name__ == '__main__':

    # simulation parameters

    train = Train(train='Intercity')

    f0 = 0.5  # traction force [N/kg]
    f1 = -f0  # reg. braking force [N/kg]
    t0 = 0  # initial time [s]
    v0a = 1  # initial speed for acceleration (low) [km/h]
    v0b = 10  # initial speed for acceleration (high) [km/h]
    v0c = 36.61894  # initial speed for braking (low) [km/h]
    v0d = 37.95880 # initial speed for braking (high) [km/h]

    ds = 100  # traveled distance [m]
    ds2 = 1  # high-resolution integration interval for cvodes [m]
    ds3 = ds2*10  # step for increase of integration interval [m]

    if int(ds/ds2) != ds/ds2 or int(ds/ds3) != ds/ds3:

        raise ValueError("Length of interval should be an integer multiple of sub-intervals!")

    # integrate with CVODES in space (high-resolution)

    integratorCVODES = TrainIntegrator(train.exportModel(), 'CVODES', {'absTol':1e-12, 'relTol':1e-14})

    timeCVODESa = [t0]
    velCVODESa = [v0a]

    timeCVODESb = [t0]
    velCVODESb = [v0b]

    timeCVODESc = [t0]
    velCVODESc = [v0c]

    timeCVODESd = [t0]
    velCVODESd = [v0d]

    for index in range(int(ds/ds2)):

        # acceleration low

        tCur = timeCVODESa[-1]
        vCur = velCVODESa[-1]/3.6

        out = integratorCVODES.solve(time=tCur, velocitySquared=vCur**2, ds=ds2, traction=f0)

        timeCVODESa += [vecToNum(out['time'])]
        velCVODESa += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # acceleration high

        tCur = timeCVODESb[-1]
        vCur = velCVODESb[-1]/3.6

        out = integratorCVODES.solve(time=tCur, velocitySquared=vCur**2, ds=ds2, traction=f0)

        timeCVODESb += [vecToNum(out['time'])]
        velCVODESb += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # braking low

        tCur = timeCVODESc[-1]
        vCur = velCVODESc[-1]/3.6

        out = integratorCVODES.solve(time=tCur, velocitySquared=vCur**2, ds=ds2, traction=f1)

        timeCVODESc += [vecToNum(out['time'])]
        velCVODESc += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # braking high

        tCur = timeCVODESd[-1]
        vCur = velCVODESd[-1]/3.6

        out = integratorCVODES.solve(time=tCur, velocitySquared=vCur**2, ds=ds2, traction=f1)

        timeCVODESd += [vecToNum(out['time'])]
        velCVODESd += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

    df = pd.DataFrame({}, index=[i*ds2 for i in range(int(ds/ds2)+1)])

    df['Time (a) [s]'] = timeCVODESa
    df['Velocity (a) [km/h]'] = velCVODESa
    df['Time (b) [s]'] = timeCVODESb
    df['Velocity (b) [km/h]'] = velCVODESb
    df['Time (c) [s]'] = timeCVODESc
    df['Velocity (c) [km/h]'] = velCVODESc
    df['Time (d) [s]'] = timeCVODESd
    df['Velocity (d) [km/h]'] = velCVODESd

    # integrate with CVODES in time (to check result)

    dt = timeCVODESa[-1]-timeCVODESa[0]

    ivp = IVP(train.exportModel())
    out = ivp.solve(tf=timeCVODESa[-1], t0=timeCVODESa[0], f=f0, v0=v0a/3.6)

    tol = 1e-8

    if abs(out[0]-ds) > tol or abs(out[1]-velCVODESa[-1]/3.6) > tol:

        raise ValueError("Space discretization with CVODES not accurate enough!")

    dt = timeCVODESb[-1]-timeCVODESb[0]

    ivp = IVP(train.exportModel())
    out = ivp.solve(tf=timeCVODESb[-1], t0=timeCVODESb[0], f=f0, v0=v0b/3.6)

    if abs(out[0]-ds) > tol or abs(out[1]-velCVODESb[-1]/3.6) > tol:

        raise ValueError("Space discretization with CVODES not accurate enough!")

    dt = timeCVODESc[-1]-timeCVODESc[0]

    ivp = IVP(train.exportModel())
    out = ivp.solve(tf=timeCVODESc[-1], t0=timeCVODESc[0], f=f1, v0=v0c/3.6)

    if abs(out[0]-ds) > tol or abs(out[1]-velCVODESc[-1]/3.6) > tol:

        raise ValueError("Space discretization with CVODES not accurate enough!")

    dt = timeCVODESd[-1]-timeCVODESd[0]

    ivp = IVP(train.exportModel())
    out = ivp.solve(tf=timeCVODESd[-1], t0=timeCVODESd[0], f=f1, v0=v0d/3.6)

    if abs(out[0]-ds) > tol or abs(out[1]-velCVODESd[-1]/3.6) > tol:

        raise ValueError("Space discretization with CVODES not accurate enough!")

    # Simulate with integrators to be benchmarked

    orderIRK = 3

    integratorERK = TrainIntegrator(train.exportModel(), 'RK', {'order':4, 'numSteps':1, 'numApproxSteps':0})
    integratorERK1 = TrainIntegrator(train.exportModel(), 'RK', {'order':4, 'numSteps':1, 'numApproxSteps':1})
    integratorIRK = TrainIntegrator(train.exportModel(), 'IRK', {'order':orderIRK, 'numSteps':1, 'numApproxSteps':0})

    timeERK0a = []
    velERK0a = []
    timeERK1a = []
    velERK1a = []
    timeIRK0a = []
    velIRK0a = []

    timeERK0b = []
    velERK0b = []
    timeERK1b = []
    velERK1b = []
    timeIRK0b = []
    velIRK0b = []

    timeERK0c = []
    velERK0c = []
    timeERK1c = []
    velERK1c = []
    timeIRK0c = []
    velIRK0c = []

    timeERK0d = []
    velERK0d = []
    timeERK1d = []
    velERK1d = []
    timeIRK0d = []
    velIRK0d = []

    for index in range(int(ds/ds3)+1):

        # acceleration low

        out = integratorERK.solve(time=t0, velocitySquared=(v0a/3.6)**2, ds=ds3*index, traction=f0)

        timeERK0a += [vecToNum(out['time'])]
        velERK0a += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorERK1.solve(time=t0, velocitySquared=(v0a/3.6)**2, ds=ds3*index, traction=f0)

        timeERK1a += [vecToNum(out['time'])]
        velERK1a += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorIRK.solve(time=t0, velocitySquared=(v0a/3.6)**2, ds=ds3*index, traction=f0)

        timeIRK0a += [vecToNum(out['time'])]
        velIRK0a += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # acceleration high

        out = integratorERK.solve(time=t0, velocitySquared=(v0b/3.6)**2, ds=ds3*index, traction=f0)

        timeERK0b += [vecToNum(out['time'])]
        velERK0b += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorERK1.solve(time=t0, velocitySquared=(v0b/3.6)**2, ds=ds3*index, traction=f0)

        timeERK1b += [vecToNum(out['time'])]
        velERK1b += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorIRK.solve(time=t0, velocitySquared=(v0b/3.6)**2, ds=ds3*index, traction=f0)

        timeIRK0b += [vecToNum(out['time'])]
        velIRK0b += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # braking low

        pCur = ds - index*ds2*10
        tCur = df.loc[pCur]['Time (c) [s]']
        vCur = df.loc[pCur]['Velocity (c) [km/h]']

        out = integratorERK.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeERK0c += [vecToNum(out['time'])]
        velERK0c += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorERK1.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeERK1c += [vecToNum(out['time'])]
        velERK1c += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorIRK.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeIRK0c += [vecToNum(out['time'])]
        velIRK0c += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        # braking high

        pCur = ds - index*ds2*10
        tCur = df.loc[pCur]['Time (d) [s]']
        vCur = df.loc[pCur]['Velocity (d) [km/h]']

        out = integratorERK.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeERK0d += [vecToNum(out['time'])]
        velERK0d += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorERK1.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeERK1d += [vecToNum(out['time'])]
        velERK1d += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

        out = integratorIRK.solve(time=tCur, velocitySquared=(vCur/3.6)**2, ds=ds3*index, traction=f1)

        timeIRK0d += [vecToNum(out['time'])]
        velIRK0d += [vecToNum(ca.sqrt(out['velSquared']))*3.6]

    indx = [i*ds3 for i in range(int(ds/ds3)+1)]

    latexify()

    fig, ax = plt.subplots(2, 2)

    ax00L = ax[0][0]

    times = timeCVODESa[0::int(ds/ds3)]
    l1=ax00L.plot(indx, abs(np.array(timeERK0a)-np.array(times)), 's', fillstyle='none', color='tab:purple', label='ERK4')
    l2=ax00L.plot(indx, abs(np.array(timeIRK0a)-np.array(times)), 'x', fillstyle='none', color='tab:gray', label='IRK{}'.format(orderIRK))
    l3=ax00L.plot(indx, abs(np.array(timeERK1a)-np.array(times)), 'o', fillstyle='none', color='tab:green', label='ERK4+')

    ax00L.legend(handles=l1+l2+l3, loc='upper left')

    ax00L.set_xlabel('Integration interval [m]')
    ax00L.set_ylabel('Time error [s]')
    ax00L.grid(visible=True)
    ax00L.set_title('Acceleration from {} km/h'.format(v0a))
    ax00L.set_ylim([-5, 60])

    ax01L = ax[0][1]

    times = timeCVODESb[0::int(ds/ds3)]
    l4=ax01L.plot(indx, abs(np.array(timeERK0b)-np.array(times)), 's', fillstyle='none', color='tab:purple', label='ERK4')
    l5=ax01L.plot(indx, abs(np.array(timeIRK0b)-np.array(times)), 'x', fillstyle='none', color='tab:gray', label='IRK{}'.format(orderIRK))
    l6=ax01L.plot(indx, abs(np.array(timeERK1b)-np.array(times)), 'o', fillstyle='none', color='tab:green', label='ERK4+')

    ax01L.set_xlabel('Integration interval [m]')
    ax01L.set_ylabel('Time error [s]')
    ax01L.grid(visible=True)
    ax01L.set_title('Acceleration from {} km/h'.format(v0b))
    ax01L.set_ylim([-0.1, 1.3])

    ax10L = ax[1][0]
    l7=ax10L.plot(indx, abs(np.array(timeERK0c)-timeCVODESc[-1]), 's', fillstyle='none', color='tab:purple', label='ERK4')
    l8=ax10L.plot(indx, abs(np.array(timeIRK0c)-timeCVODESc[-1]), 'x', fillstyle='none', color='tab:gray', label='IRK{}'.format(orderIRK))
    l9=ax10L.plot(indx, abs(np.array(timeERK1c)-timeCVODESc[-1]), 'o', fillstyle='none', color='tab:green', label='ERK4+')

    ax10L.set_xlabel('Integration interval [m]')
    ax10L.set_ylabel('Time error [s]')
    ax10L.grid(visible=True)
    ax10L.set_title('Deceleration to {} km/h'.format(v0a))
    ax10L.set_ylim([-5, 60])

    ax11L = ax[1][1]
    l10=ax11L.plot(indx, abs(np.array(timeERK0d)-timeCVODESd[-1]), 's', fillstyle='none', color='tab:purple', label='ERK4')
    l11=ax11L.plot(indx, abs(np.array(timeIRK0d)-timeCVODESd[-1]), 'x', fillstyle='none', color='tab:gray', label='IRK{}'.format(orderIRK))
    l12=ax11L.plot(indx, abs(np.array(timeERK1d)-timeCVODESd[-1]), 'o', fillstyle='none', color='tab:green', label='ERK4+')

    ax11L.set_xlabel('Integration interval [m]')
    ax11L.set_ylabel('Time error [s]')
    ax11L.grid(visible=True)
    ax11L.set_title('Deceleration to {} km/h'.format(v0b))
    ax11L.set_ylim([-0.1, 1.3])

    fig.tight_layout()
    fig.set_size_inches(8, 5)

    saveFig(fig, ax, 'figure2.pdf')
    show()
