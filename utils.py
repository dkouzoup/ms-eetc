import matplotlib
import re

import numpy as np
import casadi as ca

from types import MethodType
from distutils.spawn import find_executable

def var(tag, dim=None):
    "Wrapper to create symbolic variables in casadi."

    v = ca.MX.sym(tag) if dim is None else \
        ca.vcat([ca.MX.sym('{}_{}'.format(tag, i)) for i in range(dim)])

    return v


def vecToList(x):
    "Convert a vector (numpy.array or casadi.DM) to list."

    try:

        x = x.full()  # convert DM to array

    except:

        pass  # already an array

    return x.flatten().tolist()


def vecToNum(x):
    "Convert a vector (numpy.array or casadi.DM) with one element to float."

    xList = vecToList(x)

    if len(xList) > 1:

        raise ValueError("Vector should have only one element!")

    return xList[0]


class Options():
    "Parent class for options of different solvers."

    def __init__(self, paramsDict):

        self.overwriteDefaults(paramsDict)
        self.checkValues()


    def checkValues(self):

        pass


    def checkPositiveInteger(self, num, fieldName, allowZero=True):

        if int(num) != num or (not allowZero and not num > 0) or (allowZero and not num >= 0):

            raise ValueError("{} must be a {} positive integer!".format(fieldName, 'strictly' if not allowZero else ''))


    def checkBounds(self, num, fieldName, lowerBound, upperBound):

        if not lowerBound <= num <= upperBound:

            raise ValueError("{} must be between {} and {}!".format(fieldName, lowerBound, upperBound))


    def overwriteDefaults(self, paramsDict):
        "Overwrite default options with user input."

        for key in paramsDict:

            if not hasattr(self, key):

                raise ValueError("Specified option ({}) does not exist!".format(key))

            else:

                if isinstance(getattr(self, key), Options):

                    if not isinstance(paramsDict[key], dict):

                        raise ValueError("Nested options must be specified as a dictionary!")

                    getattr(self, key).overwriteDefaults(paramsDict[key])

                else:

                    setattr(self, key, paramsDict[key])


    def toDict(self):

        attributes = [a for a in dir(self) if not a.startswith('__') and type(getattr(self, a)) != MethodType and a != 'ignoreFields']

        optsDict = {}

        for a in attributes:

            optsDict[a] = getattr(self, a) if not isinstance(getattr(self, a), Options) else getattr(self, a).toDict()

        return optsDict


class IVP():
    "Solve the initial value problem in time domain with CVODES."

    def __init__(self, model, lossesFun=None) -> None:

        # solver
        s = var('s')
        v = var('v')
        e = var('e') if lossesFun is not None else []

        f = var('force')
        gd = var('gradient')
        cr = var('curvature')
        dt = var('dt')
        m = var('mass') if lossesFun is not None else []

        rollingResistance = model.sr0 + model.sr1*v + model.sr2*v**2
        curvatureResistance = ca.if_else(ca.fabs(cr)<=1/300, model.g*0.5*ca.fabs(cr)/(1-30*ca.fabs(cr)),
                                         model.g*0.65*ca.fabs(cr)/(1-55*ca.fabs(cr)))
        acceleration = f - rollingResistance - model.g*gd*(1/model.rho) - curvatureResistance*(1/model.rho)

        ode = ca.vertcat(dt*v, dt*acceleration, dt*lossesFun(f*m, v)/m if lossesFun is not None else [])
        t0, tf = 0, 1
        self.fun = ca.integrator('integrator', 'cvodes', {'x':ca.vertcat(s,v,e), 'p':ca.vertcat(f,gd,cr,dt,m), 'ode':ode}, t0, tf,
                                 {'abstol':1e-12, 'reltol':1e-14})
        self.withLosses = lossesFun is not None


    def solve(self, tf, t0=0, f=0, grd=0, cr=0, v0=0, s0=0, m=None):

        if m is None and self.withLosses:

            raise ValueError("Need to specify total mass when integrating losses!")

        out = self.fun(x0=[s0, v0]+([0] if self.withLosses else []), p=[f, grd, cr, tf-t0] + ([m] if self.withLosses else []))['xf']

        sf = vecToNum(out[0])
        vf = vecToNum(out[1])
        ef = vecToNum(out[2]) if self.withLosses else None

        self.t0 = t0
        self.tf = tf
        self.f = f
        self.grd = grd
        self.cr = cr
        self.s0 = s0
        self.sf = sf
        self.v0 = v0
        self.vf = vf
        self.ef = ef

        return sf, vf


def simulateCVODES(dfIn, model, totalMass, accumulatedErrors=True):
    "Simulate optimal controls with CVODES integrator to compare integration errors."

    numIntervals = len(dfIn) - 1
    posCVODES = [dfIn['Position [m]'].iloc[0]]
    velCVODES = [dfIn['Velocity [m/s]'].iloc[0]]

    ivp = IVP(model)

    for ii in range(numIntervals):

        dt = dfIn.index[ii+1] - dfIn.index[ii]

        pos = posCVODES[-1] if accumulatedErrors else dfIn['Position [m]'].iloc[ii]
        vel = velCVODES[-1] if accumulatedErrors else dfIn['Velocity [m/s]'].iloc[ii]

        posNxt, velNxt = ivp.solve(tf=dt, f=dfIn['Force [N]'].iloc[ii]/totalMass, grd=dfIn['Gradient [permil]'].iloc[ii]/1e3,
                                   cr=dfIn['Curvature [1/m]'].iloc[ii], v0=vel, s0=pos)

        posCVODES += [posNxt]
        velCVODES += [velNxt]

    dfOut = dfIn.copy()

    dfOut['Position - cvodes [m]'] = posCVODES
    dfOut['Velocity - cvodes [m/s]'] = velCVODES

    dfOut['Error position [m]'] = abs(dfOut['Position - cvodes [m]'] - dfOut['Position [m]'])
    dfOut['Error velocity [m/s]'] = abs(dfOut['Velocity - cvodes [m/s]'] - dfOut['Velocity [m/s]'])

    return dfOut


def splitLosses(fun):
    "Split losses function in two parts that are differentiable at zero."

    tol = 1e-10

    f = var('f')
    v = var('v')
    jac = ca.Function('jacobian', [f,v], [ca.jacobian(fun(f,v), f)])

    def funTr(f,v):

        alpha1 = jac(tol, v)
        beta1 = fun(0, v)

        return fun(f,v)*(f >= 0) + (alpha1*f+beta1)*(f < 0)

    def funRgb(f,v):

        alpha2 = jac(-tol, v)
        beta2 = fun(0, v)

        return fun(f,v)*(f < 0) + (alpha2*f+beta2)*(f >= 0)

    return funTr, funRgb


def postProcessDataFrame(dfIn, points, train, CVODES=True, integrateLosses=False, integrateRollingResistance=False):

    unitScaling = 1e-6/3.6  # Nm -> kWh conversion
    totalMass = train.mass*train.rho

    dfOut = dfIn.copy()

    dfOut['Speed limit [m/s]'] = points['Speed limit [m/s]'].values
    dfOut['Gradient [permil]'] = points['Gradient [permil]'].values
    dfOut['Curvature [1/m]'] = points['Curvature [1/m]'].values
    dfOut['Force (acc) [N]'] = dfOut['Force (el) [N]']*(dfOut['Force (el) [N]'] >= 0)
    dfOut['Force (rgb) [N]'] = dfOut['Force (el) [N]']*(dfOut['Force (el) [N]'] < 0)
    dfOut['Force [N]'] = dfOut['Force (acc) [N]'] + dfOut['Force (rgb) [N]'] + dfOut['Force (pnb) [N]']
    tractionPowerAtBeginning = dfOut['Force (acc) [N]']*dfOut['Velocity [m/s]']/1e3
    tractionPowerAtEnd = dfOut['Force (acc) [N]']*dfOut['Velocity [m/s]'].shift(-1)/1e3
    regBrakePowerAtBeginning = dfOut['Force (rgb) [N]']*dfOut['Velocity [m/s]']/1e3
    regBrakePowerAtEnd = dfOut['Force (rgb) [N]']*dfOut['Velocity [m/s]'].shift(-1)/1e3
    dfOut['Max. Power [kW]'] = np.maximum(tractionPowerAtBeginning, tractionPowerAtEnd)
    dfOut['Min. Power [kW]'] = np.minimum(regBrakePowerAtBeginning, regBrakePowerAtEnd)

    metersPerInterval = dfOut['Position [m]'].diff().shift(-1)
    TractiveEnergyWheel = unitScaling*metersPerInterval*dfOut['Force (acc) [N]']
    BrakingEnergyWheel = -unitScaling*metersPerInterval*dfOut['Force (rgb) [N]']

    f = var('f')
    v = var('v')
    powerLosses = train.powerLosses
    fun = ca.Function('fun', [f, v], [powerLosses(f, v)/v])

    if not integrateLosses:

        dfOut['ds [m]'] = metersPerInterval
        dfOut['vm [m/s]'] = (dfOut['Velocity [m/s]'] + dfOut['Velocity [m/s]'].shift(-1))/2

        energy = lambda x: unitScaling*x['ds [m]']*powerLosses(x['Force (el) [N]'], x['vm [m/s]'])/x['vm [m/s]']

        dfOut['Losses [kWh]'] = dfOut.apply(energy, axis=1)

        dfOut.drop(columns=['ds [m]', 'vm [m/s]'], inplace=True)

    elif integrateLosses:

        from train import TrainIntegrator

        trainIntegrator = TrainIntegrator(train.exportModel(), 'RK')
        powTr, powRgb = splitLosses(train.powerLosses)
        trainIntegrator.initLosses(powTr, powRgb, totalMass, solver='CVODES')

        ts = dfOut.index.tolist()
        ss = dfOut['Position [m]'].values.tolist()
        vs = dfOut['Velocity [m/s]'].values.tolist()
        fs = (dfOut['Force (el) [N]']/totalMass).values.tolist()
        ps = (dfOut['Force (pnb) [N]']/totalMass).values.tolist()
        gs = (dfOut['Gradient [permil]']/1e3).values.tolist()
        cr = dfOut['Curvature [1/m]'].values.tolist()

        losses = []

        for jj in range(len(ts)-1):

            dt = ts[jj+1]-ts[jj]

            eTr, eRgb = trainIntegrator.calcLosses(vs[jj], dt, fs[jj], ps[jj], gs[jj], cr[jj])

            eEl = eTr if fs[jj] >= 0 else eRgb

            losses.append(totalMass*vecToNum(eEl))

        dfOut['Losses [kWh]'] = np.append(unitScaling*np.array(losses), None)

    dfOut['Energy [kWh]'] = TractiveEnergyWheel - BrakingEnergyWheel + dfOut['Losses [kWh]']

    dfOut['Energy (pnb) [kWh]'] = -unitScaling*metersPerInterval*dfOut['Force (pnb) [N]']
    dfOut['Energy (kin) [kWh]'] = unitScaling*0.5*train.mass*(dfOut['Velocity [m/s]']**2)

    if integrateRollingResistance:

        from train import TrainIntegrator

        trainIntegrator = TrainIntegrator(train.exportModel(), 'RK')
        trainIntegrator.initRollingResistance(solver='CVODES')

        ss = dfOut['Position [m]'].values.tolist()
        vs = dfOut['Velocity [m/s]'].values.tolist()
        fs = (dfOut['Force (acc) [N]']/totalMass).values.tolist()
        ps = (dfOut['Force (pnb) [N]']/totalMass).values.tolist()
        gs = (dfOut['Gradient [permil]']/1e3).values.tolist()
        cr = dfOut['Curvature [1/m]'].values.tolist()

        rr = []

        for jj in range(len(ss)-1):

            ds = ss[jj+1]-ss[jj]

            loss, _ = trainIntegrator.calcRollingResistance(vs[jj], ds, fs[jj], ps[jj], gs[jj], cr[jj])

            rr.append(totalMass*vecToNum(loss))

        dfOut['Rolling resistance [kWh]'] = np.append(unitScaling*np.array(rr), None)

    rrFun = lambda v: (train.r0 + train.r1*v + train.r2*v**2)/totalMass
    crFun = lambda cr: train.g*0.5*abs(cr)/((1-30*abs(cr))*train.rho)*(abs(cr)<=1/300) + train.g*0.65*abs(cr)/((1-55*abs(cr))*train.rho)*(abs(cr)>1/300)

    # instantaneous specific forces
    rollingResistance = dfOut['Velocity [m/s]'].apply(rrFun)
    gradientResistance = train.g*(dfOut['Gradient [permil]']/1000)/train.rho
    curvatureResistance = dfOut['Curvature [1/m]'].apply(crFun)

    dfOut['Acceleration [m/s^2]'] = dfOut['Force [N]']/totalMass - rollingResistance - gradientResistance - curvatureResistance

    if CVODES:

        dfOut = simulateCVODES(dfOut, train.exportModel(), totalMass)

    return dfOut


def checkTTOBenchVersion(jsonDict, supportedVersions):

    if not isinstance(supportedVersions, list) or not all([isinstance(x, str) for x in supportedVersions]):

        raise TypeError("'supportedVersions' must be specified a list of strings!")

    if 'metadata' not in jsonDict or 'library version' not in jsonDict['metadata']:

        raise ValueError("Library version not found in json file!")

    else:

        pattern = r'v([\d.]+)'
        match = re.search(pattern, jsonDict['metadata']['library version'])

        if match:

            version = match.group(1)

            if version not in supportedVersions:

                raise ValueError("Import function works only for library versions {}!".format(','.join(supportedVersions)))

        else:

            raise ValueError("Unexpected format of 'library version' in json file!")


def saveFig(fig, axs, filename):

    if filename is None:

        return

    matplotlib.pyplot.savefig(filename, bbox_inches='tight')


def show():

    matplotlib.pyplot.show()


def latexify():

    if find_executable('latex'):

        params = {
            "backend": "ps",
            "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "text.usetex": True,
            "font.family": "serif",
        }

        matplotlib.rcParams.update(params)

        latexFound = True

    else:

        latexFound = False

    return latexFound


if __name__ == '__main__':

    pass
