import os
import json
import numpy as np
import casadi as ca

from utils import Options, checkTTOBenchVersion, convertUnit, splitLosses

class Train():

    def __init__(self, config, pathJSON='trains') -> None:
        """
        Constructor of Train objects.
        """

        self.g = 9.81  # acceleration of gravity [m/s^2]

        # check config
        if not isinstance(config, dict):

            raise ValueError("Train configuration should be provided as a dictionary!")

        if 'id' not in config:

            raise ValueError("Train ID must be specified in configuration!")

        # open json file
        filename = os.path.join(pathJSON, config['id']+'.json')

        with open(filename) as file:

            data = json.load(file)

        checkTTOBenchVersion(data, ['1.1', '1.2', '1.3', '1.4'])

        # overwrite json data with config values if applicable

        # NOTE: optional fields that may be missing from json but specified in config
        optionalEntries = ["max acceleration", "max deceleration"]

        usedFields = set()
        config.pop('id')

        for entry in config:

            if config[entry] is None and entry in data:  # None -> no constraint

                data.pop(entry)

                usedFields.add(entry)

            else:

                if not isinstance(config[entry], dict) or config[entry].keys() != {'unit', 'value'}:

                    raise ValueError("Configuration field '{}' should be specified as a dictionary with 'unit' and 'value' keys!".format(entry))

                if entry in data or entry in optionalEntries:

                    data[entry] = config[entry]

                    usedFields.add(entry)

        if set(config) != usedFields:

            raise ValueError("Redundant fields in train configuration: {}!".format(', '.join(set(config) - usedFields)))

        # read data
        self.length = convertUnit(data['length']['value'], data['length']['unit'])  # train length [m]

        self.mass = convertUnit(data['mass']['value'], data['mass']['unit'])  # train mass [kg]

        self.rho = convertUnit(data['rho']['value'], data['rho']['unit'])  # rotating-mass factor [-]

        if self.rho < 1:

            self.rho += 1  # 6% -> 0.06 -> 1.06

        self.velocityMax = convertUnit(data['max speed']['value'], data['max speed']['unit'])  # maximum train speed [m/s]

        self.forceMax = convertUnit(data['max traction force']['value'], data['max traction force']['unit']) if 'max traction force' in data else None  # maximum traction force [N]

        self.forceMin = convertUnit(-abs(data['max reg braking force']['value']), data['max reg braking force']['unit']) if 'max reg braking force' in data else None  # maximum regenerative braking force [N]

        self.forceMinPn = convertUnit(-abs(data['max pn braking force']['value']), data['max pn braking force']['unit']) if 'max pn braking force' in data else None # maximum pneumatic braking force [N]

        self.powerMax = convertUnit(data['max traction power']['value'], data['max traction power']['unit']) if 'max traction power' in data else None  # maximum traction power [W]

        self.powerMin = convertUnit(-abs(data['max reg braking power']['value']), data['max reg braking power']['unit']) if 'max reg braking power' in data else None # maximum regenerative braking power [W]

        self.accMax = convertUnit(data['max acceleration']['value'], data['max acceleration']['unit']) if 'max acceleration' in data else None  # maximum acceleration [m/s^2]

        self.accMin = convertUnit(-abs(data['max deceleration']['value']), data['max deceleration']['unit']) if 'max deceleration' in data else None  # maximum allowed deceleration [m/s^2]

        self.r0 = convertUnit(data['rolling resistance r0']['value'], data['rolling resistance r0']['unit'])  # constant term [N]

        self.r1 = convertUnit(data['rolling resistance r1']['value'], data['rolling resistance r1']['unit'])  # linear term [N/(m/s)]

        self.r2 = convertUnit(data['rolling resistance r2']['value'], data['rolling resistance r2']['unit'])  # quadratic term [N/(m/s)^2]

        # TODO: unify with case of dynamic efficiency
        if 'efficiency traction' in data or 'efficiency reg brake' in data:

            if 'efficiency traction' not in data or 'efficiency reg brake' not in data:

                raise ValueError("Both efficiencies need to be specified in json file!")

            if 'values' in 'efficiency traction' or 'values' in 'efficiency reg brake':

                raise ValueError("Dynamic efficiency from json file not implemented yet!")

            self.etaTraction = convertUnit(data['efficiency traction']['value'], data['efficiency traction']['unit'])
            self.etaRgBrake = convertUnit(data['efficiency reg brake']['value'], data['efficiency reg brake']['unit'])

        if "tunnel resistance" in data:

            tunnel_data = data["tunnel resistance"] # additive aerodynamic tunnel drag as dict per tunnel cross section

            cross_section_unit = tunnel_data["units"]["cross section"] # tunnel cross section [m^2]
            resistance_unit = tunnel_data["units"]["resistance (or similar)"] # resistance coefficient [kg/m]

            self.tunnelCoefficients = {
                convertUnit(cross_section, cross_section_unit): convertUnit(
                    resistance,
                    resistance_unit
                )
                for cross_section, resistance in tunnel_data["values"]
            }

        else:

            self.tunnelCoefficients = {}

        self.checkFields()


    def checkFields(self):

        if self.length is None or self.length < 0 or np.isinf(self.length):

            raise ValueError("Train length must be a positive number, not {}!".format(self.length))

        if self.mass is None or self.mass < 0 or np.isinf(self.mass):

            raise ValueError("Train mass must be a positive number, not {}!".format(self.mass))

        if self.g is None or not 9 <= self.g <= 10:

            raise ValueError("Acceleration of gravity must be between 9 and 10 m/s^2, not {}!".format(self.g))

        if self.rho is None or not 1 <= self.rho <= 1.5:

            raise ValueError("Rotation mass factor must be between 1 and 1.5, not {}!".format(self.rho))

        if self.velocityMax is None or self.velocityMax <= 0 or np.isinf(self.velocityMax):

            raise ValueError("Maximum velocity must be a strictly positive number, not {}!".format(self.velocityMax))

        if self.forceMax is not None and (self.forceMax <= 0 or np.isinf(self.forceMax)):

            raise ValueError("Maximum traction force must be strictly positive or free (None), not {}!".format(self.forceMax))

        if self.forceMinPn is not None and (self.forceMinPn > 0 or np.isinf(self.forceMinPn)):

            raise ValueError("Maximum pneumatic braking force must be negative, zero or free (None), not {}!".format(self.forceMinPn))

        if self.forceMin is not None and (self.forceMin > 0 or np.isinf(self.forceMin)):

            raise ValueError("Maximum regenerative braking force must be negative, zero or free (None), not {}!".format(self.forceMin))

        if self.forceMin == 0 and self.forceMinPn == 0:

            raise ValueError("Both brakes cannot be deactivated simultaneously!")

        if self.powerMax is not None and (self.powerMax <= 0 or np.isinf(self.powerMax)):

            raise ValueError("Maximum traction power must be strictly positive or free (None), not {}!".format(self.powerMax))

        if self.powerMin is not None and (self.powerMin >= 0 or np.isinf(self.powerMin)):

            raise ValueError("Maximum regenerative brake power must be strictly negative or free (None), not {}!".format(self.powerMin))

        if self.accMax is not None and (self.accMax <= 0 or np.isinf(self.accMax)):

            raise ValueError("Maximum acceleration must be strictly positive or free (None), not {}!".format(self.accMax))

        if self.accMin is not None and (self.accMin >= 0 or np.isinf(self.accMin)):

            raise ValueError("Maximum deceleration must be strictly negative or free (None), not {}!".format(self.accMin))

        for ii in ['0', '1', '2']:

            coef = getattr(self, 'r'+ii)

            if coef is None or coef < 0:

                raise ValueError("Rolling resistance coefficient {} must be positive, not {}!".format('r'+ii, coef))


        for crossSection, coef in self.tunnelCoefficients.items():

            if crossSection is None or crossSection <= 0 or np.isinf(crossSection):

                raise ValueError("Tunnel cross section must be positive, not {}!".format(crossSection))

            if coef is None or coef <= 0 or np.isinf(coef):

                raise ValueError("Tunnel resistance coefficient must be positive, not {}!".format(coef))


    def exportModel(self):
        "Export train model (ODE and relevant train data)."

        totalMass = self.mass*self.rho

        # specific rolling resistance coefficients
        sr0 = self.r0/totalMass
        sr1 = self.r1/totalMass
        sr2 = self.r2/totalMass

        withPnBrake = self.forceMinPn != 0

        return TrainModel(sr0, sr1, sr2, self.rho, self.g, withPnBrake)


    def powerLossesFuns(self, split=True):
        """
        Return function of specific power losses for traction and regenerative brake

        (either explicitly defined in powerLosses attribute or implicitly via the two efficiencies).

        """

        # build power losses function from etas if necessary
        if not hasattr(self, 'powerLosses'):

            if hasattr(self, 'etaTraction') and hasattr(self, 'etaRgBrake'):

                # TODO: remove this from scripts
                powerLosses = lambda f,v: f*v*(f>0)*(1 - self.etaTraction)/self.etaTraction - (1-self.etaRgBrake)*f*v*(f<0)

            else:

                raise ValueError("Power losses function of train must by either explicitly or implicitly defined!")

        else:

            powerLosses = self.powerLosses

        totalMass = self.mass*self.rho

        specificPowerLosses = lambda f,v : (1/totalMass)*powerLosses(f*totalMass, v)  # input: specific force, output: specific power losses
        specificPowerLossesTr, specificPowerLossesRgb = splitLosses(specificPowerLosses)

        return (specificPowerLossesTr, specificPowerLossesRgb) if split else specificPowerLosses


class TrainModel():
    "Class with ODE (not all Train specs are needed here)."

    def __init__(self, sr0, sr1, sr2, rho=1, g=9.81, withPnBrake=True) -> None:

        # states

        time = ca.MX.sym('time')  # [s]
        velocitySquared = ca.MX.sym('velocitySquared')  # [m^2/s^2]
        position = ca.MX.sym('position')  # [m]

        x = ca.vertcat(time, velocitySquared, position)

        # controls

        traction = ca.MX.sym('traction')  # [N/kg]
        pnBrake = ca.MX.sym('pnBrake') # [N/kg]

        u = ca.vertcat(traction, pnBrake if withPnBrake else [])

        # parameters

        gradient = ca.MX.sym('gradient')  # [-]
        gradientLinearTerm = ca.MX.sym('gradientLinearTerm')  # [1/m]
        curvature = ca.MX.sym('curvature')  # [1/m]
        curvatureLinearTerm = ca.MX.sym('curvatureLinearTerm')  # [1/m^2]
        tunnelFactor = ca.MX.sym('tunnelFactor')  # [1/m]
        ds = ca.MX.sym('ds')

        p = ca.vertcat(gradient, gradientLinearTerm, curvature, curvatureLinearTerm, tunnelFactor, ds)

        # ODE

        rollingResistance = sr0 + sr1*ca.sqrt(velocitySquared) + sr2*velocitySquared  # [N/kg]
        gradientResistance = g*(1/rho)*(gradient+gradientLinearTerm*position)  # [N/kg]
        curvatureResistance = (1/rho)*(5.07 * (curvature+curvatureLinearTerm*position))  # [N/kg]
        tunnelResistance = tunnelFactor * velocitySquared  # [N/kg]

        acceleration = traction + (pnBrake if withPnBrake else 0) - rollingResistance - gradientResistance - curvatureResistance - tunnelResistance  # [m/s^2]

        timeODE = 1/ca.sqrt(velocitySquared)
        velocityODE = 2*acceleration
        positionODE = 1

        timeODE *= ds
        velocityODE *= ds
        positionODE *= ds

        fExplicit = ca.vertcat(timeODE, velocityODE, positionODE)

        # model

        self.ode = fExplicit
        self.acceleration = acceleration
        self.accelerationFun = ca.Function('a', [x, u, gradient, gradientLinearTerm, curvature, curvatureLinearTerm, tunnelFactor], [acceleration])
        self.rollingResistance = rollingResistance
        self.parameters = p
        self.controls = u
        self.states = x

        # fields needed in TrainIntegrator or post processing of results
        self.withPnBrake = withPnBrake
        self.sr0, self.sr1, self.sr2 = sr0, sr1, sr2
        self.rho = rho
        self.g = g


class TrainIntegrator():

    def __init__(self, model, solver, optsDict={}) -> None:

        self.model = model

        # check inputs

        if solver not in {'RK', 'IRK', 'CVODES'}:

            raise ValueError("Unknown integration method!")

        params = ca.vertcat(model.controls, model.parameters)

        if solver == 'RK':

            opts = OptionsRK(optsDict)

            if opts.numApproxSteps == 0:
                ode = model.ode
                states = model.states
            else:
                ode = ca.vertcat(model.ode[1], model.ode[2])
                states = ca.vertcat(model.states[1], model.states[2])

            self.eval = ca.simpleRK(ca.Function('ode', [states, params], [ode]), opts.numSteps, opts.order)

        elif solver == 'IRK':

            opts = OptionsIRK(optsDict)

            if opts.numApproxSteps == 0:
                ode = model.ode
                states = model.states
            else:
                ode = ca.vertcat(model.ode[1], model.ode[2])
                states = ca.vertcat(model.states[1], model.states[2])

            self.eval = ca.simpleIRK(ca.Function('ode', [states, params], [ode]), opts.numSteps, opts.order, opts.collMethod, 'fast_newton', {'max_iter': opts.maxIter, 'jit': opts.jit, 'error_on_fail': False})

        elif solver == 'CVODES':

            opts = OptionsCVODES(optsDict)
            opts.numApproxSteps = 0

            t0, tf = 0, 1
            cvodesFun = ca.integrator('integrator', 'cvodes', {'x': model.states, 'p': params, 'ode': model.ode}, t0, tf, {'abstol': opts.absTol, 'reltol': opts.relTol})

            self.eval = lambda x, p, dummy: cvodesFun(x0=x, p=p)['xf']

        if opts.numApproxSteps > 0:

            ns = opts.numApproxSteps

            evalPoints = [0] + [i/ns for i in range(1, ns+1)]

            z0 = ca.vertcat(model.states[1], model.states[2])
            p0 = ca.vertcat(model.controls, model.parameters)
            zf = self.eval(z0, p0, ca.hcat(evalPoints))  # zf[0, idx]: velSq, zf[1, idx]: pos

            tApprox = model.states[0]

            for idx in range(ns):

                vCurr = ca.sqrt(zf[0, idx])
                vNext = ca.sqrt(zf[0, idx+1])
                tApprox += 2*model.parameters[-1]*(evalPoints[idx+1]-evalPoints[idx])/(vCurr + vNext)

            eval = ca.vertcat(tApprox, zf[0, ns], zf[1, ns])

            self.eval = ca.Function('xNxt', [model.states, ca.vertcat(model.controls, model.parameters), ca.MX.sym('ds')], [eval])


    def solve(self, time, velocitySquared, ds, position=0, traction=0, pnBrake=0, gradient=0, gradientLinearTerm=0, curvature=0, curvatureLinearTerm=0, tunnelFactor=0):

        withPnBrake = self.model.withPnBrake

        if not withPnBrake and pnBrake != 0:

            raise ValueError("Cannot define value for pneumatic braking when this brake is deactivated!")

        x0 = ca.vertcat(time, velocitySquared, position)
        u0 = ca.vertcat(traction, pnBrake if withPnBrake else [])
        p0 = ca.vertcat(gradient, gradientLinearTerm, curvature, curvatureLinearTerm, tunnelFactor, ds)
        x1 = self.eval(x0, ca.vertcat(u0, p0), 1)

        out = {}
        out['time'] = x1[0]
        out['velSquared'] = x1[1]
        out['position'] = x1[2]

        return out


    def initLosses(self, lossesTrFun, lossesRgbFun, totalMass, solver='CVODES'):

        self.lossesTrFun = lossesTrFun
        self.lossesRgbFun = lossesRgbFun

        mdl = self.model
        vel = ca.MX.sym('v')
        pos = ca.MX.sym('pos')

        velDot = ca.substitute(mdl.acceleration, mdl.states[1], vel**2)
        velDot = ca.substitute(velDot, mdl.states[2], pos)

        energyTrDot = lossesTrFun(mdl.controls[0]*totalMass, vel)/totalMass  # tractive energy
        energyBrDot = lossesRgbFun(mdl.controls[0]*totalMass, vel)/totalMass  # braking energy

        dt = ca.MX.sym('dt')
        x = ca.vertcat(vel, pos, ca.MX.sym('eTr'), ca.MX.sym('eBr'))
        p = ca.vertcat(mdl.controls, mdl.parameters[0], mdl.parameters[1], mdl.parameters[2], mdl.parameters[3], mdl.parameters[4], dt)
        xdot = ca.vertcat(velDot, vel, energyTrDot, energyBrDot)

        if solver == 'RK':

            numSteps = 2

            fun = ca.Function('rhs', [x, p[:-1]], [xdot])

            self.lossesIntegrator = ca.simpleRK(fun, numSteps, 4)

        elif solver == 'CVODES':

            t0, tf = 0, 1
            cvodesFun = ca.integrator('integrator', 'cvodes', {'x':x, 'p':p, 'ode':dt*xdot}, t0, tf, {'abstol':1e-8, 'reltol':1e-6})

            self.lossesIntegrator = lambda x, p, dt: cvodesFun(x0=x, p=ca.vertcat(p, dt))['xf']

        else:

            raise ValueError("Unknown solver!")


    def calcLosses(self, velocity, dt, position=0, traction=0, pnBrake=0, gradient=0, gradientLinearTerm=0, curvature=0, curvatureLinearTerm=0, tunnelFactor=0):

        mdl = self.model

        out = self.lossesIntegrator(ca.vertcat(velocity, position, 0, 0), ca.vertcat(traction, pnBrake if mdl.withPnBrake else [], gradient, gradientLinearTerm, curvature, curvatureLinearTerm, tunnelFactor), dt)

        lossesTr, lossesRgb = out[2], out[3]

        return lossesTr, lossesRgb


    def initRollingResistance(self, solver='CVODES'):

        mdl = self.model

        bDot = mdl.ode[1]
        sDot = mdl.ode[2]
        eDot = mdl.rollingResistance*mdl.parameters[-1]

        x = ca.vertcat(mdl.states[1], mdl.states[2], ca.MX.sym('e'))
        p = ca.vertcat(mdl.controls, mdl.parameters)
        xdot = ca.vertcat(bDot, sDot, eDot)

        if solver == 'RK':

            fun = ca.Function('rhs', [x, p], [xdot])

            self.rollingResistanceIntegrator = ca.simpleRK(fun, 2, 4)

        elif solver == 'CVODES':

            t0, tf = 0, 1
            cvodesFun = ca.integrator('integrator', 'cvodes', {'x':x, 'p':p, 'ode':xdot}, t0, tf, {'abstol':1e-8, 'reltol':1e-6})

            self.rollingResistanceIntegrator = lambda x, p, dummy: cvodesFun(x0=x, p=p)['xf']

        else:

            raise ValueError("Unknown solver!")


    def calcRollingResistance(self, velocity, ds, position=0, traction=0, pnBrake=0, gradient=0, gradientLinearTerm=0, curvature=0, curvatureLinearTerm=0, tunnelFactor=0):

        mdl = self.model

        out = self.rollingResistanceIntegrator(ca.vertcat(velocity**2, position, 0), ca.vertcat(traction, pnBrake if mdl.withPnBrake else [],
                                                                                      gradient, gradientLinearTerm, curvature, curvatureLinearTerm, tunnelFactor, ds), 1)

        losses = out[2]

        return losses, ca.sqrt(out[0])


class OptionsRK(Options):

    def __init__(self, paramsDict):

        self.order = 4  # integration order

        self.numSteps = 1  # number of integration steps inside shooting interval

        self.numApproxSteps = 0  # option to use approximate equation for integration of time (active if > 0)

        super().__init__(paramsDict)


    def checkValues(self):

        if self.order != 4:

            raise ValueError("Only explicit Runge-Kutta of order 4 is currently implemented in casadi!")

        self.checkPositiveInteger(self.numSteps, 'Number of integration steps', allowZero=False)

        self.checkPositiveInteger(self.numApproxSteps, 'Number of time approximation steps', allowZero=True)


class OptionsIRK(Options):

    def __init__(self, paramsDict):

        self.order = 2  # integration order

        self.numSteps = 1  # number of integration steps inside shooting interval

        self.numApproxSteps = 0  # option to use approximate equation for integration of time (active if > 0)

        self.collMethod = 'radau'  # choice of collocation method ('radau' or 'legendre')

        self.maxIter = 10  # maximum number of Newton iterations

        self.jit = False  # just-in-time compilation for faster evaluation

        super().__init__(paramsDict)


    def checkValues(self):

        if int(self.order) != self.order or not 1 <= self.order <= 9:

            raise ValueError("Order of implicit Runge-Kutta should be a positive integer between 1 and 9!")

        self.checkPositiveInteger(self.numSteps, 'Number of integration steps', allowZero=False)

        self.checkPositiveInteger(self.numApproxSteps, 'Number of time approximation steps', allowZero=True)

        if self.collMethod not in {'radau', 'legendre'}:

            raise ValueError("Unknown collocation method: {}!".format(self.collMethod))

        self.checkPositiveInteger(self.maxIter, 'Maximum number of iterations', allowZero=False)

        if not isinstance(self.jit, bool):

            raise ValueError("JIT option must be a boolean!")


class OptionsCVODES(Options):

    def __init__(self, paramsDict):

        self.absTol = 1e-8
        self.relTol = 1e-6

        super().__init__(paramsDict)


    def checkValues(self):

        self.checkBounds(self.absTol, 'Absolute tolerance', 1e-20, 1e-1)
        self.checkBounds(self.relTol, 'Relative tolerance', 1e-20, 1e-1)


if __name__ == '__main__':

    # Example on how to simulate one step in space

    ds = 150  # interval length [m]
    t0 = 0  # initial time [s]
    v0 = 40/3.6  # initial velocity [m/s]
    gd = -15/1e3  # slope [-]
    cr = 1/300 # curvature [1/m]
    f0 = 0.4  # specific force [N/kg]

    trainSpecs = Train(config={'id':'NL_intercity_VIRM6'})
    integrator = TrainIntegrator(trainSpecs.exportModel(), 'RK', optsDict={'numApproxSteps':2})

    solution = integrator.solve(t0, v0**2, ds, traction=f0, gradient=gd, curvature=cr)

    print(solution)
