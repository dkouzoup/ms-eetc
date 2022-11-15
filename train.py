import numpy as np
import casadi as ca

from utils import Options

from data import dataIntercityTrain

class Train():
    "Class with train specifications."

    def __init__(self, train='Intercity'):

        if train == 'Intercity':

            data = dataIntercityTrain()

        else:

            raise ValueError("Unknown train specified!")

        # train mass parameters

        self.mass = data['mass']  # train mass [kg]

        self.g = data['g']  # acceleration of gravity [m/s^2]

        self.rho = data['rho']  # rotating-mass factor [-]

        # velocity limits

        self.velocityMax = data['velocityMax']  # maximum train speed [m/s]

        # force limits

        self.forceMax = data['forceMax']  # maximum traction force [N]

        self.forceMin = data['forceMin']  # maximum regenerative braking force [N]

        self.forceMinPn = data['forceMinPn']  # maximum pneumatic braking force [N]

        # power limits and losses

        self.powerMax = data['powerMax']  # maximum traction power [W]

        self.powerMin = data['powerMin']  # maximum regenerative braking power [W]

        self.powerLosses = lambda f,v: 0  # losses [W] as function of: force [N] and velocity [m/s]

        # acceleration/deceleration limits

        self.accMax = data['accMax']  # maximum acceleration [m/s^2]

        self.accMin = data['accMin']  # maximum allowed deceleration [m/s^2]

        # rolling resistance polynomial: f = r0 + r1*v + r2*v^2 [m/s -> N]

        self.r0 = data['r0']  # constant term [-]

        self.r1 = data['r1']  # linear term [-]

        self.r2 = data['r2']  # quadratic term [-]

        self.checkFields()


    def checkFields(self):

        if self.mass is None or self.mass < 0 or np.isinf(self.mass):

            raise ValueError("Train mass must be a positive number, not {}!".format(self.mass))

        if self.g is None or not 9 <= self.g <= 10:

            raise ValueError("Acceleration of gravity must be between 9 and 10 m/s^2, not {}!".format(self.g))

        if self.rho is None or not 1 <= self.rho <= 1.5:

            raise ValueError("Rotation mass factor must be between 1 and 1.5, not {}!".format(self.rho))

        if self.velocityMax <= 0 or np.isinf(self.velocityMax):

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

        if not callable(self.powerLosses):

            raise ValueError("Power losses function must be a callable function handle!")


    def exportModel(self):
        "Export train model (ODE and relevant train data)."

        totalMass = self.mass*self.rho

        # specific rolling resistance coefficients
        sr0 = self.r0/totalMass
        sr1 = self.r1/totalMass
        sr2 = self.r2/totalMass

        withPnBrake = self.forceMinPn != 0

        return TrainModel(sr0, sr1, sr2, self.rho, self.g, withPnBrake)


class TrainModel():
    "Class with ODE (not all Train specs are needed here)."

    def __init__(self, sr0, sr1, sr2, rho=1, g=9.81, withPnBrake=True) -> None:

        # states

        time = ca.MX.sym('time')
        velocitySquared = ca.MX.sym('velocitySquared')

        x = ca.vertcat(time, velocitySquared)

        # controls

        traction = ca.MX.sym('traction')
        pnBrake = ca.MX.sym('pnBrake')

        u = ca.vertcat(traction, pnBrake if withPnBrake else [])

        # parameters

        gradient = ca.MX.sym('gradient')
        ds = ca.MX.sym('ds')

        p = ca.vertcat(gradient, ds)

        # ODE

        rollingResistance = sr0 + sr1*ca.sqrt(velocitySquared) + sr2*velocitySquared
        acceleration = traction + (pnBrake if withPnBrake else 0) - rollingResistance - g*gradient*(1/rho)
        timeODE = 1/ca.sqrt(velocitySquared)
        velocityODE = 2*acceleration

        timeODE *= ds
        velocityODE *= ds

        fExplicit = ca.vertcat(timeODE, velocityODE)

        # model

        self.ode = fExplicit
        self.acceleration = acceleration
        self.accelerationFun = ca.Function('a', [x, u, gradient], [acceleration])
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

            ode = model.ode if opts.numApproxSteps == 0 else model.ode[1]
            states = model.states if opts.numApproxSteps == 0 else model.states[1]

            self.eval = ca.simpleRK(ca.Function('ode', [states, params], [ode]), opts.numSteps, opts.order)

        elif solver == 'IRK':

            opts = OptionsIRK(optsDict)

            ode = model.ode if opts.numApproxSteps == 0 else model.ode[1]
            states = model.states if opts.numApproxSteps == 0 else model.states[1]

            self.eval = ca.simpleIRK(ca.Function('ode', [states, params], [ode]), opts.numSteps, opts.order, opts.collMethod, 'fast_newton', {'max_iter':opts.maxIter, 'jit':opts.jit, 'error_on_fail':False})

        elif solver == 'CVODES':

            opts = OptionsCVODES(optsDict)
            opts.numApproxSteps = 0

            states = model.states

            cvodesFun = ca.integrator('integrator', 'cvodes', {'x':model.states, 'p':params, 'ode':model.ode}, {'tf':1, 'abstol':opts.absTol, 'reltol':opts.relTol})

            self.eval = lambda x, p, dummy: cvodesFun(x0=x, p=p)['xf']

        if opts.numApproxSteps > 0:

            ns = opts.numApproxSteps

            evalPoints = [0] + [i/ns for i in range(1, ns+1)]

            b0 = model.states[1]
            p0 = ca.vertcat(model.controls, model.parameters)
            bf = self.eval(b0, p0, ca.hcat(evalPoints))

            tApprox = model.states[0]

            for idx in range(ns):

                vCurr = ca.sqrt(bf[idx])
                vNext = ca.sqrt(bf[idx+1])
                tApprox += 2*model.parameters[1]*(evalPoints[idx+1]-evalPoints[idx])/(vCurr + vNext)

            eval = ca.vertcat(tApprox, bf[-1])

            self.eval = ca.Function('xNxt', [model.states, ca.vertcat(model.controls, model.parameters), ca.MX.sym('ds')], [eval])


    def solve(self, time, velocitySquared, ds, traction=0, pnBrake=0, gradient=0):

        withPnBrake = self.model.withPnBrake

        if not withPnBrake and pnBrake != 0:

            raise ValueError("Cannot define value for pneumatic braking when this brake is deactivated!")

        x0 = ca.vertcat(time, velocitySquared)
        u0 = ca.vertcat(traction, pnBrake if withPnBrake else [])
        p0 = ca.vertcat(gradient, ds)
        x1 = self.eval(x0, ca.vertcat(u0, p0), 1)

        out = {}
        out['time'] = x1[0]
        out['velSquared'] = x1[1]

        return out


    def initLosses(self, lossesTrFun, lossesRgbFun, totalMass, solver='CVODES'):

        self.lossesTrFun = lossesTrFun
        self.lossesRgbFun = lossesRgbFun

        mdl = self.model
        vel = ca.MX.sym('v')

        velDot = ca.substitute(mdl.acceleration, mdl.states[1], vel**2)

        energyTrDot = lossesTrFun(mdl.controls[0]*totalMass, vel)/totalMass  # tractive energy
        energyBrDot = lossesRgbFun(mdl.controls[0]*totalMass, vel)/totalMass  # braking energy

        dt = ca.MX.sym('dt')
        x = ca.vertcat(vel, ca.MX.sym('eTr'), ca.MX.sym('eBr'))
        p = ca.vertcat(mdl.controls, mdl.parameters[0], dt)
        xdot = ca.vertcat(velDot, energyTrDot, energyBrDot)

        if solver == 'RK':

            numSteps = 2

            fun = ca.Function('rhs', [x, p[:-1]], [xdot])

            self.lossesIntegrator = ca.simpleRK(fun, numSteps, 4)

        elif solver == 'CVODES':

            cvodesFun = ca.integrator('integrator', 'cvodes', {'x':x, 'p':p, 'ode':dt*xdot}, {'tf':1, 'abstol':1e-8, 'reltol':1e-6})

            self.lossesIntegrator = lambda x, p, dt: cvodesFun(x0=x, p=ca.vertcat(p, dt))['xf']

        else:

            raise ValueError("Unknown solver!")


    def calcLosses(self, velocity, dt, traction=0, pnBrake=0, gradient=0):

        mdl = self.model

        out = self.lossesIntegrator(ca.vertcat(velocity, 0, 0), ca.vertcat(traction, pnBrake if mdl.withPnBrake else [], gradient), dt)

        lossesTr, lossesRgb = out[1], out[2]

        return lossesTr, lossesRgb


    def initRollingResistance(self, solver='CVODES'):

        mdl = self.model

        bDot = mdl.ode[1]
        eDot = mdl.rollingResistance*mdl.parameters[1]

        ds = ca.MX.sym('ds')
        x = ca.vertcat(mdl.states[1], ca.MX.sym('e'))
        p = ca.vertcat(mdl.controls, mdl.parameters, ds)
        xdot = ca.vertcat(bDot, eDot)

        if solver == 'RK':

            fun = ca.Function('rhs', [x, p[:-1]], [xdot])

            self.rollingResistanceIntegrator = ca.simpleRK(fun, 2, 4)

        elif solver == 'CVODES':

            cvodesFun = ca.integrator('integrator', 'cvodes', {'x':x, 'p':p, 'ode':ds*xdot}, {'tf':1, 'abstol':1e-8, 'reltol':1e-6})

            self.rollingResistanceIntegrator = lambda x, p, ds: cvodesFun(x0=x, p=ca.vertcat(p, ds))['xf']

        else:

            raise ValueError("Unknown solver!")


    def calcRollingResistance(self, velocity, ds, traction=0, pnBrake=0, gradient=0):

        mdl = self.model

        out = self.rollingResistanceIntegrator(ca.vertcat(velocity**2, 0), ca.vertcat(traction, pnBrake if mdl.withPnBrake else [], gradient, ds), 1)

        losses = out[1]

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
    f0 = 0.4  # specific force [N/kg]

    trainSpecs = Train(train='Intercity')
    integrator = TrainIntegrator(trainSpecs.exportModel(), 'RK', optsDict={'numApproxSteps':2})

    solution = integrator.solve(t0, v0**2, ds, f0, gradient=gd)

    print(solution)
