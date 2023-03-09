
import numpy as np
import pandas as pd
import casadi as ca

from data import dataLosses

def forceToLoad(force, velocity, forceMax, powerMax):
    "Convert force [N] to load [%] (force must be positive)."

    turningPoint = powerMax/forceMax

    return 100*(force/forceMax)*(velocity <= turningPoint) + 100*(force*velocity/powerMax)*(velocity > turningPoint)


def loadToForce(load, velocity, forceMax, powerMax):
    "Convert load [%] to force [N] (load must be positive)."

    turningPoint = powerMax/forceMax

    return (load/100)*(forceMax*(velocity <= turningPoint) + (powerMax/velocity)*(velocity > turningPoint))


def createSpline(loads, velocities, losses, forceMax, powerMax):

    loadsLoc = loads.copy()

    eps = 1e-4
    loadsLoc[-1] += eps  # to avoid artifacts when load is 100.000000001

    lut = ca.interpolant('name','bspline',[loadsLoc, velocities], losses)

    vMin = min(velocities)
    vMax = max(velocities)

    fix = lambda var: var.full()[0][0] if isinstance(var, ca.DM) else var

    def spline(f,v):

        # clipping to bounds to approximate low velocities with losses at minimum available velocity
        v = v*(vMin <= v)*(v <= vMax) + vMin*(v < vMin) + vMax*(v > vMax)

        absf = f*(f >= 0) - f*(f<0)
        load = forceToLoad(absf, v, forceMax, powerMax)
        load = fix(load)
        loss = lut(ca.vcat([load, v]))

        loss = fix(loss)

        return loss

    return lambda f,v: spline(f,v)


def motorLossesFunction(train, detailedOutput=False):

    minSpeed = 20  # [km/h]
    maxSpeed = 160  # [km/h]
    minFreq = 20  # [Hz]
    maxFreq = 170  # [Hz]
    powFreq = 55  # frequency where maximum power meets maximum force [Hz]

    HzToKmPerHour = lambda f: ((f - minFreq)/(maxFreq - minFreq))*(maxSpeed - minSpeed) + minSpeed

    forceMax = train.forceMax
    powerMax = forceMax*HzToKmPerHour(powFreq)/3.6

    # update train parameters to match data
    train.powerMax = powerMax
    train.powerMin = -powerMax
    train.forceMin = -forceMax*(train.forceMin != 0)
    train.velocityMax = maxSpeed/3.6

    numMotors = 4

    configA, configB = dataLosses()

    minLosses = np.minimum(np.array(configA['losses']), np.array(configB['losses']))*numMotors
    fun = createSpline(configB['loads'], [HzToKmPerHour(f)/3.6 for f in configB['frequencies']], minLosses.ravel(order='F'), forceMax, powerMax)

    if not detailedOutput:

        return fun

    else:

        def configToDataFrame(config):

            velocities = [HzToKmPerHour(f)/3.6 for f in config['frequencies']]

            df = pd.DataFrame(index=velocities)

            for i, l in enumerate(config['loads']):

                df[l] = [l*numMotors for l in config['losses'][i]]

            return df

        return {'fun':fun, 'dfA':configToDataFrame(configA), 'dfB':configToDataFrame(configB)}


def totalLossesFunction(train, auxiliaries=27000, etaGear=1):

    # function handle for motor losses
    motorLossesFun = motorLossesFunction(train)

    def totalLossesFun(f, v):

        # power at wheel
        pWheelTraction = f*v
        pWheelBraking = -f*v

        # gear losses
        gearLossesTraction = ((1-etaGear)/etaGear)*pWheelTraction
        gearLossesBraking = (1-etaGear)*pWheelBraking

        gearLosses = gearLossesTraction*(f >= 0) + gearLossesBraking*(f < 0)

        # motor and converter losses
        motorLosses = motorLossesFun(f, v)

        # transformer losses
        R = 10  # trafo resistance [Ohm]
        V = 15000  # voltage at catenary, assumed constant [V]

        PmTraction = pWheelTraction + gearLosses + motorLosses + auxiliaries
        PmBraking = pWheelBraking - gearLosses - motorLosses - auxiliaries  # by allowing this to be negative (in insufficient braking) we spare the 3rd case AND have the same results

        trafoLossesTraction = (V - ca.sqrt(V**2 - 4*R*PmTraction))**2/(4*R)

        trafoLossesBraking = (V - ca.sqrt(V**2 + 4*R*PmBraking))**2/(4*R)

        trafoLosses = trafoLossesTraction*(f >= 0) + trafoLossesBraking*(f < 0)

        # total losses
        totalLosses = gearLosses + motorLosses + auxiliaries + trafoLosses

        totalLosses *= motorLosses > 0  # outside of boundaries spline returns 0 and we want to keep it that way

        return totalLosses

    return totalLossesFun


if __name__ == '__main__':

    pass
