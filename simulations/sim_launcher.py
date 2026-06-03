from mseetc.efficiency import totalLossesFunction
from mseetc.ocp import casadiSolver


def get_power_loss_function(train, mode="perfect",* ,auxiliaries: float = 27_000, eta_gear: float = 0.96):

    if mode == "perfect":

        return lambda f, v: 0

    elif mode == "static":

        return lambda f, v: (f>0)*f*v*(1-train.etaTraction)/train.etaTraction - (f<0)*f*v*(1-train.etaRgBrake)

    elif mode == "dynamic":

        return totalLossesFunction(train, auxiliaries=auxiliaries, etaGear=eta_gear)

    else:

        raise ValueError("mode must be one of: 'perfect', 'static', 'dynamic'")



if __name__ == '__main__':

    from mseetc.train import Train
    from mseetc.track import Track

    # Timetable
    startPosition = 0       # [m]
    endPosition = 20000     # [m]
    duration = 60*20        # [s]

    train = Train(config={'id':'CH_Stadler_FLIRT_TPF'}, pathJSON='../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    track = Track(config={'id':'CH_ZH_LU'}, pathJSON='../tracks')
    track = Track(config={'id':'CH_StGallen_Wil'}, pathJSON='../tracks')
    track.updateTrainLengthDependentValues(train)
    track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')

    opts = {'numIntervals':600, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, opts)

    df, stats = solver.solve(duration)

    # print some info
    if df is not None:

        print("")
        print("Objective value = {:.2f} {}".format(stats['Cost'], 'kWh' if solver.opts.energyOptimal else 's'))
        print("")
        print("Maximum acceleration: {:5.2f}, with bound {}".format(df.max()['Acceleration [m/s^2]'], train.accMax if train.accMax is not None else 'None'))
        print("Maximum deceleration: {:5.2f}, with bound {}".format(df.min()['Acceleration [m/s^2]'], train.accMin if train.accMin is not None else 'None'))

    else:

        print("Solver failed!")