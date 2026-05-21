from ocp import casadiSolver

if __name__ == '__main__':

    from train import Train
    from track import Track

    # Timetable
    startPosition = 0       # [m]
    endPosition = 20000     # [m]
    duration = 60*20        # [s]

    train = Train(config={'id':'SBB_Flirt_2'}, pathJSON='../trains')

    track = Track(config={'id':'CH_ZH_LU'}, pathJSON='../tracks')
    # track = Track(config={'id':'CH_StGallen_Wil'}, pathJSON='../tracks')
    track.updateLimits(positionStart=startPosition, positionEnd=endPosition, unit='m')
    track.updateTrainLengthDependentValues(train)

    opts = {'numIntervals':400, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':0}, 'energyOptimal':True}

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