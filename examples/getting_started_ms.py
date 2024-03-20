from mseetc.train import Train
from mseetc.track import Track
from mseetc.ms.ocp import casadiSolver

if __name__ == '__main__':


    # Example on how to solve an OCP

    train = Train(config={'id':'NL_Intercity_VIRM6', 'max deceleration':None, 'max acceleration':{'unit':'m/s^2', 'value':0.45}}, pathJSON='../trains')

    track = Track(config={'id':'00_var_speed_limit_100'}, pathJSON='../tracks')

    opts = {'numIntervals':200, 'integrationMethod':'RK', 'integrationOptions':{'numApproxSteps':1}, 'energyOptimal':True}

    solver = casadiSolver(train, track, opts)

    df, stats = solver.solve(1541)

    # print some info
    if df is not None:

        print("")
        print("Objective value = {:.2f} {}".format(stats['Cost'], 'kWh' if solver.opts.energyOptimal else 's'))
        print("")
        print("Maximum acceleration: {:5.2f}, with bound {}".format(df.max()['Acceleration [m/s^2]'], train.accMax if train.accMax is not None else 'None'))
        print("Maximum deceleration: {:5.2f}, with bound {}".format(df.min()['Acceleration [m/s^2]'], train.accMin if train.accMin is not None else 'None'))

    else:

        print("Solver failed!")
