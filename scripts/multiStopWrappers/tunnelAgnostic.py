"""
Evaluate a trajectory that was planned without tunnels against the real track.

Some driver advisory systems plan the speed profile on a simplified model - no
drivetrain losses and no tunnel resistance - while the train physically has to hold
the advised profile through the tunnels anyway. Reproducing that means planning and
reporting on two different models:

  plan    optimal control problem on a track whose tunnels have been removed
  report  the same speed profile, but the force needed to hold it against the real
          track, and the energy that force costs with the real loss model

The reporting step cannot be done by postProcessDataFrame: tunnel resistance is part
of the dynamics, not of the energy accounting, and postProcessDataFrame takes the
forces of the trajectory as given. Feeding it the real cross sections changes only the
drivetrain loss term and leaves the tunnel work unaccounted for. The forces have to be
back-calculated from the speed profile against the real track instead, which is what
forceEstimator does for measured odometry runs.
"""

from mseetc.estimator import forceEstimator, energyEstimator


NO_TUNNELS = [(0.0, 0.0, "infinity")]  # what Track imports for a file without a 'tunnels' key


def removeTunnels(track):
    """
    Remove every tunnel from a track, in place.

    Leaves the track in the state it would have been loaded in had its file carried no
    'tunnels' key at all, so the cross section is 'no tunnel' over the whole length and
    computeTunnelFactor returns 0 everywhere. Must be called before updateLimits, while
    the track still covers its full length.
    """

    track.importTunnelTuples(NO_TUNNELS)

    return track


def estimateEnergyOnTrack(dfPlan, positionStart, train, track, opts):
    """
    Back-calculate the force needed to follow a planned speed profile on a given track,
    and the energy it costs, using the loss function currently set on the train.

    dfPlan        trajectory from casadiSolver.solve(), positions relative to the section
    positionStart position of the section start on the track, to make those absolute
    train         train whose powerLosses is the loss model to report against
    track         freshly loaded track, tunnels intact, reversed for a return trip.
                  forceEstimator crops it itself, so it must not be cropped already
    opts          dict with numIntervals, integrationMethod and integrationOptions

    Returns (dfEstimate, energyStats), the latter being the dict of energyEstimator:
    used, regenerated and net energy plus the drivetrain and mechanical loss split.
    """

    target = dfPlan[["Position [m]", "Velocity [m/s]"]].copy()

    target["Position [m]"] = target["Position [m]"] + positionStart  # onto the track

    estimatorOpts = {
        "numIntervals": opts["numIntervals"],
        "integrationMethod": opts["integrationMethod"],
        "integrationOptions": opts["integrationOptions"],
        # the profile is prescribed, so braking curves would only add discretization nodes
        "withEtcsBrakingCurves": False,
    }

    estimator = forceEstimator(target, train, track, optsDict=estimatorOpts,
                               trainLengthDependentValues=True)

    dfEstimate = estimator.estimate()

    energyStats = energyEstimator(dfEstimate, train,
                                  optsDict={"numIntervals": opts["numIntervals"]}).estimate()

    return dfEstimate, energyStats


def reportForceLimitViolations(dfEstimate, train):
    """
    Count the points where holding the planned profile would need more than the train has.

    A trajectory planned without tunnel resistance can ask for more force or power than
    the train can deliver once the tunnels are back. That does not make the estimate
    wrong, it makes the advised profile unholdable, which is worth knowing about.
    """

    force = dfEstimate["Force (el) [N]"].to_numpy(dtype=float)
    velocity = dfEstimate["Velocity [m/s]"].to_numpy(dtype=float)

    overForce = 0 if train.forceMax is None else int((force > train.forceMax).sum())
    overPower = 0 if train.powerMax is None else int((force*velocity > train.powerMax).sum())

    return overForce, overPower
