from mseetc.etcs import BrakingTarget, EtcsBrakingCurveCalculator

from mseetc.track import Track
from mseetc.train import Train


if __name__ == '__main__':

    """
    Simple tool to plot etcs braking curves for a single speed decrease.
    """


    ####################################################################################################################
    ### Input
    directoryTrain = '../../nightTests'
    trainId = 'trainNight'

    directoryTrack = '../../nightTests'
    trackId = 'trackNight'

    brakingTargetPosition = 5000  # [m]
    brakingTargetOverlap = 100  # [m]
    brakingTargetInitialVelocity = 140/3.6  # [m/s]
    brakingTargetFinalVelocity = 40 / 3.6  # [m/s]

    addConstantVelocitySections = False
    ####################################################################################################################


    train = Train(config={'id': trainId}, pathJSON=directoryTrain)

    track = Track(config={'id': trackId}, pathJSON=directoryTrack)
    track.updateTrainLengthDependentValues(train)

    target = BrakingTarget(
            position=brakingTargetPosition,
            overlap=brakingTargetOverlap,
            permittedVelocity=brakingTargetInitialVelocity,
            targetVelocity=brakingTargetFinalVelocity
    )

    calculator = EtcsBrakingCurveCalculator(train, track, distancePre=5000, distancePost=1000)
    curve_set, interventionPoints = calculator.computeTarget(target)

    calculator.printInterventionPoints(interventionPoints)

    if addConstantVelocitySections:

        curve_set = calculator.processCurvesBeforeTarget(curve_set, target)

        if target.targetVelocity > 0:
            curve_set = calculator.processCurvesAfterTarget(curve_set, target)

    calculator.plotCurves(curve_set, target)