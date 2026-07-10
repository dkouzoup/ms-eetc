from mseetc.etcs import BrakingTarget, EtcsBrakingCurveCalculator

if __name__ == '__main__':

    from mseetc.track import Track
    from mseetc.train import Train

    train = Train(config={'id': 'trainNight'}, pathJSON='../nightTests')

    track = Track(config={'id': 'trackNight'}, pathJSON='../nightTests')
    track.updateTrainLengthDependentValues(train)

    target = BrakingTarget(
            position=5000,
            overlap= 100,
            permittedVelocity=140/3.6,
            targetVelocity=40/3.6
    )

    addConstantVelocitySections = False

    calculator = EtcsBrakingCurveCalculator(train, track, distancePre=5000, distancePost=1000)
    curve_set, interventionPoints = calculator.computeTarget(target)

    calculator.printInterventionPoints(interventionPoints)

    if addConstantVelocitySections:

        curve_set = calculator.processCurvesBeforeTarget(curve_set, target)

        if target.targetVelocity > 0:
            curve_set = calculator.processCurvesAfterTarget(curve_set, target)

    calculator.plotCurves(curve_set, target)