import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.ocp import OptionsCasadiSolver
from mseetc.track import Track, computeDiscretizationPoints
from mseetc.train import Train, TrainIntegrator
from mseetc.utils import computeTunnelFactor, get_power_loss_function


class forceEstimator():

    def __init__(self, df, train, track, optsDict={}, trainLengthDependentValues=False, plotInterpolation=False):

        # input checking
        track.checkFields()
        train.checkFields()
        self.validateTargetDf(df, track.length)

        self.train = train
        self.track = track

        # targetTrajectory
        self.targetTimes = df.index.to_numpy(dtype=float)
        self.targetPositions = df["Position [m]"].to_numpy(dtype=float)
        self.targetVelocities = df["Velocity [m/s]"].to_numpy(dtype=float)

        self.positionStart = self.targetPositions[0]
        self.positionEnd = self.targetPositions[-1]

        self.opts = OptionsCasadiSolver(optsDict)

        # track
        if trainLengthDependentValues:

            track.updateTrainLengthDependentValues(train)

        track.updateLimits(positionStart=self.positionStart, positionEnd=self.positionEnd, unit='m')

        self.numIntervals = self.opts.numIntervals
        self.points = computeDiscretizationPoints(track, self.numIntervals, self.opts, np.array([], dtype=float))

        self.positionsInterp = self.points.index.to_numpy(dtype=float)
        self.steps = np.diff(self.positionsInterp)

        targetPositionsRelative = self.targetPositions - self.positionStart

        self.targetTimesInterpolated = np.interp(self.positionsInterp, targetPositionsRelative, self.targetTimes)
        self.targetVelocitiesInterpolated = np.interp(self.positionsInterp, targetPositionsRelative, self.targetVelocities)

        if plotInterpolation:

            self.plotInterpolationComparison()

        # train
        trainModel = train.exportModel()
        self.totalMass = train.mass * train.rho
        self.trainIntegrator = TrainIntegrator(trainModel, self.opts.integrationMethod, self.opts.integrationOptions.toDict())


    def plotInterpolationComparison(self):

        fig, ax = plt.subplots(figsize=(24, 12))
        ax.plot(self.targetPositions / 1000, self.targetVelocities * 3.6, label="original velocities")
        ax.plot(self.positionsInterp / 1000, self.targetVelocitiesInterpolated * 3.6, linestyle="--", label="interpolated velocities")
        ax.set_title("Interpolation Comparison")
        ax.set_xlabel("Position [km]")
        ax.set_ylabel("Velocity [km/h]")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_xlim(self.targetPositions.min() / 1000, self.targetPositions.max() / 1000)
        ax.figure.tight_layout()

        plt.show()


    def validateTargetDf(self, df, trackLength):

        if not isinstance(df, pd.DataFrame):

            raise ValueError("Trajectory must be provided as a pandas DataFrame!")

        requiredColumns = ["Position [m]", "Velocity [m/s]"]

        for column in requiredColumns:

            if column not in df.columns:

                raise ValueError("Trajectory DataFrame must contain column '{}'!".format(column))

        if len(df.index) == 0:

            raise ValueError("Trajectory DataFrame must not be empty!")

        try:

            times = df.index.to_numpy(dtype=float)
            positions = df["Position [m]"].to_numpy(dtype=float)
            velocities = df["Velocity [m/s]"].to_numpy(dtype=float)

        except ValueError:

            raise ValueError("Trajectory time index, positions and velocities must be numeric!")

        if positions[-1] > trackLength:

            raise ValueError(
                "Last trajectory position must not be larger than track length! "
                "Got {:.3f} m, track length is {:.3f} m.".format(positions[-1], trackLength)
            )

        if len(positions) < 2:

            raise ValueError("Trajectory DataFrame must contain at least two points!")

        if np.any(np.diff(times) <= 0):

            raise ValueError("Trajectory time index must be strictly increasing!")

        if np.any(np.diff(positions) <= 0):

            raise ValueError("Trajectory positions must be strictly increasing!")

        if np.any(velocities < 0):

            raise ValueError("Trajectory velocities must not be negative!")

        if positions[0] < 0:

            raise ValueError("First trajectory position must not be negative!")


    def estimate(self):

        time = self.targetTimes[0]
        velSq = self.targetVelocities[0] * self.targetVelocities[0]

        estimatedForcesEl = []
        estimatedForcesPnb = []
        integratedTimes = [time]
        integratedVelocities = [np.sqrt(velSq)]

        forceElLower = (self.train.forceMin + self.train.forceMinPn) / self.totalMass
        forceElUpper = self.train.forceMax / self.totalMass

        for i in range(self.numIntervals):

            forceEstimate, timeEstimate, velocityEstimate = self.bisection(forceElLower, forceElUpper, time, velSq, i)

            if forceEstimate < self.train.forceMin / self.totalMass:

                forceElEstimate = self.train.forceMin / self.totalMass
                forcePnEstimate = forceEstimate - forceElEstimate

            else:

                forceElEstimate = forceEstimate
                forcePnEstimate = 0

            estimatedForcesEl.append(forceElEstimate)
            estimatedForcesPnb.append(forcePnEstimate)
            integratedTimes.append(timeEstimate)
            integratedVelocities.append(velocityEstimate)

            time = timeEstimate
            velSq = velocityEstimate * velocityEstimate

        estimatedForcesEl.append(0)
        estimatedForcesPnb.append(0)

        dfEstimate = pd.DataFrame(index=np.array(integratedTimes, dtype=float))
        dfEstimate.index.name = "Time [s]"

        dfEstimate["Position [m]"] = self.positionsInterp + self.positionStart
        dfEstimate["Velocity [m/s]"] = np.array(integratedVelocities, dtype=float)
        dfEstimate["Force (el) [N]"] = np.array(estimatedForcesEl, dtype=float) * self.totalMass
        dfEstimate["Force (pnb) [N]"] = np.array(estimatedForcesPnb, dtype=float) * self.totalMass

        return dfEstimate


    def bisection(self, lower, upper, time, velSq, i, tolerance=1e-8, maxIterations=60):

        targetVelocity = self.targetVelocitiesInterpolated[i + 1]

        tLower, velocityLower = self.integrate(lower, 0, time, velSq, i)
        tUpper, velocityUpper = self.integrate(upper, 0, time, velSq, i)

        errorLower = velocityLower - targetVelocity
        errorUpper = velocityUpper - targetVelocity

        if abs(errorLower) <= tolerance:

            return lower, tLower, velocityLower

        if abs(errorUpper) <= tolerance:

            return upper, tUpper, velocityUpper

        if errorLower * errorUpper > 0:

            raise ValueError(
                "Bisection requires a sign change in interval {}! "
                "Target velocity is {:.6f} m/s, "
                "Fel={} gives {:.6f} m/s and Fel={} gives {:.6f} m/s.".format(i, targetVelocity, lower, velocityLower, upper, velocityUpper)
            )

        for _ in range(maxIterations):

            middle = 0.5 * (lower + upper)

            tMiddle, velocityMiddle = self.integrate(middle, 0, time, velSq, i)

            errorMiddle = velocityMiddle - targetVelocity

            if abs(errorMiddle) <= tolerance or abs(upper - lower) <= tolerance:

                return middle, tMiddle, velocityMiddle

            if errorLower * errorMiddle <= 0:

                upper = middle
                errorUpper = errorMiddle

            else:

                lower = middle
                errorLower = errorMiddle

        return middle, tMiddle, velocityMiddle


    def integrate(self, Fel, Fpb, time, velSq, i):

        grad = self.points.iloc[i]['Gradient [permil]'] / 1e3
        gradLinearTerm = self.points.iloc[i]["Gradient linear term [permil/m]"] / 1e3
        curv = self.points.iloc[i]['Curvature [1/m]']
        curvLinearTerm = self.points.iloc[i]["Curvature linear term [1/m^2]"]
        crossSection = self.points.iloc[i]['CrossSection [m^2]']
        tunnelFactor = computeTunnelFactor(crossSection, self.train, self.opts)

        out = self.trainIntegrator.solve(time=time,
                                    velocitySquared=velSq,
                                    ds=self.steps[i],
                                    traction=Fel,
                                    pnBrake=Fpb,
                                    gradient=grad,
                                    gradientLinearTerm=gradLinearTerm,
                                    curvature=curv,
                                    curvatureLinearTerm=curvLinearTerm,
                                    tunnelFactor=tunnelFactor)

        integratedTime = float(np.asarray(out["time"], dtype=float).squeeze())
        integratedVelocitySquared = float(np.asarray(out["velSquared"], dtype=float).squeeze())

        if np.isnan(integratedTime):

            # too large negative force, train is moving backwards
            integratedTime = time + self.steps[i] / np.sqrt(velSq)

        if np.isnan(integratedVelocitySquared):

            integratedVelocitySquared = 0

        integratedVelocity = np.sqrt(max(integratedVelocitySquared, 0))

        return integratedTime, integratedVelocity


def plotVelocityComparison(dfTarget, dfEstimate):

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.plot(dfTarget["Position [m]"] / 1000, dfTarget["Velocity [m/s]"] * 3.6, label="original trajectory")
    ax.plot(dfEstimate["Position [m]"] / 1000, dfEstimate["Velocity [m/s]"] * 3.6, linestyle="--", label="estimated trajectory")
    ax.set_title("Estimation Comparison - Velocity")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, dfTarget["Position [m]"].max() / 1000)
    ax.figure.tight_layout()

    plt.show()


def plotForceComparison(dfTarget, dfEstimate):

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.step(dfTarget["Position [m]"] / 1000, dfTarget["Force (el) [N]"] / 1000, where="post", label="original trajectory")
    ax.step(dfEstimate["Position [m]"] / 1000, dfEstimate["Force (el) [N]"] / 1000, linestyle="--", where="post", label="estimated trajectory")
    ax.set_title("Estimation Comparison - Force")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Force [kN]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, dfTarget["Position [m]"].max() / 1000)
    ax.figure.tight_layout()

    plt.show()


def plotTimeCoparison(dfTarget, dfEstimate):

    fig, ax = plt.subplots(figsize=(24, 12))
    ax.plot(dfTarget["Position [m]"] / 1000, dfTarget.index.to_numpy(), label="original trajectory")
    ax.plot(dfEstimate["Position [m]"] / 1000, dfEstimate.index.to_numpy(), linestyle="--", label="estimated trajectory")
    ax.set_title("Estimation Comparison - Time")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Time [s]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_xlim(0, dfTarget["Position [m]"].max() / 1000)
    ax.figure.tight_layout()

    plt.show()



class energyEstimator():

    def __init__(self, df, train, track=None, optsDict={}):

        # input checking
        train.checkFields()
        self.checkDf(df)

        self.opts = OptionsCasadiSolver(optsDict)

        # train
        self.train = train
        trainModel = train.exportModel()
        self.totalMass = train.mass * train.rho
        self.trainIntegrator = TrainIntegrator(trainModel, self.opts.integrationMethod, self.opts.integrationOptions.toDict())

        self.powerLossesTr, self.powerLossesRgb = train.powerLossesFuns()

        # targetTrajectory
        self.times = df.index.to_numpy(dtype=float)
        self.positions = df["Position [m]"].to_numpy(dtype=float)
        self.velocities = df["Velocity [m/s]"].to_numpy(dtype=float)
        self.forcesEl = df["Force (el) [N]"].to_numpy(dtype=float) / self.totalMass
        self.forcesPnb = df["Force (pnb) [N]"].to_numpy(dtype=float) / self.totalMass

        self.ds = np.diff(self.positions)


        if self.opts.integrateLosses:

            if track is None:

                raise ValueError("Track must be provided for loss integration.")

            self.points = computeDiscretizationPoints(track, self.opts.numIntervals, self.opts, np.array([], dtype=float))

            if not np.array_equal(self.points.index.to_numpy(), self.positions):

                raise ValueError("Force profile has been computed using a different track discretization")

            self.trainIntegrator.initLosses(self.powerLossesTr, self.powerLossesRgb, self.totalMass)


    def checkDf(self, df):

        if not isinstance(df, pd.DataFrame):

            raise ValueError("Trajectory must be provided as a pandas DataFrame!")

        requiredColumns = ["Position [m]", "Velocity [m/s]", "Force (el) [N]", "Force (pnb) [N]"]

        for column in requiredColumns:

            if column not in df.columns:

                raise ValueError("Trajectory DataFrame must contain column '{}'!".format(column))

        if len(df.index) == 0:

            raise ValueError("Trajectory DataFrame must not be empty!")

        try:

            times = df.index.to_numpy(dtype=float)
            positions = df["Position [m]"].to_numpy(dtype=float)
            velocities = df["Velocity [m/s]"].to_numpy(dtype=float)
            df["Force (el) [N]"].to_numpy(dtype=float)
            df["Force (pnb) [N]"].to_numpy(dtype=float)

        except ValueError:

            raise ValueError("Trajectory time index, positions and velocities must be numeric!")

        if len(positions) < 2:

            raise ValueError("Trajectory DataFrame must contain at least two points!")

        if np.any(np.diff(times) <= 0):

            raise ValueError("Trajectory time index must be strictly increasing!")

        if np.any(np.diff(positions) <= 0):

            raise ValueError("Trajectory positions must be strictly increasing!")

        if np.any(velocities < 0):

            raise ValueError("Trajectory velocities must not be negative!")

        if positions[0] < 0:

            raise ValueError("First trajectory position must not be negative!")


    def estimate(self):

        opts = self.opts

        driveTrainLosses = 0.0
        usedEnergy = 0.0
        regeneratedEnergy = 0.0

        for i in range(opts.numIntervals):

            if opts.energyOptimal:

                if not opts.integrateLosses:

                    # approximating interval with mid-point rule
                    vMid = (self.velocities[i] + self.velocities[i+1]) / 2

                    if self.forcesEl[i] > 0:

                        loss = self.powerLossesTr(self.forcesEl[i], vMid) / vMid
                        loss = np.asarray(loss, dtype=float).squeeze()
                        driveTrainLosses += loss * self.ds[i]
                        usedEnergy += self.ds[i] * (self.forcesEl[i] + loss)

                    else:

                        loss = self.powerLossesRgb(self.forcesEl[i], vMid) / vMid
                        loss = np.asarray(loss, dtype=float).squeeze()
                        driveTrainLosses += loss * self.ds[i]
                        regeneratedEnergy += self.ds[i] * (-self.forcesEl[i] - loss)

                else:

                    grad = self.points.iloc[i]['Gradient [permil]'] / 1e3
                    gradLinearTerm = self.points.iloc[i]["Gradient linear term [permil/m]"] / 1e3
                    curv = self.points.iloc[i]['Curvature [1/m]']
                    curvLinearTerm = self.points.iloc[i]["Curvature linear term [1/m^2]"]
                    crossSection = self.points.iloc[i]['CrossSection [m^2]']
                    tunnelFactor = computeTunnelFactor(crossSection, self.train, self.opts)

                    energyLossesTr, energyLossesRgb = self.trainIntegrator.calcLosses(self.velocities[i], self.times[i+1] - self.times[i], 0, self.forcesEl[i], self.forcesPnb[i], grad, gradLinearTerm, curv, curvLinearTerm, tunnelFactor)

                    if self.forcesEl[i] > 0:

                        driveTrainLosses += energyLossesTr
                        usedEnergy = self.ds[i] * (self.forcesEl[i] + energyLossesTr)

                    else:

                        driveTrainLosses += energyLossesRgb
                        regeneratedEnergy = self.ds[i] * -self.forcesEl[i] - energyLossesRgb


        driveTrainLosses = driveTrainLosses * self.totalMass  / 3.6e6  # [kWh]
        usedEnergy = usedEnergy * self.totalMass  / 3.6e6  # [kWh]
        regeneratedEnergy = regeneratedEnergy * self.totalMass  / 3.6e6  # [kWh]
        netEnergyUsed = usedEnergy - regeneratedEnergy  # [kWh]
        mechanicalLosses = netEnergyUsed - driveTrainLosses  # [kWh]

        energyDict = {
            "Used energy [kWh]": usedEnergy,
            "Regenerated energy [kWh]": regeneratedEnergy,
            "Net energy used [kWh]": netEnergyUsed,
            "Drive train losses [kWh]": driveTrainLosses,
            "Mechanical losses [kWh]": mechanicalLosses
        }

        return energyDict


    def printStats(self, energyStats):

        print("")
        print("Energy statistics")
        print("-----------------")

        for key, value in energyStats.items():
            print("{}: {:.2f}".format(key, value))


if __name__ == '__main__':

    dfTarget = pd.read_pickle("../data/StGallenWilTrajectory01.pkl")

    train = Train(config={'id':'CH_Stadler_FLIRT_TPF'}, pathJSON='../trains')
    train.forceMinPn = 0
    train.withPnBrake = False
    train.powerLosses = get_power_loss_function(train, "static")

    track = Track(config={'id':'CH_StGallen_Wil'}, pathJSON='../tracks')

    optsDict = {'numIntervals': 600, 'integrationMethod': 'RK', 'integrationOptions': {'numApproxSteps': 1}, 'energyOptimal': True}

    forceEstimator = forceEstimator(dfTarget, train, track, optsDict=optsDict, trainLengthDependentValues=True)
    dfEstimate = forceEstimator.estimate()

    plotVelocityComparison(dfTarget, dfEstimate)
    plotForceComparison(dfTarget, dfEstimate)
    plotTimeCoparison(dfTarget, dfEstimate)

    energyEstimator = energyEstimator(dfEstimate, train, track=track, optsDict=optsDict)
    energyStats = energyEstimator.estimate()
    energyEstimator.printStats(energyStats)