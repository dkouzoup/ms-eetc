from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.track import Track


def shiftCurveByTime(dfCurve, timeShift):

    positionsOriginal = dfCurve.index.to_numpy()
    velocitiesOriginal = dfCurve["Velocity [m/s]"].to_numpy()

    velocitiesShifted = velocitiesOriginal.copy()
    positionsShifted = positionsOriginal - velocitiesShifted * timeShift

    dfCurveShifted = pd.DataFrame(
        {"Velocity [m/s]": velocitiesShifted},
        index=positionsShifted,
    )

    dfCurveShifted.index.name = "Position [m]"

    return dfCurveShifted


def computeCeilingSpeedLimits(V_permitted_mps):

    dV_ebi_min = 7.5/3.6
    dV_ebi_max = 15.0/3.6
    V_ebi_min = 110.0/3.6
    V_ebi_max = 210.0/3.6

    C_ebi = (dV_ebi_max - dV_ebi_min) / (V_ebi_max - V_ebi_min)

    if V_permitted_mps <= V_ebi_min:

        dV_ebi = dV_ebi_min

    else:
        dV_ebi = min(dV_ebi_min + C_ebi * (V_permitted_mps - V_ebi_min),dV_ebi_max,)

    dV_warning = 0.5 * dV_ebi
    dV_sbi = 0.75 * dV_ebi

    return {
        "Warning [m/s]": V_permitted_mps + dV_warning,
        "SBI [m/s]": V_permitted_mps + dV_sbi,
        "EBI [m/s]": V_permitted_mps + dV_ebi,
    }


def addStartPointToCurve(curve, velocity, start_position):

    delta_s = curve.index.to_numpy(dtype=float)[1] - curve.index.to_numpy(dtype=float)[0]

    start_point = pd.DataFrame(
        {"Velocity [m/s]": [velocity, velocity]},
        index=[start_position, curve.index.to_numpy(dtype=float)[0] - delta_s],
    )
    start_point.index.name = "Position [m]"

    return pd.concat([start_point, curve])


def addEndPointToCurve(curve, velocity, end_position):

    delta_s = curve.index.to_numpy(dtype=float)[-1] - curve.index.to_numpy(dtype=float)[-2]

    end_point = pd.DataFrame(
        {"Velocity [m/s]": [velocity, velocity]},
        index=[curve.index.to_numpy(dtype=float)[-1] + delta_s, end_position],
    )
    end_point.index.name = "Position [m]"

    return pd.concat([curve, end_point])


def trimCurveToMaxVelocity(curve, maxVelocity):

    velocities = curve["Velocity [m/s]"].to_numpy(dtype=float)
    keep_mask = velocities <= maxVelocity

    return curve[keep_mask].copy()


def trimCurveFromMinVelocity(curve, minVelocity):

    velocities = curve["Velocity [m/s]"].to_numpy(dtype=float)
    keep_mask = velocities >= minVelocity

    return curve[keep_mask].copy()


def trimCurveFromMinPosition(curve, minPosition):

    positions = curve.index.to_numpy(dtype=float)
    keep_mask = positions >= minPosition

    return curve[keep_mask].copy()


@dataclass(frozen=True)
class BrakingTarget:
    position: float  # [m]
    overlap: float  # [m]
    permittedVelocity: float  # [m/s]
    targetVelocity: float  # [m/s]

    @property
    def EoA(self):
        return self.position

    @property
    def SvL(self):
        return self.position + self.overlap


class EtcsBrakingCurveCalculator:

    def __init__(self, trainBrakingData, track, distancePre=3000, distancePost=1000):

        self.trainBrakingData = trainBrakingData
        self.track = track

        self.dt = 0.1  # [s]
        self.distancePre = distancePre
        self.distancePost = distancePost

        # ETCS Constants
        self.T_warning = 2.0  # [s]
        self.T_driver = 4.0  # [s]

        self.curveStyles = {
            "EBD": {"color": "blue", "linestyle": "-", "linewidth": 2.0},
            "EBI": {"color": "blue", "linestyle": ":", "linewidth": 1.5},
            "SBD": {"color": "green", "linestyle": "-", "linewidth": 2.0},
            "SBI1": {"color": "green", "linestyle": "--", "linewidth": 1.5},
            "SBI2": {"color": "blue", "linestyle": "--", "linewidth": 1.5},
            "SBI": {"color": "red", "linestyle": "-", "linewidth": 2.0},
            "W": {"color": "orange", "linestyle": "-", "linewidth": 2.0},
            "P": {"color": "grey", "linestyle": "-", "linewidth": 2.0},
            "I": {"color": "gold", "linestyle": "-", "linewidth": 2.0},
        }


    def validateInput(self, target):

        if not 0 <= target.permittedVelocity < 400 / 3.6:
            raise ValueError("permittedVelocity must be between 0 and 400 km/h.")

        if not 0 < target.EoA < self.track.length:
            raise ValueError("EoA must lie within the track length.")

        if not 0 < target.SvL < self.track.length:
            raise ValueError("SvL must lie within the track length.")

        if not 0 <= target.targetVelocity < target.permittedVelocity:
            raise ValueError("targetVelocity must be lower than permittedVelocity.")


    def compute_A_brake_safe(self):

        trainBrakingData = self.trainBrakingData

        braking = trainBrakingData["A_brake_emergency [m/s^2]"]

        velocities = braking["velocity [m/s]"]
        A_emergency_values = braking["value [m/s^2]"]

        K_dry_rst = trainBrakingData["K_dry_rst [-]"]
        M_NVAVADH = trainBrakingData["M_NVAVADH [-]"]
        K_wet_rst = trainBrakingData["K_wet_rst [-]"]

        K_wet_corr = K_wet_rst + M_NVAVADH * (1 - K_wet_rst)

        A_brake_safe_values = [
            A_emergency * K_dry_rst * K_wet_corr
            for A_emergency in A_emergency_values
        ]

        return {
            "velocity [m/s]": velocities,
            "value [m/s^2]": A_brake_safe_values,
        }


    def compute_A_gradient(self, currentPosition):

        positions = self.track.gradients.index.values
        gradients = self.track.gradients["Gradient [permil]"].values

        idx = bisect_right(positions, currentPosition) - 1
        idx = max(0, min(idx, len(positions) - 1))

        if idx == 0:

            return 0  # [start of track has been exceeded]

        gradient = gradients[idx]
        return 9.81 * gradient * 0.001


    def computeBrakingCurve(self, brakingProfile, target_position, permittedVelocity, targetVelocity):

        max_velocity = permittedVelocity * 1.4

        positions = [target_position]
        velocities = [targetVelocity]

        threshold_velocities = list(brakingProfile["velocity [m/s]"])
        braking_values = list(brakingProfile["value [m/s^2]"])

        threshold_velocities.append(200)  # basically inf velocity

        for idx in range(len(threshold_velocities) - 1):

            if targetVelocity > threshold_velocities[idx + 1]:
                continue

            threshold_velocity = threshold_velocities[idx + 1]
            A_brake = braking_values[idx]

            while True:

                A_gradient = self.compute_A_gradient(positions[-1])

                v_old = velocities[-1]
                v_new = v_old - (A_brake - A_gradient) * self.dt
                x_new = positions[-1] - 0.5 * (v_new + v_old) * self.dt

                positions.append(x_new)
                velocities.append(v_new)

                if v_new >= threshold_velocity or v_new >= max_velocity:
                    break

            if velocities[-1] >= max_velocity:
                break

        curve = pd.DataFrame(
            {"Velocity [m/s]": velocities[::-1]},
            index=positions[::-1],
        )

        curve.index.name = "Position [m]"

        return curve


    def compute_EBI_curve(self, EBD_curve, targetVelocity):

        trainBrakingData = self.trainBrakingData

        T_traction = trainBrakingData["T_traction [s]"]
        T_be = trainBrakingData["T_be [s]"]
        Kt_int = trainBrakingData["Kt_int [-]"]
        v_uncertainty = trainBrakingData["v_uncertainty [%]"] * 0.01

        positionsEBD = EBD_curve.index.to_numpy()
        velocitiesEBD = EBD_curve["Velocity [m/s]"].to_numpy()

        t_be = T_be * Kt_int
        T_berem = max(t_be - T_traction, 0)

        positionsEBI = []
        velocitiesEBI = []

        for pos, vel in zip(positionsEBD, velocitiesEBD):

            A_est1 = 0.1  # todo
            A_est2 = min(A_est1, 0.4)

            V_est = (vel - A_est1 * T_traction - A_est2 * T_berem) / (1 + v_uncertainty)
            V_est = max(V_est, 0.0)

            if V_est < targetVelocity:
                velocitiesEBI.append(targetVelocity)
                positionsEBI.append(positionsEBI[-1] + 1)

                break

            velocitiesEBI.append(V_est)

            V_delta_0 = V_est * v_uncertainty
            V_delta1 = A_est1 * T_traction
            V_delta2 = A_est2 * T_berem
            D_bec = T_traction * (V_est + V_delta_0 + 0.5 * V_delta1) + T_berem * (
                        V_est + V_delta_0 + V_delta1 + 0.5 * V_delta2)
            positionsEBI.append(pos - D_bec)

        EBI_curve = pd.DataFrame(
            {"Velocity [m/s]": velocitiesEBI},
            index=positionsEBI,
        )

        EBI_curve.index.name = "Position [m]"

        return EBI_curve


    def compute_SBI_curve(self, SBI1_curve, SBI2_curve):

        positionsSBI1 = SBI1_curve.index.to_numpy()
        velocitiesSBI1 = SBI1_curve["Velocity [m/s]"].to_numpy()

        positionsSBI2 = SBI2_curve.index.to_numpy()
        velocitiesSBI2 = SBI2_curve["Velocity [m/s]"].to_numpy()

        # Use only the overlapping position range
        minPosition = max(positionsSBI1.min(), positionsSBI2.min())
        maxPosition = min(positionsSBI1.max(), positionsSBI2.max())

        step = 10.0  # [m]
        positionsSBI = np.arange(minPosition, maxPosition + step, step)
        positionsSBI[-1] = maxPosition

        velocitiesSBI1_interpol = np.interp(positionsSBI, positionsSBI1, velocitiesSBI1)
        velocitiesSBI2_interpol = np.interp(positionsSBI, positionsSBI2, velocitiesSBI2)

        velocitiesSBI = np.minimum(velocitiesSBI1_interpol, velocitiesSBI2_interpol)

        SBI_curve = pd.DataFrame(
            {"Velocity [m/s]": velocitiesSBI},
            index=positionsSBI,
        )

        SBI_curve.index.name = "Position [m]"

        return SBI_curve


    def postPreProcessCurves(self, curves, target):

        permittedVelocity = target.permittedVelocity
        start_position = target.position - self.distancePre

        speedLimits = computeCeilingSpeedLimits(permittedVelocity)

        curves["EBI"] = trimCurveToMaxVelocity(curves["EBI"], speedLimits["EBI [m/s]"])
        curves["EBI"] = addStartPointToCurve(curves["EBI"], speedLimits["EBI [m/s]"], start_position)

        curves["SBI"] = trimCurveToMaxVelocity(curves["SBI"], speedLimits["SBI [m/s]"])
        curves["SBI"] = addStartPointToCurve(curves["SBI"], speedLimits["SBI [m/s]"], start_position)

        curves["W"] = trimCurveToMaxVelocity(curves["W"], speedLimits["Warning [m/s]"])
        curves["W"] = addStartPointToCurve(curves["W"], speedLimits["Warning [m/s]"], start_position)

        curves["P"] = trimCurveToMaxVelocity(curves["P"], permittedVelocity)
        curves["P"] = addStartPointToCurve(curves["P"], permittedVelocity, start_position)

        curves["I"] = trimCurveToMaxVelocity(curves["I"], permittedVelocity)

        curves["EBD"] = trimCurveFromMinPosition(curves["EBD"], curves["EBI"].index.to_numpy(dtype=float)[2])

        curves["SBI2"] = trimCurveFromMinPosition(curves["SBI2"], curves["EBI"].index.to_numpy(dtype=float)[2])

        curves["SBD"] = trimCurveFromMinPosition(curves["SBD"], curves["SBI"].index.to_numpy(dtype=float)[2])

        curves["SBI1"] = trimCurveFromMinPosition(curves["SBI1"], curves["SBI"].index.to_numpy(dtype=float)[2])

        return curves

    def postPostProcessCurves(self, curves, target):

        targetVelocity = target.targetVelocity
        end_position = target.position + self.distancePost

        speedLimits = computeCeilingSpeedLimits(targetVelocity)

        curves["EBI"] = trimCurveFromMinVelocity(curves["EBI"], speedLimits["EBI [m/s]"])
        curves["EBI"] = addEndPointToCurve(curves["EBI"], speedLimits["EBI [m/s]"], end_position)

        curves["SBI"] = trimCurveFromMinVelocity(curves["SBI"], speedLimits["SBI [m/s]"])
        curves["SBI"] = addEndPointToCurve(curves["SBI"], speedLimits["SBI [m/s]"], end_position)

        curves["W"] = trimCurveFromMinVelocity(curves["W"], speedLimits["Warning [m/s]"])
        curves["W"] = addEndPointToCurve(curves["W"], speedLimits["Warning [m/s]"], end_position)

        curves["P"] = addEndPointToCurve(curves["P"], targetVelocity, end_position)

        curves["I"] = addEndPointToCurve(curves["I"], targetVelocity, curves["P"].index.to_numpy(dtype=float)[-4])

        curves["EBD"] = trimCurveFromMinVelocity(curves["EBD"], speedLimits["EBI [m/s]"])

        curves["SBI2"] = trimCurveFromMinVelocity(curves["SBI2"], speedLimits["SBI [m/s]"])

        curves["SBD"] = trimCurveFromMinVelocity(curves["SBD"], speedLimits["EBI [m/s]"])

        curves["SBI1"] = trimCurveFromMinVelocity(curves["SBI1"], speedLimits["SBI [m/s]"])

        return curves


    def computeTarget(self, target):

        self.validateInput(target)
        trainBrakingData = self.trainBrakingData

        df_A_brake_safe = self.compute_A_brake_safe()
        T_indication = max(0.8 * trainBrakingData["T_bs [s]"], 5) + self.T_driver

        curves = {}

        curves["EBD"] = self.computeBrakingCurve(df_A_brake_safe, target.SvL, target.permittedVelocity, target.targetVelocity)

        curves["EBI"] = self.compute_EBI_curve(curves["EBD"], target.targetVelocity)

        curves["SBI2"] = shiftCurveByTime(curves["EBI"], trainBrakingData["T_bs2 [s]"])

        curves["SBD"] = self.computeBrakingCurve(self.trainBrakingData["A_brake_service [m/s^2]"], target.EoA, target.permittedVelocity, target.targetVelocity)

        curves["SBI1"] = shiftCurveByTime(curves["SBD"], trainBrakingData["T_bs1 [s]"])

        curves["SBI"] = self.compute_SBI_curve(curves["SBI1"], curves["SBI2"])

        curves["W"] = shiftCurveByTime(curves["SBI"], self.T_warning)

        curves["P"] = shiftCurveByTime(curves["SBI"], self.T_driver)

        curves["I"] = shiftCurveByTime(curves["P"], T_indication)

        curves = self.postPreProcessCurves(curves, target)

        if target.targetVelocity > 0:

            curves = self.postPostProcessCurves(curves, target)

        return curves


    def plotCurves(self, curves, target):

        targetPosition = target.EoA
        permittedVelocity = target.permittedVelocity
        targetVelocity = target.targetVelocity

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.step(
            np.array(
                [targetPosition - self.distancePre, targetPosition, targetPosition + self.distancePost]) / 1000,
            np.array([permittedVelocity, permittedVelocity, targetVelocity]) * 3.6,
            label="Speed limit", color="black", linewidth=2.0
        )

        for name, curve in curves.items():

            style = {}

            if self.curveStyles is not None and name in self.curveStyles:
                style = self.curveStyles[name]

            ax.plot(curve.index.values / 1000, curve["Velocity [m/s]"] * 3.6, label=name, **style)

        ax.set_title("ETCS Braking Curves")
        ax.set_xlabel("Position [km]")
        ax.set_ylabel("Velocity [km/h]")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.figure.tight_layout()

        plt.show()


# def computeStepApproximation(curve, step=50):
#
#     positions = curve.index.to_numpy(dtype=float)
#     velocities = curve["Velocity [m/s]"].to_numpy(dtype=float)
#
#     samplePositions = np.arange(positions[1], positions[-1], step)
#     sampleVelocities = np.interp(samplePositions, positions, velocities)
#
#     approximation = pd.DataFrame(
#         {"Velocity [m/s]": sampleVelocities},
#         index=samplePositions,
#     )
#
#     approximation.index.name = "Position [m]"
#
#     return approximation
#
#
# def plotApproximation(approximation, curve, name):
#
#     curveStyles = getCurveStyles()
#
#     fig, ax = plt.subplots(figsize=(16, 8))
#
#     style = {}
#
#     if curveStyles is not None and name in curveStyles:
#         style = curveStyles[name]
#
#     ax.plot(curve.index.values / 1000, curve["Velocity [m/s]"] * 3.6, label=name, **style)
#     ax.step(approximation.index.values / 1000, approximation["Velocity [m/s]"] * 3.6, label="Step Approxmiation")
#
#     ax.set_title("ETCS Braking Curves")
#     ax.set_xlabel("Position [km]")
#     ax.set_ylabel("Velocity [km/h]")
#     ax.grid(True, which="both", linestyle="--", alpha=0.5)
#     ax.legend(loc="upper right")
#     ax.figure.tight_layout()
#
#     plt.show()


if __name__ == '__main__':

    # Convention:
    # Braking accelerations are stored as negative values.
    # Gradient acceleration is positive for uphill and negative for downhill.

    trainBrakingData = {
        "A_brake_emergency [m/s^2]": {
            "velocity [m/s]": [0, 20, 40, 60],
            "value [m/s^2]": [-0.9, -0.85, -0.8, -0.75],
        },
        "A_brake_service [m/s^2]": {
            "velocity [m/s]": [0, 20, 40, 60],
            "value [m/s^2]": [-0.5, -0.45, -0.4, -0.35],
        },
        "K_dry_rst [-]": 0.8,
        "M_NVAVADH [-]": 0,
        "K_wet_rst [-]": 0.9,
        "T_traction [s]": 1,
        "T_be [s]": 4,
        "Kt_int [-]": 1.15,
        "v_uncertainty [%]": 2.98,
        "T_bs [s]": 3,
        "T_bs1 [s]": 3,
        "T_bs2 [s]": 3,
    }

    track = Track(config={'id': 'CH_StGallen_Wil'}, pathJSON='../tracks')

    target = BrakingTarget(
            position=5000,
            overlap= 100,
            permittedVelocity=160/3.6,
            targetVelocity=0/3.6
    )

    calculator = EtcsBrakingCurveCalculator(trainBrakingData, track, distancePre=5000, distancePost=1000)
    curve_set = calculator.computeTarget(target)

    calculator.plotCurves(curve_set, target)

    # approximation = computeStepApproximation(curves["P"])
    # plotApproximation(approximation, curves["P"], "P")
