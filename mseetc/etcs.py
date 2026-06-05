from bisect import bisect_right
from idlelib.editor import prepstr
from multiprocessing.spawn import prepare

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mseetc.track import Track


def compute_A_brake_safe(trainBrakingData):

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


def compute_A_gradient(currentPosition, gradients):

    positions = gradients.index.values
    gradients = gradients["Gradient [permil]"].values

    idx = bisect_right(positions, currentPosition)
    idx = max(0, min(idx, len(positions) - 1))

    if idx == 0:

        return 0

    gradient = gradients[idx-1]
    A_gradient = 9.81 * gradient * 0.001
    return A_gradient


def compute_braking_curve(braking_profile, gradients, target_position, permittedVelocity, targetVerlocity, dt=0.1):

    max_velocity = permittedVelocity * 1.4

    positions = [target_position]
    velocities = [targetVerlocity]

    threshold_velocities = list(braking_profile["velocity [m/s]"])
    braking_values = list(braking_profile["value [m/s^2]"])

    threshold_velocities.append(200)  # basically inf velocity

    for idx in range(len(threshold_velocities) - 1):

        if targetVerlocity > threshold_velocities[idx + 1]:

            continue

        threshold_velocity = threshold_velocities[idx + 1]
        A_brake = braking_values[idx]

        while True:

            A_gradient = compute_A_gradient(positions[-1], gradients)

            v_old = velocities[-1]
            v_new = v_old - (A_brake - A_gradient) * dt
            x_new = positions[-1] - 0.5 * (v_new + v_old) * dt

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


def compute_EBI_curve(EBD_curve, trainBrakingData, targetSpeed):

    positionsEBD = EBD_curve.index.to_numpy()
    velocitiesEBD = EBD_curve["Velocity [m/s]"].to_numpy()

    T_traction = trainBrakingData["T_traction [s]"]
    T_be = trainBrakingData["T_be [s]"]
    Kt_int = trainBrakingData["Kt_int [-]"]
    v_uncertainty = trainBrakingData["v_uncertainty [%]"] * 0.01

    t_be = T_be * Kt_int
    T_berem = max(t_be - T_traction, 0)

    positionsEBI = []
    velocitiesEBI = []

    for pos, vel in zip(positionsEBD, velocitiesEBD):

        A_est1 = 0.1 # todo
        A_est2 = min(A_est1, 0.4)

        V_est = (vel - A_est1*T_traction - A_est2*T_berem)/(1+v_uncertainty)
        V_est = max(V_est, 0.0)

        if V_est < targetSpeed:

            velocitiesEBI.append(targetSpeed)
            positionsEBI.append(positionsEBI[-1]+1)

            break

        velocitiesEBI.append(V_est)

        V_delta_0 = V_est * v_uncertainty
        V_delta1 = A_est1 * T_traction
        V_delta2 = A_est2 * T_berem
        D_bec = T_traction * (V_est+V_delta_0+0.5*V_delta1) + T_berem * (V_est+V_delta_0+V_delta1+0.5*V_delta2)
        positionsEBI.append(pos - D_bec)

    EBI_curve = pd.DataFrame(
        {"Velocity [m/s]": velocitiesEBI},
        index=positionsEBI,
    )

    EBI_curve.index.name = "Position [m]"

    return EBI_curve


def shift_curve_by_time(dfCurve, timeShift):

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



def compute_SBI_curve(SBI1_curve, SBI2_curve):

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


def getCurveStyles():

    curveStyles = {
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

    return curveStyles


def plotCurves(curves, targetPosition, permittedSpeed, targetSpeed, prePottingDistance, postPlottingDistance):

    curveStyles = getCurveStyles()

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.step(
        np.array([targetPosition - prePottingDistance, targetPosition, targetPosition + postPlottingDistance]) / 1000,
        np.array([permittedSpeed, permittedSpeed, targetSpeed]) * 3.6,
        label="Speed limit", color="black", linewidth= 2.0
    )

    for name, curve in curves.items():

        style = {}

        if curveStyles is not None and name in curveStyles:
            style = curveStyles[name]

        ax.plot(curve.index.values / 1000, curve["Velocity [m/s]"] * 3.6, label=name, **style)

    ax.set_title("ETCS Braking Curves")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    plt.show()


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


def postPreProcessCurves(curves, targetPosition, permittedSpeed, plottingDistance):

    start_position = targetPosition - plottingDistance

    speedLimits = computeCeilingSpeedLimits(permittedSpeed)

    curves["EBI"] = trimCurveToMaxVelocity(curves["EBI"], speedLimits["EBI [m/s]"])
    curves["EBI"] = addStartPointToCurve(curves["EBI"], speedLimits["EBI [m/s]"], start_position)

    curves["SBI"] = trimCurveToMaxVelocity(curves["SBI"], speedLimits["SBI [m/s]"])
    curves["SBI"] = addStartPointToCurve(curves["SBI"], speedLimits["SBI [m/s]"], start_position)

    curves["W"] = trimCurveToMaxVelocity(curves["W"], speedLimits["Warning [m/s]"])
    curves["W"] = addStartPointToCurve(curves["W"], speedLimits["Warning [m/s]"], start_position)

    curves["P"] = trimCurveToMaxVelocity(curves["P"], permittedSpeed)
    curves["P"] = addStartPointToCurve(curves["P"], permittedSpeed, start_position)

    curves["I"] = trimCurveToMaxVelocity(curves["I"], permittedSpeed)


    curves["EBD"] = trimCurveFromMinPosition(curves["EBD"], curves["EBI"].index.to_numpy(dtype=float)[2])

    curves["SBI2"] = trimCurveFromMinPosition(curves["SBI2"], curves["EBI"].index.to_numpy(dtype=float)[2])

    curves["SBD"] = trimCurveFromMinPosition(curves["SBD"], curves["SBI"].index.to_numpy(dtype=float)[2])

    curves["SBI1"] = trimCurveFromMinPosition(curves["SBI1"], curves["SBI"].index.to_numpy(dtype=float)[2])

    return curves


def postPostProcessCurves(curves, targetPosition, targetSpeed, postPlottingDistance):

    end_position = targetPosition + postPlottingDistance

    speedLimits = computeCeilingSpeedLimits(targetSpeed)

    curves["EBI"] = trimCurveFromMinVelocity(curves["EBI"], speedLimits["EBI [m/s]"])
    curves["EBI"] = addEndPointToCurve(curves["EBI"], speedLimits["EBI [m/s]"], end_position)

    curves["SBI"] = trimCurveFromMinVelocity(curves["SBI"], speedLimits["SBI [m/s]"])
    curves["SBI"] = addEndPointToCurve(curves["SBI"], speedLimits["SBI [m/s]"], end_position)

    curves["W"] = trimCurveFromMinVelocity(curves["W"], speedLimits["Warning [m/s]"])
    curves["W"] = addEndPointToCurve(curves["W"], speedLimits["Warning [m/s]"], end_position)

    curves["P"] = addEndPointToCurve(curves["P"], targetSpeed, end_position)

    curves["I"] = addEndPointToCurve(curves["I"], targetSpeed, curves["P"].index.to_numpy(dtype=float)[-4])


    curves["EBD"] = trimCurveFromMinVelocity(curves["EBD"], speedLimits["EBI [m/s]"])

    curves["SBI2"] = trimCurveFromMinVelocity(curves["SBI2"], speedLimits["SBI [m/s]"])

    curves["SBD"] = trimCurveFromMinVelocity(curves["SBD"], speedLimits["EBI [m/s]"])

    curves["SBI1"] = trimCurveFromMinVelocity(curves["SBI1"], speedLimits["SBI [m/s]"])

    return curves


def computeStepApproximation(curve, step=50):

    positions = curve.index.to_numpy(dtype=float)
    velocities = curve["Velocity [m/s]"].to_numpy(dtype=float)

    samplePositions = np.arange(positions[1], positions[-1], step)
    sampleVelocities = np.interp(samplePositions, positions, velocities)

    approximation = pd.DataFrame(
        {"Velocity [m/s]": sampleVelocities},
        index=samplePositions,
    )

    approximation.index.name = "Position [m]"

    return approximation


def plotApproximation(approximation, curve, name):

    curveStyles = getCurveStyles()

    fig, ax = plt.subplots(figsize=(16, 8))

    style = {}

    if curveStyles is not None and name in curveStyles:
        style = curveStyles[name]

    ax.plot(curve.index.values / 1000, curve["Velocity [m/s]"] * 3.6, label=name, **style)
    ax.step(approximation.index.values / 1000, approximation["Velocity [m/s]"] * 3.6, label="Step Approxmiation")

    ax.set_title("ETCS Braking Curves")
    ax.set_xlabel("Position [km]")
    ax.set_ylabel("Velocity [km/h]")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.figure.tight_layout()

    plt.show()


if __name__ == '__main__':

    # ETCS Constants
    T_warning = 2  # [s]
    T_driver = 4  # [s]

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

    targetPosition = 5000  # [m]
    overlap = 100  # [m]
    permittedSpeed = 160 / 3.6  # [m/s]
    targetSpeed = 24  # [m/s]

    prepPlottingDistance = 3000  # [m]
    postPlottingDistance = 1000  # [m]

    SvL = targetPosition + overlap
    EoA = targetPosition

    assert permittedSpeed >= 0 and permittedSpeed < 400 / 3.6

    assert targetPosition > 0 and targetPosition < track.length

    assert targetSpeed >= 0 and targetSpeed < permittedSpeed


    df_A_brake_safe = compute_A_brake_safe(trainBrakingData)

    T_indication = max(0.8 * trainBrakingData["T_bs [s]"], 5) + T_driver

    curves = {}

    curves["EBD"] = compute_braking_curve(df_A_brake_safe, track.gradients, SvL, permittedSpeed, targetSpeed)

    curves["EBI"] = compute_EBI_curve(curves["EBD"], trainBrakingData, targetSpeed)

    curves["SBI2"] = shift_curve_by_time(curves["EBI"], trainBrakingData["T_bs2 [s]"])

    curves["SBD"] = compute_braking_curve(trainBrakingData["A_brake_service [m/s^2]"], track.gradients, EoA, permittedSpeed, targetSpeed)

    curves["SBI1"] = shift_curve_by_time(curves["SBD"], trainBrakingData["T_bs1 [s]"])

    curves["SBI"] = compute_SBI_curve(curves["SBI1"], curves["SBI2"])

    curves["W"] = shift_curve_by_time(curves["SBI"], T_warning)

    curves["P"] = shift_curve_by_time(curves["SBI"], T_driver)

    curves["I"] = shift_curve_by_time(curves["P"], T_indication)

    curves = postPreProcessCurves(curves, targetPosition, permittedSpeed, prepPlottingDistance)

    if targetSpeed > 0:

        curves = postPostProcessCurves(curves, targetPosition, targetSpeed, postPlottingDistance)

    plotCurves(curves, targetPosition, permittedSpeed, targetSpeed, prepPlottingDistance, postPlottingDistance)

    # approximation = computeStepApproximation(curves["P"])
    # plotApproximation(approximation, curves["P"], "P")
