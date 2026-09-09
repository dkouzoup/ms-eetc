"""
Signals that can be plotted from a trajectory dataframe.

The wrappers store forces in N, velocities in m/s and positions in m, while the
quantities one wants to look at are km/h, kN, kW and km. Every signal declares the
raw columns it needs, so that the dashboard only offers what a given run contains
(the estimator, for instance, stores no energies).
"""

X_AXES = {
    'Distance traveled [km]': ('Distance traveled [m]', 1e-3),
    'Position (track) [km]': ('Position (abs) [m]', 1e-3),
    'Position (section) [km]': ('Position [m]', 1e-3),
    'Time (journey) [s]': ('Time (abs) [s]', 1),
    'Time (section) [s]': ('Time [s]', 1),
}

# the speed limits are a property of the track, not a column of the trajectory
SPEED_LIMIT = 'Speed limit [km/h]'

ETCS_CURVE = 'ETCS braking curve [km/h]'

# limits the trajectory has to respect, drawn as steps on top of the velocity
LIMIT_SIGNALS = (SPEED_LIMIT, ETCS_CURVE)


def _scaled(column, factor):

    return {'columns': [column], 'fun': lambda df: df[column]*factor}


def _cumulative(column):

    return {'columns': [column], 'fun': lambda df: df[column].fillna(0).cumsum()}


SIGNALS = {
    'Velocity [km/h]': _scaled('Velocity [m/s]', 3.6),
    'Velocity - cvodes [km/h]': _scaled('Velocity - cvodes [m/s]', 3.6),
    ETCS_CURVE: _scaled('ETCS braking curve [m/s]', 3.6),
    'Force [kN]': {
        'columns': ['Force (el) [N]', 'Force (pnb) [N]'],
        'fun': lambda df: (df['Force (el) [N]'].fillna(0) + df['Force (pnb) [N]'].fillna(0))*1e-3,
    },
    'Force (el) [kN]': _scaled('Force (el) [N]', 1e-3),
    'Force (pnb) [kN]': _scaled('Force (pnb) [N]', 1e-3),
    'Power (el) [kW]': {
        'columns': ['Force (el) [N]', 'Velocity [m/s]'],
        'fun': lambda df: df['Force (el) [N]']*df['Velocity [m/s]']*1e-3,
    },
    'Acceleration [m/s^2]': _scaled('Acceleration [m/s^2]', 1),
    'Gradient [permil]': _scaled('Gradient [permil]', 1),
    'Cumulative energy [kWh]': _cumulative('Energy [kWh]'),
    'Cumulative losses [kWh]': _cumulative('Losses [kWh]'),
}

STEP_SIGNALS = {SPEED_LIMIT, ETCS_CURVE, 'Force [kN]', 'Force (el) [kN]', 'Force (pnb) [kN]',
    'Gradient [permil]'}

DEFAULT_SIGNALS = ['Velocity [km/h]']


def isAvailable(name, df):

    return name in SIGNALS and all(column in df.columns for column in SIGNALS[name]['columns'])


def availableSignals(df):

    return [name for name in SIGNALS if isAvailable(name, df)]


def evaluate(name, df):

    return SIGNALS[name]['fun'](df)


def availableXAxes(df):

    return [name for name, (column, _) in X_AXES.items() if column in df.columns]


def evaluateXAxis(name, df):

    column, factor = X_AXES[name]

    return df[column]*factor
