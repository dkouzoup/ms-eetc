"""
Dashboard for the results of the multi-stop wrappers.

Run it with:

    streamlit run scripts/dashboard/app.py

Point the sidebar at a data directory (e.g. "innocheque") and select any number of
run directories to overlay their trajectories, compare their energy consumption and
inspect the parameters each of them was computed with.
"""

import sys

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parents[1] / 'multiStopWrappers'))

import signals as sig

from runData import findRuns, loadSection

CHART_HEIGHT = 320

# the x range every chart is zoomed to, set by dragging over any one of them
ZOOM_KEY = 'xRange'

# bumped whenever that range changes, to give the charts keys they have never had before
# and so leave them without a stale brush drawn over them
ZOOM_GENERATION = 'zoomGeneration'

# energy quantities the wrappers report, in the order in which they are offered
ENERGY_COLUMNS = ['Energy SIM [kWh]', 'Energy OCP [kWh]', 'Net energy used [kWh]',
    'Used energy [kWh]', 'Regenerated energy [kWh]', 'Drive train losses [kWh]',
    'Mechanical losses [kWh]']


def defaultSearchRoot():
    "Start from the data directory of the wrapper configuration, if there is one."

    try:

        from wrapperConfig import loadConfig

        directory = Path(loadConfig()['directory'])

    except Exception:

        return str(Path.cwd())

    if not directory.is_absolute():

        directory = (Path(__file__).parents[1] / 'multiStopWrappers' / directory).resolve()

    return str(directory.parent if directory.parent.is_dir() else directory)


def searchRootKey(root):
    "Cheap fingerprint of the directory tree, so that new runs are picked up automatically."

    patterns = ('*', '*/*', '*/*/*')

    return sum(p.stat().st_mtime for pattern in patterns for p in root.glob(pattern) if p.is_dir())


@st.cache_data(show_spinner=False)
def cachedRuns(root, treeKey):

    return findRuns(root)


@st.cache_data(show_spinner=False)
def cachedTrajectory(path, positionStart, timeOffset, absolutePositions, withEtcs,
    distanceOffset, mtime):

    from runData import Section

    return loadSection(Section(index=0, name='', path=Path(path), positionStart=positionStart,
        timeOffset=timeOffset, absolutePositions=absolutePositions,
        withEtcsBrakingCurves=withEtcs, distanceOffset=distanceOffset))


def trajectoryOf(section):

    return cachedTrajectory(str(section.path), section.positionStart, section.timeOffset,
        section.absolutePositions, section.withEtcsBrakingCurves, section.distanceOffset,
        section.path.stat().st_mtime)


def turnaroundDistances(runs):
    "Distance at which the return trip starts, for the runs that cover both directions."

    distances = set()

    for run in runs:

        offsets = [s.distanceOffset for s in run.sections
            if s.direction == 'reverse' and s.distanceOffset is not None]

        if offsets:

            distances.add(min(offsets))

    return sorted(distances)


def speedLimitSeries(section, df):

    """
    Speed limits of a section, placed on the x-axis currently plotted.

    They are kept as the step function the track defines, so the steps land exactly where
    the limit changes instead of on the nearest discretization point of the solver. The
    breakpoints are mapped onto the chosen axis through the trajectory itself, which also
    puts them in the right place on a time axis.
    """

    limits = section.speedLimits

    if not limits:

        return None

    positions = np.asarray(limits['Position [m]'], dtype=float)
    velocities = np.asarray(limits['Speed limit [m/s]'], dtype=float)*3.6

    if isValue(section.positionEnd) and isValue(section.positionStart):

        # draw the last step to the end of the section rather than to its last change
        positions = np.append(positions, section.positionEnd - section.positionStart)
        velocities = np.append(velocities, velocities[-1])

    return np.interp(positions, df['Position [m]'], sig.evaluateXAxis(xAxis, df)), velocities


def flatten(value, prefix=''):
    "Turn nested metadata into a flat {'a.b.c': value} dictionary."

    flat = {}

    if isinstance(value, dict):

        for key, item in value.items():

            flat.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))

    elif isinstance(value, list):

        flat[prefix] = ', '.join(str(v) for v in value) if value else ''

    else:

        flat[prefix] = value

    return flat


def sectionLabel(section):

    return f"{section.index}: {section.name}"


def isValue(value):

    return value is not None and not pd.isna(value)


def runParameters(run):
    "Flat parameter table of a run, the same one the parameters tab shows."

    if run.metadata is None:

        return {}

    metadata = {key: value for key, value in run.metadata.items() if key != 'sections'}

    return {
        **flatten(metadata),
        **{f"solver.{key}": value for key, value in solverParameters(run).items()},
    }


def solverParameters(run):

    """
    Solver options of a run, collapsed over its journey sections.

    The wrappers set up one solver per section, so the options are stored per section
    even though they only really differ in the number of shooting intervals. Options
    that do differ are listed section by section.
    """

    options = [flatten(section.get('solverOptions') or {}) for section in run.metadata.get('sections', [])]
    options = [option for option in options if option]

    if not options:

        return {}

    collapsed = {}

    for key in sorted({key for option in options for key in option}):

        values = [option.get(key) for option in options]

        collapsed[key] = values[0] if len(set(map(str, values))) == 1 else \
            ', '.join(str(value) for value in values)

    return collapsed


def shortNames(runs):
    "Shortest names that still tell the selected runs apart, used as column headers."

    for key in (lambda run: run.path.name, lambda run: run.name):

        names = [key(run) for run in runs]

        if len(set(names)) == len(names):

            return {run.name: name for run, name in zip(runs, names)}

    return {run.name: run.name for run in runs}


def energyPerSection(run, column):

    "Energy of every section of a run, indexed by section index."

    if run.isLegacy:

        return {}

    return {entry['index']: entry[column] for entry in run.metadata.get('sections', [])
        if isValue(entry.get(column))}


def availableEnergyColumns(runs):

    return [column for column in ENERGY_COLUMNS if any(energyPerSection(run, column) for run in runs)]


def relativeDifference(value, reference):
    "Difference in percent of the reference, which may be negative (recuperating sections)."

    if not isValue(value) or not isValue(reference) or reference == 0:

        return None

    return 100*(value - reference)/abs(reference)


############################################################################
### Sidebar: which runs to look at

st.set_page_config(page_title='ms-eetc results', layout='wide')

st.sidebar.title('Runs')

if 'searchRoot' not in st.session_state:

    st.session_state.searchRoot = defaultSearchRoot()

searchRoot = st.sidebar.text_input('Data directory', key='searchRoot',
    help='Directory holding the results, e.g. "innocheque" or "innocheque/S31". '
         'Its subdirectories are scanned for "ocp" and "estimator" run folders.')

if st.sidebar.button('Rescan', width='stretch'):

    st.cache_data.clear()

root = Path(searchRoot.strip()).expanduser()  # a pasted path often brings whitespace along

if not root.is_dir():

    st.sidebar.error(f"Not a directory: {root}")
    st.stop()

runs = cachedRuns(str(root), searchRootKey(root))

if not runs:

    st.sidebar.warning('No runs found below this directory.')
    st.info('No result directories found. Run `ocpMultiStop.py` or `estimatorMultiStop.py` first, '
        'or point the sidebar at another data directory.')
    st.stop()

runsByName = {run.name: run for run in runs}

parameterTables = {run.name: runParameters(run) for run in runs}

with st.sidebar.expander('Filters'):

    kinds = st.multiselect('Kind', sorted({run.kind for run in runs}),
        default=sorted({run.kind for run in runs}))

    # every parameter that is not the same across all runs is worth filtering on
    values = {}

    for table in parameterTables.values():

        for key, value in table.items():

            values.setdefault(key, set()).add(str(value))

    filterable = sorted(key for key, seen in values.items() if len(seen) > 1)

    parameter = st.selectbox('Parameter', ['(none)'] + filterable,
        help='Keep only the runs whose parameter has one of the selected values, '
             'e.g. solver.withEtcsBrakingCurves.')

    choices = sorted(values.get(parameter, []))

    wanted = st.multiselect('Value', choices, default=choices) if choices else None

matching = [run for run in runs if run.kind in kinds
    and (wanted is None or str(parameterTables[run.name].get(parameter)) in wanted)]

if not matching:

    st.sidebar.warning('No run matches the filters.')

selectedNames = st.sidebar.multiselect('Runs to compare', [run.name for run in matching],
    help='Each entry is one timestamped output directory of a wrapper.')

selectedRuns = [runsByName[name] for name in selectedNames]

if not selectedRuns:

    st.info(f"Found {len(matching)} run(s) in `{root}`. Select one or more in the sidebar "
        "to plot them.")
    st.stop()

sectionOptions = sorted({sectionLabel(s) for run in selectedRuns for s in run.sections if s.hasTrajectory})

selectedSections = st.sidebar.multiselect('Journey sections', sectionOptions, default=sectionOptions)

st.sidebar.title('Plots')

xAxis = st.sidebar.selectbox('x-axis', list(sig.X_AXES), index=0)

signalNames = st.sidebar.multiselect('Signals', list(sig.SIGNALS), default=sig.DEFAULT_SIGNALS)

showLimits = st.sidebar.multiselect('Overlay on velocity', sig.LIMIT_SIGNALS,
    default=[sig.SPEED_LIMIT],
    help='Drawn as dashed steps on every velocity chart. The speed limits come from the '
         'track, adapted to the train length; the ETCS braking curves are what the '
         'trajectory is actually constrained with when they are enabled.')

combineSections = st.sidebar.checkbox('All sections in one chart',
    value=xAxis.startswith(('Distance traveled', 'Position (track)')),
    help='Only meaningful on an axis spanning the whole trip, where sections do not overlap.')

incomplete = [run.name for run in selectedRuns if not run.isComplete]

if incomplete:

    st.sidebar.warning('Still running or stopped early, so some sections are missing: '
        + ', '.join(incomplete))

legacyRuns = [run.name for run in selectedRuns if run.isLegacy and run.kind == 'ocp']

if legacyRuns and xAxis.startswith(('Position (track)', 'Time (journey)')):

    st.sidebar.warning('Runs without metadata.json have no track offsets; they are drawn '
        'relative to the start of each section: ' + ', '.join(legacyRuns))


############################################################################
### Data collection

def collectData():
    "Build one long dataframe with every selected signal of every selected section."

    frames = []
    limitsDone = set()

    for run in selectedRuns:

        for section in run.sections:

            if not section.hasTrajectory or sectionLabel(section) not in selectedSections:

                continue

            df = trajectoryOf(section)

            x = sig.evaluateXAxis(xAxis, df) if xAxis in sig.availableXAxes(df) else None

            if x is None:

                continue

            for name in signalNames:

                if not sig.isAvailable(name, df):

                    continue

                frames.append(pd.DataFrame({
                    'x': x.values,
                    'y': sig.evaluate(name, df).values,
                    'Signal': name,
                    'Series': run.name,
                    'Section': sectionLabel(section),
                }))

            velocitySignals = [n for n in signalNames if n.startswith('Velocity')]

            for limit in showLimits:

                key = (limit, sectionLabel(section))

                if not velocitySignals or key in limitsDone:

                    continue  # one limit per section is enough, they are shared by the runs

                if limit == sig.SPEED_LIMIT:

                    series = speedLimitSeries(section, df)

                elif sig.isAvailable(limit, df):

                    series = x.values, sig.evaluate(limit, df).values

                else:

                    series = None

                if series is None:

                    continue

                limitsDone.add(key)

                for name in velocitySignals:

                    frames.append(pd.DataFrame({
                        'x': series[0],
                        'y': series[1],
                        'Signal': name,
                        'Series': limit.replace(' [km/h]', ''),
                        'Section': sectionLabel(section),
                    }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def lineChart(data, signalName, xRange=None):

    """
    One signal against the current x-axis.

    Dragging over a chart selects a range of the axis, which every chart is then drawn
    with, so that the plots below a speed profile show the same stretch of the trip.
    """

    interpolate = 'step-after' if signalName in sig.STEP_SIGNALS else 'linear'

    scale = alt.Scale(zero=False, nice=False, domain=list(xRange)) if xRange else \
        alt.Scale(zero=False, nice=False)

    encoding = dict(
        x=alt.X('x:Q', title=xAxis, scale=scale),
        y=alt.Y('y:Q', title=signalName, scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N', title=None,
            legend=alt.Legend(orient='top', columns=2, labelLimit=400)),
        detail=alt.Detail('Section:N'),
        tooltip=[alt.Tooltip('Series:N'), alt.Tooltip('Section:N'),
            alt.Tooltip('x:Q', title=xAxis, format='.3f'),
            alt.Tooltip('y:Q', title=signalName, format='.2f')],
    )

    isLimit = data['Series'].isin([name.replace(' [km/h]', '') for name in sig.LIMIT_SIGNALS])

    limits = data[isLimit]
    trajectories = data[~isLimit]

    chart = alt.Chart(trajectories).mark_line(interpolate=interpolate, clip=True).encode(**encoding)

    if not limits.empty:

        chart += alt.Chart(limits).mark_line(interpolate='step-after', strokeDash=[6, 3],
            clip=True).encode(**encoding)

    turnarounds = turnaroundDistances(selectedRuns) if xAxis.startswith('Distance traveled') else []

    if turnarounds:

        # from here on the train is on its way back
        marks = pd.DataFrame({'x': [distance*1e-3 for distance in turnarounds]})

        chart += alt.Chart(marks).mark_rule(strokeDash=[2, 2], color='gray', size=2,
            clip=True).encode(x=alt.X('x:Q'),
            tooltip=alt.Tooltip('x:Q', title='Return trip starts [km]', format='.2f'))

    brush = alt.selection_interval(encodings=['x'], name='brush')

    return chart.properties(height=CHART_HEIGHT).add_params(brush)


def selectedRange(event):
    "The x range a viewer dragged over a chart, if any."

    selection = getattr(event, 'selection', None) or {}

    candidates = [selection] + [value for value in selection.values() if isinstance(value, dict)]

    for candidate in candidates:

        bounds = candidate.get('x')

        if bounds is not None and len(bounds) == 2:

            return min(bounds), max(bounds)

    return None


def applyRange(xRange):
    "Zoom every chart to a range, or reset them all when given None."

    if xRange is None:

        st.session_state.pop(ZOOM_KEY, None)

    else:

        st.session_state[ZOOM_KEY] = xRange

    st.session_state[ZOOM_GENERATION] = st.session_state.get(ZOOM_GENERATION, 0) + 1

    st.rerun()


def showChart(chart, key):

    """
    Draw a chart and adopt any range dragged over it.

    Every chart of the page is redrawn with that range, which is what ties the plots
    below a speed profile to the stretch of trip being looked at.
    """

    event = st.altair_chart(chart, width='stretch', on_select='rerun',
        key=f"{key}-{st.session_state.get(ZOOM_GENERATION, 0)}")

    chosen = selectedRange(event)

    if chosen is not None and chosen != st.session_state.get(ZOOM_KEY):

        applyRange(chosen)


############################################################################
### Main area

st.title('Multi-stop results')

st.caption(f"{len(selectedRuns)} run(s) from `{root}`")

trajectoriesTab, energyTab, parametersTab = st.tabs(['Trajectories', 'Energy', 'Parameters'])

with trajectoriesTab:

    data = collectData()

    if data.empty:

        st.info('Nothing to plot: select at least one signal and one section.')

    else:

        xRange = st.session_state.get(ZOOM_KEY)

        if xRange is not None:

            zoomed, reset = st.columns([4, 1])
            zoomed.caption(f"Zoomed to {xRange[0]:.3f} - {xRange[1]:.3f} of {xAxis}. "
                'Drag over a chart to zoom further, all charts follow.')

            if reset.button('Reset zoom', width='stretch'):

                applyRange(None)

        else:

            st.caption('Drag over a chart to zoom every chart to that range.')

        for name in signalNames:

            signalData = data[data['Signal'] == name]

            if signalData.empty:

                st.caption(f"{name}: not available in the selected runs.")
                continue

            st.subheader(name)

            if combineSections:

                showChart(lineChart(signalData, name, xRange), f"chart-{name}")

            else:

                for label in sorted(signalData['Section'].unique()):

                    st.caption(f"Section {label}")
                    showChart(lineChart(signalData[signalData['Section'] == label], name, xRange),
                        f"chart-{name}-{label}")

with energyTab:

    energyColumns = availableEnergyColumns(selectedRuns)

    if not energyColumns:

        st.info('The selected runs report no energies. Runs written before metadata.json '
            'existed carry none; re-run the wrappers to record them.')

    else:

        settings, reference = st.columns(2)

        column = settings.selectbox('Energy quantity', energyColumns,
            help='"Energy SIM [kWh]" is the consumption of the optimized trajectory re-evaluated '
                 'with the simulation losses, i.e. what the train would actually use.')

        names = shortNames(selectedRuns)

        baselineName = reference.selectbox('Compared against', [run.name for run in selectedRuns],
            format_func=lambda name: names[name],
            help='Every other run is shown as a difference with respect to this one.')

        baseline = runsByName[baselineName]
        others = [run for run in selectedRuns if run.name != baselineName]

        energies = {run.name: energyPerSection(run, column) for run in selectedRuns}

        # sections of the sidebar selection, labeled by the reference run where possible
        labels = {}

        for run in [baseline] + others:

            for section in run.sections:

                if sectionLabel(section) in selectedSections:

                    labels.setdefault(section.index, section.name)

        indices = sorted(idx for idx in labels if any(idx in energies[run.name] for run in selectedRuns))

        if not indices:

            st.info('None of the selected sections has an energy for this quantity.')

        else:

            # totals are only comparable over the sections that every run has in common
            common = [idx for idx in indices if all(idx in energies[run.name] for run in selectedRuns)]
            totals = {run.name: sum(energies[run.name][idx] for idx in common) for run in selectedRuns}

            def deltaColumns(run):
                "Header names of the difference columns, kept short when there is only one."

                suffix = '' if len(others) == 1 else f" {names[run.name]}"

                return f"Δ{suffix} [kWh]", f"Δ{suffix} [%]"

            ### totals, at a glance

            if not common:

                st.warning(f"The selected runs have no section with a \"{column}\" in common, "
                    'so their totals cannot be compared. Pick another energy quantity or '
                    'deselect the runs that do not report it.')

            else:

                st.subheader(f"Total over {len(common)} section(s)")

                cards = st.columns(len(selectedRuns))

                cards[0].metric(names[baselineName], f"{totals[baselineName]:.2f} kWh",
                    help='Reference run')

                for card, run in zip(cards[1:], others):

                    difference = totals[run.name] - totals[baselineName]
                    percent = relativeDifference(totals[run.name], totals[baselineName])

                    card.metric(names[run.name], f"{totals[run.name]:.2f} kWh",
                        delta=f"{difference:+.2f} kWh" + (f"  ({percent:+.1f} %)" if percent is not None else ''),
                        delta_color='inverse')  # spending more energy is not an improvement

                if len(common) < len(indices):

                    skipped = [f"{idx}: {labels[idx]}" for idx in indices if idx not in common]
                    st.caption('Excluded from the totals, as not every run has them: ' + ', '.join(skipped))

            ### per section

            st.subheader('Per section')

            rows = []

            for idx in indices + (['total'] if common else []):

                isTotal = idx == 'total'

                if isTotal:

                    row = {'Section': f"Total ({len(common)} sections)"}
                    valueOf = lambda run: totals[run.name]

                else:

                    row = {'Section': f"{idx}: {labels[idx]}"}
                    valueOf = lambda run: energies[run.name].get(idx)

                base = valueOf(baseline)
                row[f"{names[baselineName]} [kWh]"] = base

                for run in others:

                    value = valueOf(run)
                    deltaColumn, percentColumn = deltaColumns(run)

                    row[f"{names[run.name]} [kWh]"] = value
                    row[deltaColumn] = value - base if isValue(value) and isValue(base) else None
                    row[percentColumn] = relativeDifference(value, base)

                rows.append(row)

            energy = pd.DataFrame(rows)

            st.dataframe(energy, width='stretch', hide_index=True, column_config={
                name: st.column_config.NumberColumn(format='%.2f' if '[kWh]' in name else '%+.1f')
                for name in energy.columns if name != 'Section'})

            ### differences per section

            if others:

                differences = []

                for run in others:

                    for idx in indices:

                        value, base = energies[run.name].get(idx), energies[baseline.name].get(idx)

                        if not (isValue(value) and isValue(base)):

                            continue

                        differences.append({
                            'Run': names[run.name],
                            'Section': f"{idx}: {labels[idx]}",
                            'Δ [kWh]': value - base,
                            'Δ [%]': relativeDifference(value, base),
                        })

                if differences:

                    chart = alt.Chart(pd.DataFrame(differences)).mark_bar().encode(
                        x=alt.X('Section:N', title=None),
                        y=alt.Y('Δ [kWh]:Q', title=f"Δ {column} vs {names[baselineName]}"),
                        color=alt.Color('Run:N', title=None, legend=alt.Legend(orient='top', columns=1)),
                        xOffset='Run:N',
                        tooltip=['Run:N', 'Section:N', alt.Tooltip('Δ [kWh]:Q', format='+.2f'),
                            alt.Tooltip('Δ [%]:Q', format='+.1f')],
                    ).properties(height=CHART_HEIGHT)

                    st.altair_chart(chart, width='stretch')

with parametersTab:

    withMetadata = [run for run in selectedRuns if not run.isLegacy]

    if not withMetadata:

        st.info('None of the selected runs has a metadata.json. '
            'Re-run the wrappers to record the parameters of a run.')

    else:

        columns = {run.name: runParameters(run) for run in withMetadata}

        # rendered as text: a column mixes ids, numbers and flags, which has no common type
        parameters = pd.DataFrame(columns).astype(str).replace('None', '')
        parameters.index.name = 'Parameter'

        onlyDifferences = st.checkbox('Only parameters that differ', value=len(withMetadata) > 1)

        if onlyDifferences and len(withMetadata) > 1:

            parameters = parameters[parameters.nunique(axis=1) > 1]

        if parameters.empty:

            st.success('The selected runs used identical parameters.')

        else:

            st.dataframe(parameters, width='stretch')

        for run in withMetadata:

            with st.expander(f"Sections of {run.name}"):

                sectionRows = []

                for entry in run.metadata.get('sections', []):

                    row = {k: v for k, v in entry.items() if not isinstance(v, (dict, list))}
                    row.update({f"stats.{k}": v for k, v in (entry.get('stats') or {}).items()})
                    sectionRows.append(row)

                if sectionRows:

                    st.dataframe(pd.DataFrame(sectionRows), width='stretch', hide_index=True)

                else:

                    st.caption('No section details recorded.')

        if len(withMetadata) < len(selectedRuns):

            st.caption('Runs without metadata.json are not listed above: '
                + ', '.join(run.name for run in selectedRuns if run.isLegacy))
