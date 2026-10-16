# Scripts Overview

This directory contains analysis tools, multi-stop processing wrappers, simulation launchers, and utilities for importing and comparing Swisstopo track data.

## Directory Structure

```text
scripts/
├── analyzer/
│   ├── analyzeEtcsBrakingCurve.py
│   └── analyzeNightData.py
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── runData.py
│   └── signals.py
├── multiStopWrappers/
│   ├── config.example.json
│   ├── estimatorMultiStop.py
│   ├── journeyFromTimetable.py
│   ├── ocpMultiStop.py
│   ├── runMetadata.py
│   └── wrapperConfig.py
├── simulations/
│   ├── simLauncherETCS.py
│   └── simLauncherTimingPoints.py
└── swisstopo/
    ├── compareSbbAndSwisstopoTracks.py
    └── importTrackFromSwisstopo.py
```

## Analyzer

- **`analyzeEtcsBrakingCurve.py`**
  Visualizes ETCS braking curves for a single speed reduction.

- **`analyzeNightData.py`**
  Validates the full measurement-processing workflow, including position alignment, force and energy estimation, OCP optimization, and comparison with recorded train data.

## Multi-Stop Wrappers

- **`estimatorMultiStop.py`**
  Processes multiple measured journey sections from CSV files and saves force, energy, and summary results.

- **`ocpMultiStop.py`**
  Computes and stores energy-optimal trajectories for all sections of a multi-stop journey.

- **`journeyFromTimetable.py`**
  Builds the journey files of a line, one per direction, from an operator timetable extract.

- **`runMetadata.py`**
  Creates the timestamped output directory of a run and records its parameters in `metadata.json`.

Both wrappers read their inputs (data directory and train/track/journey ids) from
`config.json` in this directory. That file is not tracked by git, so local data paths stay private. To get started, copy the template and adapt it:

```bash
cd scripts/multiStopWrappers
cp config.example.json config.json
```

The keys `trainDirectory`, `trackDirectory` and `journeyDirectory` are optional and default to `directory`.

### Journeys from a timetable

`journeyFromTimetable.py` writes the journey files of a line straight from an operator
timetable extract, so that the timing points are the operator's own times rather than
something read off a public timetable by hand:

```bash
cd scripts/multiStopWrappers
python journeyFromTimetable.py            # one journey file per direction
python journeyFromTimetable.py --dry-run  # report what it would write
```

It reads the extract named by `timetable` in `config.json` (relative to `directory`) and
the track named by `trackId`, matches the commercial stops of each timetable against the
`station names` of the track, and writes one journey file per direction, named after the
timetable it comes from (`S4:RW-SA` becomes `CH_Journeys_S4_RW_SA`). A timetable that runs
against the station order of the track - the return service - is written with its positions
mirrored on the track length, ready for use with a reversed track.

Each stop becomes an arrival and a departure stopping point, so there is one journey section
per stop-to-stop leg. Passing points are dropped: the track does not say where they are.

`--basis` chooses between the two sets of times an extract carries:

| basis | |
|---|---|
| `operational` (default) | the second-precision times the operator plans the train to run to |
| `commercial` | the published times, rounded to whole minutes |

Prefer `operational`. Published times are rounded to the minute, which hides the dwell at
every stop whose arrival and departure fall in the same minute and leaves every section
running time uncertain by up to a minute - uncertain in both directions, so some sections
come out too generous and others too short for the train to actually run, which then has to
be papered over with running time supplements. Operational times need none of that, but they
are also markedly tighter: on the S4 they give 6594 s of running time over the trip where
the commercial ones give 7680 s, the difference being the dwell they no longer hide.
### Output layout

Every run writes into its own timestamped directory, so results computed with different
parameters no longer overwrite each other:

```text
<directory>/
├── ocp/
│   ├── 20260907-100152/
│   │   ├── metadata.json      # parameters of the run and energy per section
│   │   ├── df_section_0.pkl   # optimized trajectory per section
│   │   └── ...
│   └── 20260907-100456/
│       └── ...
└── estimator/
    └── 20260907-095402/
        └── ...
```

A run is named after the moment it was computed and nothing else. What it actually is, is
written in its `metadata.json`, which the dashboard shows and filters on.

`metadata.json` is the single record of a run. It holds the train, track and journey ids,
the loss functions, the complete solver options and, per journey section, its position and
time offsets, the speed limits of the track, the solver stats and the resulting energy. The
offsets are what allows the dashboard to place the sections of a journey on the track and to
overlay measured recordings with optimized trajectories.

Directories written before this layout existed (results lying directly in `ocp/` or
`estimator/`) are still readable by the dashboard, but without parameters or track offsets.

### Round trips

With `roundTrip = True` the wrapper computes the return trip in the same run, so that the
energy of a full round trip can be read off directly. The track is reversed with
`Track.reverse()` and each direction is computed from its own journey:

```json
"journeys": {"forward": "CH_Journeys_S4_RW_SA", "reverse": "CH_Journeys_S4_SA_RW"}
```

A return service is a timetable of its own, not the mirror image of the outbound one - the
S4 takes 6594 s of running time one way and 6612 s the other, distributed differently over
the legs - so there is no way to derive one direction from the other. `journeyFromTimetable.py`
writes both, the return one with its positions already mirrored on the track length, ready
for the reversed track. Each section records the `journeyId` it was computed from.

Sections are numbered along the trip and carry the direction they belong to:

```text
index  name        direction  distanceOffset [m]
    0  forward 0   forward                   0.0
    3  forward 3   forward               11828.9
    4  reverse 0   reverse               17220.7
    7  reverse 3   reverse               30532.3
```

`distanceOffset [m]` is the distance covered before a section starts, which lays the whole
trip out on one continuous axis. The return trip runs back over the same line, so `reverse 0`
covers the same stretch of it as `forward 3` - at its own timetable, not the outbound one
read backwards.

### Repeated runs

Everything that determines the results of a run - the ids and directories it reads, the loss
functions, the discretization and the solver options - is collected in the `inputs` block of
`metadata.json` and fingerprinted into `inputsHash`. Before computing anything, a wrapper
looks for an existing run with the same fingerprint and asks what to do:

```text
A run with identical inputs already exists:
  ../../innocheque/S31/ocp/20260907-100152
[r]eplace it, [k]eep both, or [a]bort?
```

`replace` drops the old run only once the new results are safely written, so an interrupted
or failing run never destroys what was there. Without a terminal (a run started from a
script or an editor), both runs are kept and a note is printed.

## Dashboard

- **`app.py`**
  Streamlit dashboard to compare runs: overlaid speed, force, power and energy profiles,
  the difference in energy consumption per section and in total, and a table of the
  parameters in which the selected runs differ.
- **`runData.py`**
  Discovers run directories below a search root and loads their trajectories.
- **`signals.py`**
  Signals that can be plotted and the raw columns each of them needs.

The dashboard reads only the files written by the wrappers, so it needs neither casadi nor
`mseetc`. Give it its own virtual environment, since streamlit requires a newer numpy than
the solver environment:

```bash
python3 -m venv .env_dashboard
.env_dashboard/bin/pip install -r scripts/dashboard/requirements.txt
.env_dashboard/bin/activate
streamlit run scripts/dashboard/app.py
```

Point the "Data directory" field in the sidebar at any directory holding results (e.g.
`innocheque`, or a single dataset such as `innocheque/S31`); its subdirectories are scanned
for `ocp` and `estimator` run folders. Select any number of runs to overlay them, and use
the track-wide position axis to see a whole journey, or a per-section axis to compare
sections against each other.

For a round trip, the `Distance traveled [km]` axis lays the outbound and return legs out in
a row and marks the turnaround with a dashed vertical line, rather than folding the two
directions onto the same positions.

Dragging over any chart zooms every chart to that range, so the force and power below a
speed profile show the same stretch of the trip; "Reset zoom" restores the full extent.

Since runs are named only by their timestamp, the "Filters" section of the sidebar narrows
the list down by kind and by any parameter that is not the same across all runs - selecting
`solver.withEtcsBrakingCurves` and the value `True`, for instance, leaves only the runs that
used the braking curves.

### Speed limits

When ETCS braking curves are enabled, the solver replaces the speed limits of its
discretization points with the braking curves, so the `Speed limit [m/s]` column of a
trajectory holds the curves rather than the limits. `ocpMultiStop.py` therefore records the
speed limits of the track separately, per journey section, in `metadata.json`:

```json
"speedLimits": {
    "Position [m]": [0.0, 275.24, 347.10, ...],
    "Speed limit [m/s]": [11.11, 13.89, 22.22, ...]
}
```

These are the breakpoints of the track itself, as the solver saw them: adapted to the train
length (a speed increase only applies once the whole train has passed it) and cropped to the
section, with positions relative to its start. Keeping the breakpoints rather than sampling
them onto the discretization points means the steps are drawn exactly where the limit
changes. The dashboard offers them as `Speed limit [km/h]` and the braking curves as
`ETCS braking curve [km/h]`, both drawn as dashed steps over the velocity.

### Tunnels

Both wrappers record the stretches a section runs in a tunnel, per journey section, in
`metadata.json`:

```json
"tunnels": [
    {"start [m]": 899.0, "end [m]": 9499.0, "cross section [m^2]": 26.0}
]
```

The cross sections of a track are a step function, infinite on open track, so a stretch of
tunnel is a step with a finite cross section. They are taken off the track as the run
prepared it - cropped to the section, reversed for a return trip and adapted to the train
length - so the positions are in the same frame as the trajectory, and a run computed with
`tunnelsInOcp = False` records none.

"Shade tunnels" in the sidebar draws them as light gray bands behind every chart. The band
ends are mapped onto whichever axis is plotted through the trajectory itself, so on a time
axis they become the stretches of time the train spends inside; the tooltip gives the length
and the cross section. Runs computed before the tunnels were recorded carry none and are
drawn without shading.

## Simulations

- **`simLauncherETCS.py`**
  Compares optimized trajectories with and without ETCS braking-curve constraints.

- **`simLauncherTimingPoints.py`**
  Runs an energy-optimal simulation with timing-point constraints and plots the resulting time and velocity profiles.

## Swisstopo

- **`importTrackFromSwisstopo.py`**
  Converts Swisstopo CSV data into a TTOBench track JSON file.

- **`compareSbbAndSwisstopoTracks.py`**
  Compares SBB and Swisstopo track data, optimized trajectories, and resulting energy costs.