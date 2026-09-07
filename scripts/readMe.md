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

- **`runMetadata.py`**
  Creates the timestamped output directory of a run and records its parameters in `metadata.json`.

Both wrappers read their inputs (data directory and train/track/journey ids) from
`config.json` in this directory. That file is not tracked by git, so local data paths stay private. To get started, copy the template and adapt it:

```bash
cd scripts/multiStopWrappers
cp config.example.json config.json
```

The keys `trainDirectory`, `trackDirectory` and `journeyDirectory` are optional and default to `directory`.
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