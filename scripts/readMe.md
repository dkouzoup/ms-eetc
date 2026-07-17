# Scripts Overview

This directory contains analysis tools, multi-stop processing wrappers, simulation launchers, and utilities for importing and comparing Swisstopo track data.

## Directory Structure

```text
scripts/
├── analyzer/
│   ├── analyzeEtcsBrakingCurve.py
│   └── analyzeNightData.py
├── multiStopWrappers/
│   ├── estimatorMultiStop.py
│   └── ocpMultiStop.py
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