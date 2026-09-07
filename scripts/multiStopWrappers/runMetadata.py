import hashlib
import json
import shutil
import subprocess
import sys

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

METADATA_FILE = 'metadata.json'

TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S'


def createRunDirectory(parent, timestamp=None):

    """
    Create a new timestamped directory for the results of a single run.

    Runs are named after the moment they were computed and nothing else: what a run
    actually is, is written in its metadata.json. Results of different runs never
    overwrite each other, so that they can be compared side by side later on.

    Parameters
    ----------
    parent : path-like
        Directory holding all runs of a given kind, e.g. "<data>/ocp".
    timestamp : datetime, optional
        Defaults to the current local time.
    """

    timestamp = datetime.now() if timestamp is None else timestamp

    name = timestamp.strftime(TIMESTAMP_FORMAT)

    runDirectory = Path(parent) / name

    for attempt in range(1, 100):  # a second is enough to tell runs apart, but do not rely on it

        try:

            runDirectory.mkdir(parents=True, exist_ok=False)

            return runDirectory

        except FileExistsError:

            runDirectory = Path(parent) / f"{name}-{attempt}"

    raise RuntimeError(f"Could not create a run directory in '{parent}'.")


def hashInputs(inputs):
    "Fingerprint of everything that determines the results of a run."

    serialized = json.dumps(toJSONSafe(inputs), sort_keys=True)

    return hashlib.sha256(serialized.encode()).hexdigest()[:12]


def findRunWithSameInputs(parent, inputsHash):
    "Most recent existing run computed from identical inputs, if there is one."

    parent = Path(parent)

    if not parent.is_dir():

        return None

    matches = [directory for directory in sorted(parent.iterdir()) if directory.is_dir()
        and (readMetadata(directory) or {}).get('inputsHash') == inputsHash]

    return matches[-1] if matches else None


def resolveDuplicateRun(existing):

    """
    Ask what to do about an existing run that used exactly the same inputs.

    Returns 'replace' to compute the run again and drop the old results once the new
    ones are in place, or 'keep' to leave the old run alone and add a new one.
    """

    print(f"\nA run with identical inputs already exists:\n  {existing}")

    if not sys.stdin.isatty():

        print("Not running interactively, keeping both.\n")

        return 'keep'

    while True:

        answer = input("[r]eplace it, [k]eep both, or [a]bort? ").strip().lower()

        if answer in ('r', 'replace'):

            return 'replace'

        if answer in ('k', 'keep'):

            return 'keep'

        if answer in ('a', 'abort'):

            sys.exit("Aborted, nothing was computed.")


def removeRun(runDirectory):
    "Delete the results of a superseded run."

    shutil.rmtree(runDirectory)

    print(f"Removed superseded run {runDirectory}")


def getGitCommit():
    "Return the current commit of the repository, or None if it cannot be determined."

    try:

        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
            capture_output=True, text=True, check=True)

    except (subprocess.CalledProcessError, FileNotFoundError):

        return None

    return result.stdout.strip()


def toJSONSafe(value):
    "Recursively convert numpy/pandas/pathlib objects into plain JSON types."

    if isinstance(value, dict):

        return {str(k): toJSONSafe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):

        return [toJSONSafe(v) for v in value]

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, np.ndarray):

        return toJSONSafe(value.tolist())

    if isinstance(value, (np.integer, np.floating, np.bool_)):

        value = value.item()

    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):

        return None

    if value is None or isinstance(value, (bool, int, float, str)):

        return value

    if pd.isna(value) is True:

        return None

    return str(value)  # last resort, e.g. lambdas and casadi functions


def writeMetadata(runDirectory, metadata):
    "Serialize the parameters of a run next to its results."

    path = Path(runDirectory) / METADATA_FILE

    with open(path, 'w') as f:

        json.dump(toJSONSafe(metadata), f, indent=4)

    return path


def readMetadata(runDirectory):
    "Read back the parameters of a run, or None if the run has no metadata file."

    path = Path(runDirectory) / METADATA_FILE

    if not path.is_file():

        return None

    with open(path) as f:

        return json.load(f)


def describeTrain(train):
    "Collect the train attributes that influence the resulting trajectories."

    # forceMinPn is where the pneumatic brake ends up, whether it was disabled or not
    keys = ['mass', 'rho', 'length', 'velocityMax', 'powerMax', 'powerMin',
        'forceMax', 'forceMin', 'forceMinPn', 'accMax', 'accMin', 'etaTraction', 'etaRgBrake']

    return {key: getattr(train, key) for key in keys if hasattr(train, key)}
