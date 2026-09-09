"""
Discovery and loading of multi-stop wrapper results.

A "run" is one directory written by ocpMultiStop.py or estimatorMultiStop.py. It
contains one pickled trajectory per journey section and a metadata.json holding every
parameter that was used together with the resulting energies.
"""

import json

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

METADATA_FILE = 'metadata.json'

TIMESTAMP_FORMAT = '%Y%m%d-%H%M%S'

RUN_KINDS = ('ocp', 'estimator')

MAX_SEARCH_DEPTH = 4  # how deep below the search root run directories are looked for


@dataclass
class Section:
    "One journey section of a run, together with its offsets on the track."

    index: int
    name: str
    path: Path = None  # pickled trajectory, None if the solver did not converge
    positionStart: float = None
    positionEnd: float = None
    timeOffset: float = 0.0
    absolutePositions: bool = False  # estimator results are already given in track coordinates
    withEtcsBrakingCurves: bool = False
    direction: str = None  # 'forward' or 'reverse' for a round trip, None for a one-way run
    distanceOffset: float = None  # distance covered before this section, along the whole trip
    speedLimits: dict = None  # step function {'Position [m]': [...], 'Speed limit [m/s]': [...]}
    info: dict = field(default_factory=dict)

    @property
    def hasTrajectory(self):

        return self.path is not None and self.path.is_file()


@dataclass
class Run:
    "One directory of results written by a multi-stop wrapper."

    path: Path
    kind: str
    metadata: dict = None
    sections: list = field(default_factory=list)

    @property
    def isLegacy(self):
        "True for result directories written before metadata.json was introduced."

        return self.metadata is None

    @property
    def isComplete(self):
        "False while a run is still going, or if it stopped before its last section."

        return bool((self.metadata or {}).get('complete', True))


    @property
    def label(self):

        return (self.metadata or {}).get('label') or self.path.name

    @property
    def timestamp(self):

        try:

            return datetime.strptime(self.path.name.split('_')[0], TIMESTAMP_FORMAT)

        except ValueError:

            return None

    @property
    def name(self):
        "Short unique-enough name shown in the sidebar and in plot legends."

        if self.path.name == self.kind:

            return f"{self.path.parent.name}/{self.kind} (legacy)"  # results written into "ocp" itself

        dataset = self.path.parent.parent.name  # e.g. "S31" in "S31/ocp/<run>"

        return f"{dataset}/{self.kind}/{self.path.name}"

def findRuns(root, kinds=RUN_KINDS, includeLegacy=True):

    """
    Collect all runs found below a search root.

    The root can be a single run directory, a dataset directory such as
    "innocheque/S31", or a parent of many datasets - all of them are scanned.
    """

    root = Path(root).expanduser()

    if not root.is_dir():

        return []

    runs = []
    seen = set()

    for kind in kinds:

        for parent in findKindDirectories(root, kind):

            candidates = sorted(d for d in parent.iterdir() if d.is_dir())

            for candidate in candidates:

                run = readRun(candidate, kind)

                if run is not None and run.path not in seen:

                    runs.append(run)
                    seen.add(run.path)

            if includeLegacy:

                run = readRun(parent, kind)  # results written directly into "ocp"/"estimator"

                if run is not None and run.path not in seen:

                    runs.append(run)
                    seen.add(run.path)

    runs.sort(key=lambda r: (r.name))

    return runs


def findKindDirectories(root, kind):
    "Locate the 'ocp'/'estimator' directories below the search root."

    if root.name == kind:

        return [root]

    if root.parent.name == kind:

        return [root.parent]  # root is a single run directory

    directories = [root / kind] if (root / kind).is_dir() else []

    for depth in range(1, MAX_SEARCH_DEPTH):

        pattern = '/'.join(['*']*depth + [kind])
        directories += [d for d in root.glob(pattern) if d.is_dir()]

    return directories


def readRun(path, kind):
    "Build a Run from a directory, or return None if it holds no results."

    path = Path(path)

    metadata = readMetadata(path)

    sections = readSections(path, kind, metadata)

    if not sections:

        return None

    return Run(path=path, kind=kind, metadata=metadata, sections=sections)


def readMetadata(path):

    metadataPath = Path(path) / METADATA_FILE

    if not metadataPath.is_file():

        return None

    with open(metadataPath) as f:

        return json.load(f)


def readSections(path, kind, metadata):
    "Read the section list from metadata.json, falling back to globbing the directory."

    if metadata is not None:

        sections = []

        for entry in metadata.get('sections', []):

            file = entry.get('file')

            sections.append(Section(
                index=entry.get('index', len(sections)),
                name=entry.get('name') or f"section {entry.get('index', len(sections))}",
                path=(path / file) if file else None,
                positionStart=entry.get('positionStart [m]'),
                positionEnd=entry.get('positionEnd [m]'),
                timeOffset=entry.get('timeOffset [s]', 0.0) or 0.0,
                absolutePositions=kind == 'estimator',
                withEtcsBrakingCurves=bool((entry.get('solverOptions') or {}).get('withEtcsBrakingCurves')),
                direction=entry.get('direction'),
                distanceOffset=entry.get('distanceOffset [m]'),
                speedLimits=entry.get('speedLimits'),
                info=entry,
            ))

        return sections

    # legacy layout: positions and parameters of the run are unknown

    pattern = 'df_section_*.pkl' if kind == 'ocp' else '*_df_force_estimate.pkl'

    sections = []

    for idx, file in enumerate(sorted(path.glob(pattern))):

        name = file.stem.replace('df_', '').replace('_force_estimate', '').replace('_', ' ').strip()

        sections.append(Section(index=idx, name=name or file.stem, path=file,
            absolutePositions=kind == 'estimator'))

    return sections


def addSpeedLimits(df, section):

    """
    Name the limit the trajectory was constrained with for what it is.

    With ETCS braking curves enabled the solver overwrites the speed limits of its
    discretization points with the curves, so the column holds those rather than the
    speed limits. The limits of the track come from Section.speedLimits instead.
    """

    if section.withEtcsBrakingCurves and 'Speed limit [m/s]' in df.columns:

        df['ETCS braking curve [m/s]'] = df.pop('Speed limit [m/s]')

    return df


def loadSection(section):

    """
    Load the trajectory of one section as a dataframe.

    Time and position are returned both relative to the start of the section and,
    where the offsets are known, absolute on the track and within the journey.
    """

    df = pd.read_pickle(section.path)

    df = df.copy()
    df['Time [s]'] = df.index.values

    # ocp trajectories restart at zero in every section, estimator ones are already on the track
    offset = 0.0 if section.absolutePositions or section.positionStart is None else section.positionStart

    df['Position (abs) [m]'] = df['Position [m]'] + offset

    df['Time (abs) [s]'] = df['Time [s]'] + (section.timeOffset or 0.0)

    if section.distanceOffset is not None:

        # lays the sections of a trip out in a row, the return trip continuing the outbound one
        df['Distance traveled [m]'] = df['Position [m]'] + section.distanceOffset

    return addSpeedLimits(df, section)
