"""
How much energy does a second of running time buy, section by section?

Timetabling allocates the running time of a trip stop by stop, and the energy an
optimal driving strategy needs is a convex decreasing function of the time a section
is given. Where that function is steep, a few seconds more save a lot of energy; where
it is flat, seconds can be taken away almost for free. This script measures that
function: for every journey section of a trip it solves the time optimal problem once,
to get the shortest running time the train can do at all, and then the energy optimal
problem on a grid of running times around the scheduled one.

The settings of the sweep are not entered here. They are read off an existing ocp run,
so that the curves are directly comparable with the energies that run reported and the
two cannot drift apart: the baseline run fixes the loss models, the ETCS braking
curves, the tunnel handling, the discretization and the journeys.

Results go to a new timestamped directory under "sensitivity" in the input folder:
metadata.json with one entry per section, holding the scheduled and the minimum
running time and the sampled (time, energy) points. `reallocateTimes.py` turns those
into a new time allocation; nothing else is written, the trajectories themselves are
not kept.

Sections are independent, so they are swept in parallel worker processes.

Usage (paths and ids come from config.json, see config.example.json):

    python timeSensitivity.py                       # latest complete ocp run as baseline
    python timeSensitivity.py --run 20260918-080754
    python timeSensitivity.py --workers 4 --dry-run
    python timeSensitivity.py --resume 20260918-223840   # finish an interrupted sweep

Sections are written to metadata.json as they finish, so an interrupted sweep loses at
most the sections that were still being solved; --resume picks up the rest.
"""

import argparse
import json

from concurrent.futures import ProcessPoolExecutor, as_completed
from math import floor
from pathlib import Path

import casadi as ca

from runMetadata import METADATA_FILE, createRunDirectory, describeTrain, getGitCommit, \
    readMetadata, writeMetadata
from tunnelAgnostic import estimateEnergyOnTrack, removeTunnels, reportForceLimitViolations
from wrapperConfig import loadConfig

# running times to sample, as a fraction of the scheduled running time of the section.
# The range has to cover what a reallocation may want to do with a section, in both
# directions, and is denser around the scheduled time, where the curve is read off.
FRACTIONS = [0.90, 0.95, 1.00, 1.08, 1.20, 1.40]

# closest to the time optimum that the energy optimal problem is sampled. Exactly at the
# time optimum the feasible set collapses to a point and ipopt has nothing to work with.
MIN_MARGIN = 0.02

# The terminal time of a journey section is an upper bound on the arrival, and ocp.py keeps
# that bound in place whatever the objective is. Minimizing the arrival time under it answers
# "how early could this train arrive, if it still arrives on time", which is infeasible as
# soon as the section cannot be run in its scheduled time at all. The shortest running time
# the section can do is the question here, so the time optimal problem is given a bound wide
# enough never to bind.
TIME_OPTIMAL_BOUND = 4

# a worker holds a whole nlp and peaks around 1.5 GB on the longest sections, so the
# default stays well clear of the memory of a machine that is being worked on
WORKERS = 3


def latestCompleteRun(runsDirectory):

    "The most recent ocp run that ran to the end, by the timestamp it is named after."

    runs = sorted(p for p in Path(runsDirectory).iterdir()
                  if p.is_dir() and (p / METADATA_FILE).is_file())

    for run in reversed(runs):

        metadata = readMetadata(run)

        if metadata.get('type') == 'ocp' and metadata.get('complete'):

            return run

    raise FileNotFoundError(f"No complete ocp run in '{runsDirectory}' to take the settings from.")


def loadPowerLossFunction(path):
    "Load a serialized casadi power-loss function (train-specific, measured)."

    return ca.Function.deserialize(Path(path).read_text())


def buildSetup(inputs, direction, sectionIdx):

    """
    Train, track, journey, solver options and interval count of one journey section,
    exactly as ocpMultiStop.py builds them for the same section.
    """

    from mseetc.journey import Journey
    from mseetc.track import Track
    from mseetc.train import Train
    from mseetc.utils import get_power_loss_function

    directory = Path(inputs['directory'])

    train = Train(config={'id': inputs['trainId']}, pathJSON=inputs['trainDirectory'])

    if not inputs['train']['withPnBrake']:

        train.forceMinPn = 0  # the solver reads the flag off this, see Train.exportModel

    lossFunctions = {
        'perfect': lambda: get_power_loss_function(train, 'perfect'),
        'static': lambda: get_power_loss_function(train, 'static'),
        'dynamic': lambda: loadPowerLossFunction(directory / inputs['losses']['lossesFile']),
    }

    losses = {key: lossFunctions[inputs['losses'][key]]() for key in ('ocp', 'sim')}

    journey = Journey(config={'id': inputs['journeys'][direction]}, pathJSON=inputs['journeyDirectory'])
    journey.selectJourneySection(sectionIdx)

    track = Track(config={'id': inputs['trackId']}, pathJSON=inputs['trackDirectory'])

    if direction == 'reverse':

        track.reverse()

    if not inputs['tunnelsInOcp']:

        removeTunnels(track)  # before updateLimits, while the track is still complete

    track.updateTrainLengthDependentValues(train)
    track.updateLimits(positionStart=journey.positionStart, positionEnd=journey.positionEnd, unit='m')

    numOfIntervals = floor((journey.positionEnd - journey.positionStart)
                           * inputs['discretization']['intervalsPerMeter'])

    opts = dict(inputs['solver'], numIntervals=numOfIntervals)

    return train, track, journey, losses, opts


def realTrack(inputs, direction):

    """
    A freshly loaded, uncropped track with its tunnels intact, for the reporting step.

    forceEstimator crops the track it is given in place, so every estimate needs its own.
    """

    from mseetc.track import Track

    track = Track(config={'id': inputs['trackId']}, pathJSON=inputs['trackDirectory'])

    if direction == 'reverse':

        track.reverse()

    return track


def energyOfSolution(df, inputs, direction, train, journey, opts, losses):

    """
    The energy the trajectory costs under the reporting loss model, as ocpMultiStop.py
    accounts for it: through the real track when the ocp never saw the tunnels, through
    the loss model alone when it did.
    """

    from mseetc.utils import postProcessDataFrame

    train.powerLosses = losses['sim']

    try:

        if inputs['tunnelsInOcp']:

            dfSim = postProcessDataFrame(df, None, train, opts, integrateLosses=True)

            return dfSim['Energy [kWh]'].sum(), None

        dfEstimate, energyStats = estimateEnergyOnTrack(
            df, journey.positionStart, train, realTrack(inputs, direction), opts)

        overForce, overPower = reportForceLimitViolations(dfEstimate, train)

        return energyStats['Net energy used [kWh]'], {'force': overForce, 'power': overPower}

    finally:

        train.powerLosses = losses['ocp']


def sweepSection(arguments):

    """
    One journey section: its time optimum, and the energy optimum on a grid of running
    times around the scheduled one. Runs in a worker process of its own.
    """

    from mseetc.ocp import casadiSolver

    inputs, direction, sectionIdx, fractions = arguments

    train, track, journey, losses, opts = buildSetup(inputs, direction, sectionIdx)

    name = f"{direction} {sectionIdx}"
    scheduled = journey.terminalTime

    result = {
        'direction': direction,
        'sectionIdx': sectionIdx,
        'name': name,
        'journeyId': inputs['journeys'][direction],
        'positionStart [m]': journey.positionStart,
        'positionEnd [m]': journey.positionEnd,
        'numIntervals': opts['numIntervals'],
        'scheduled time [s]': scheduled,
        'minimum time [s]': None,
        'samples': [],
    }

    # time optimal solution: the shortest running time of this section, which is the
    # left edge of every energy curve and tells whether the timetable is feasible at all

    train.powerLosses = losses['ocp']

    journey.terminalTime = TIME_OPTIMAL_BOUND*scheduled

    timeSolver = casadiSolver(train, track, journey, dict(opts, energyOptimal=False))

    try:

        dfTime, statsTime = timeSolver.solve()

        result['time optimal'] = {'status': statsTime['Solver status'], 'time [s]': statsTime['Cost']}

        minimum = statsTime['Cost'] if dfTime is not None else None

    except Exception as error:

        # solve() integrates its own solution before handing it back, and that integration
        # can fail on a profile that is all acceleration and braking. The section still has
        # an energy curve, it just has no left edge to anchor it on.

        result['time optimal'] = {'status': repr(error), 'time [s]': None}

        minimum = None

    result['minimum time [s]'] = minimum

    del timeSolver

    # energy optimal solutions on the time grid, never closer to the time optimum than
    # MIN_MARGIN: exactly at it the feasible set collapses to a point

    floorTime = minimum*(1 + MIN_MARGIN) if minimum is not None else 0

    # the grid is laid out on the running time the section actually needs, which is the
    # scheduled one unless the timetable gives it less than it can run in
    base = max(scheduled, floorTime)

    times = sorted({round(max(fraction*base, floorTime), 3) for fraction in fractions}
                   | ({round(floorTime, 3)} if minimum is not None else set()))

    result['fractions'] = fractions
    result['grid base [s]'] = base

    solver = casadiSolver(train, track, journey, dict(opts, energyOptimal=True))

    for time in times:

        journey.terminalTime = time

        sample = {'time [s]': time}

        try:

            df, stats = solver.solve()

            sample.update({
                'status': stats['Solver status'],
                'IP iterations': stats['IP iterations'],
                'CPU time [s]': stats['CPU time [s]'],
            })

        except Exception as error:

            sample['status'] = repr(error)

            result['samples'].append(sample)

            print(f"  {name}: t = {time:7.1f} s -> {error!r}", flush=True)

            continue

        if df is not None:

            sample['Energy OCP [kWh]'] = df['Energy [kWh]'].sum()

            try:

                energy, violations = energyOfSolution(df, inputs, direction, train, journey, opts, losses)

                sample['Energy SIM [kWh]'] = energy
                sample['forceLimitViolations'] = violations

            except Exception as error:

                # the reporting step integrates the planned profile on the real track and can
                # fail on its own (a stiff profile at the short end of the grid). One sample
                # without a reported energy is a gap in the curve, not a lost section.

                sample['reporting error'] = repr(error)

                print(f"  {name}: t = {time:7.1f} s -> reporting failed, {error!r}", flush=True)

        result['samples'].append(sample)

        if 'Energy SIM [kWh]' in sample:

            print(f"  {name}: t = {time:7.1f} s -> {sample['Energy SIM [kWh]']:8.2f} kWh", flush=True)

        elif df is None:

            print(f"  {name}: t = {time:7.1f} s -> {stats['Solver status']}", flush=True)

    return result


def main():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default=None,
                        help="name of the ocp run to take the settings from "
                             "(default: the latest complete one)")
    parser.add_argument('--resume', default=None,
                        help="name of an interrupted sweep to continue, skipping the "
                             "sections it already holds")
    parser.add_argument('--replace', action='store_true',
                        help="with --resume and --sections: sweep those sections again even "
                             "though the sweep holds them, merging the new samples into the old")
    parser.add_argument('--fractions', nargs='*', type=float, default=FRACTIONS,
                        help="running times to sample, as fractions of the running time the "
                             f"section needs (default: {' '.join(str(f) for f in FRACTIONS)})")
    parser.add_argument('--workers', type=int, default=WORKERS,
                        help=f"worker processes to sweep sections in (default: {WORKERS})")
    parser.add_argument('--sections', nargs='*', default=None,
                        help="restrict to these sections, e.g. --sections forward:0 forward:2 reverse:13")
    parser.add_argument('--dry-run', action='store_true',
                        help="report what would be swept, solve nothing")
    args = parser.parse_args()

    config = loadConfig()

    directory = Path(config['directory'])

    resumed = readMetadata(directory / 'sensitivity' / args.resume) if args.resume else None

    if resumed is not None:

        assert args.run is None or args.run == resumed['baselineRun'], (
            f"Sweep {args.resume} was started from ocp run {resumed['baselineRun']}, not {args.run}")

        args.run = resumed['baselineRun']

    baseline = Path(directory / 'ocp' / args.run) if args.run else latestCompleteRun(directory / 'ocp')

    baselineMetadata = readMetadata(baseline)

    assert baselineMetadata is not None, f"No {METADATA_FILE} in '{baseline}'"

    inputs = baselineMetadata['inputs']

    sections = selectSections(inputs, args.sections)

    results = list(resumed['sections']) if resumed is not None else []

    previous = {}

    if results:

        if args.replace:

            assert args.sections, "--replace needs --sections: it says which ones to sweep again"

            previous = {(result['direction'], result['sectionIdx']): result for result in results
                        if (result['direction'], result['sectionIdx']) in set(sections)}

            results = [result for result in results
                       if (result['direction'], result['sectionIdx']) not in previous]

        swept = {(result['direction'], result['sectionIdx']) for result in results}

        sections = [section for section in sections if section not in swept]

        print(f"Resuming sweep {args.resume}: {len(results)} sections kept"
              + (f", {len(previous)} to be swept again" if previous else ""))

    print(f"Baseline run {baseline.name}: losses {inputs['losses']['ocp']}/{inputs['losses']['sim']}, "
          f"ETCS {inputs['solver']['withEtcsBrakingCurves']}, tunnels in ocp {inputs['tunnelsInOcp']}")
    print(f"Sweeping {len(sections)} sections at {len(FRACTIONS)} running times each, "
          f"plus one time optimal solve, in {args.workers} workers")

    if args.dry_run:

        for direction, sectionIdx in sections:

            print(f"  {direction} {sectionIdx}")

        return

    runDirectory = (directory / 'sensitivity' / args.resume) if resumed is not None \
        else createRunDirectory(directory / 'sensitivity')
    print(f"Results of this run are saved in {runDirectory}")

    def saveMetadata(complete):

        writeMetadata(runDirectory, {
            "type": "sensitivity",
            "script": Path(__file__).name,
            "complete": complete,
            "gitCommit": getGitCommit(),
            "baselineRun": baseline.name,
            "inputs": inputs,
            "fractions": args.fractions,  # the default of this call, sections carry their own
            "minMargin": MIN_MARGIN,
            "timeOptimalBound": TIME_OPTIMAL_BOUND,
            "sections": results,
        })

    saveMetadata(complete=False)

    # longest section first: they dominate the sweep, and starting them last would leave
    # the workers waiting for a single one at the end
    tasks = [(inputs, direction, sectionIdx, args.fractions) for direction, sectionIdx in
             sorted(sections, key=lambda s: -sectionLength(inputs, *s))]

    total = len(results) + len(tasks)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:

        futures = [pool.submit(sweepSection, task) for task in tasks]

        for future in as_completed(futures):

            try:

                result = future.result()

            except Exception as error:  # a worker that dies must not take the sweep with it

                print(f"Section failed: {error!r}", flush=True)

                continue

            results.append(mergeSamples(previous.get((result['direction'], result['sectionIdx'])),
                                        result))

            minimum = result['minimum time [s]']

            print("[{}/{}] {}: {}".format(len(results), total, result['name'],
                                          f"minimum {minimum:.1f} s" if minimum is not None
                                          else "no time optimum"), flush=True)

            saveMetadata(complete=False)

    results.sort(key=lambda r: (r['direction'] != 'forward', r['sectionIdx']))

    saveMetadata(complete=True)

    print(f"Wrote {len(results)} sections and {METADATA_FILE} to {runDirectory}")


def mergeSamples(old, new):

    """
    Fold an earlier sweep of the same section into a new one.

    Refining a section means sampling running times the first sweep did not cover, not
    throwing away the ones it did, so the samples are the union of both, one per running
    time. A time optimum the new sweep could not find is taken from the old one.
    """

    if old is None:

        return new

    samples = {round(sample['time [s]'], 1): sample for sample in old['samples']}
    samples.update({round(sample['time [s]'], 1): sample for sample in new['samples']})

    new['samples'] = [samples[time] for time in sorted(samples)]

    if new['minimum time [s]'] is None:

        new['minimum time [s]'] = old['minimum time [s]']
        new['time optimal'] = old['time optimal']

    new['fractions'] = sorted(set(old.get('fractions', [])) | set(new['fractions']))

    return new


def sectionLength(inputs, direction, sectionIdx):

    "Length of one journey section, to sweep the expensive ones first."

    from mseetc.journey import Journey

    journey = Journey(config={'id': inputs['journeys'][direction]}, pathJSON=inputs['journeyDirectory'],
                      sectionIdx=sectionIdx)

    return journey.positionEnd - journey.positionStart


def selectSections(inputs, selection):

    "The (direction, sectionIdx) pairs to sweep, all of the trip unless restricted."

    from mseetc.journey import Journey

    directions = ['forward', 'reverse'] if inputs['roundTrip'] else ['forward']

    counts = {direction: len(Journey(config={'id': inputs['journeys'][direction]},
                                     pathJSON=inputs['journeyDirectory']).journeySectionBounds)
              for direction in directions}

    if selection is None:

        return [(direction, idx) for direction in directions for idx in range(counts[direction])]

    sections = []

    for part in selection:

        direction, _, index = part.partition(':')

        assert direction in directions, f"Unknown direction '{direction}', expected one of {directions}"
        assert index.isdigit() and int(index) < counts[direction], \
            f"'{part}' is not a section of journey '{inputs['journeys'][direction]}'"

        sections.append((direction, int(index)))

    return sections


if __name__ == '__main__':

    main()
