"""
Spend the running time of a trip where it saves the most energy.

`timeSensitivity.py` measures, section by section, what a second of running time is
worth: the energy an optimal driving strategy needs as a function of the time the
section is given. Those curves are convex and fall steeply on some sections and hardly
at all on others, which means the timetable's own allocation is almost never the one
that spends a fixed amount of running time best.

This script reads the curves and reallocates. The trip keeps the duration it has - the
train leaves and arrives when the timetable says - and every section keeps a reserve
over the shortest running time it could physically do. Within that, time is moved from
sections where it buys little to sections where it buys a lot, by repeatedly handing
the next step of time to whichever section saves the most energy with it, which is the
optimal allocation for convex curves.

Two variants are written, differing in where the running time comes from:

  running  the running times are redistributed among themselves and every dwell is
           left exactly as the timetable has it
  dwell    the dwells are cut to the minimum waiting time the timetable gives for the
           stop as well, and everything that frees is running time to be allocated

Both are written as journey files, one per direction, next to the ones they are
derived from. Nothing else is written: what the reallocation did is printed, and the
journey files carry their provenance in their description.

Usage (paths and ids come from config.json, see config.example.json):

    python reallocateTimes.py                      # latest complete sweep
    python reallocateTimes.py --run 20260923-091430 --margin 0.05
    python reallocateTimes.py --variants running --dry-run
"""

import argparse
import json

from heapq import heapify, heappop, heappush
from math import ceil, floor
from pathlib import Path

from runMetadata import METADATA_FILE, readMetadata
from wrapperConfig import loadConfig

# the operator plans this line on a six second grid, in running times and dwells alike,
# so a reallocation that lands anywhere else would not be a timetable anybody could run
STEP = 6

MARGIN = 0.05  # reserve every section keeps over its shortest possible running time

# most of what the timetable holds above the minimum waiting time of a stop is not slack:
# the long dwells are connection waits. The 'dwell' variant may take at most this share of
# a dwell, so that harvesting running time cannot quietly dismantle the connections.
MAX_DWELL_CUT = 0.25

SUFFIXES = {'running': '_Realloc', 'dwell': '_ReallocDwell'}

LIBRARY_VERSION = 'TTOBench v1.5'

STOP = 'commercialStop'


def latestCompleteRun(runsDirectory):

    "The most recent sensitivity sweep that ran to the end."

    runs = sorted(p for p in Path(runsDirectory).iterdir()
                  if p.is_dir() and (p / METADATA_FILE).is_file())

    for run in reversed(runs):

        metadata = readMetadata(run)

        if metadata.get('type') == 'sensitivity' and metadata.get('complete'):

            return run

    raise FileNotFoundError(f"No complete sensitivity sweep in '{runsDirectory}'.")


def roundUp(value):

    return int(ceil(value/STEP - 1e-9)*STEP)


def roundDown(value):

    return int(floor(value/STEP + 1e-9)*STEP)


def lowerEnvelope(samples):

    """
    The convex lower envelope of the sampled (time, energy) points.

    The energy of a section is a convex function of its running time, so the envelope is
    the most that can be read out of a handful of samples without inventing curvature
    between them. It also absorbs the odd point that a solver tolerance put a little too
    high, which matters here: the allocation is driven by differences between samples,
    and a single point out of line would otherwise turn into a bogus marginal value.
    """

    points = sorted((sample['time [s]'], sample['Energy SIM [kWh]']) for sample in samples)

    hull = []

    for point in points:

        while len(hull) >= 2 and slope(hull[-2], hull[-1]) >= slope(hull[-1], point):

            hull.pop()

        hull.append(point)

    return hull


def slope(left, right):

    return (right[1] - left[1])/(right[0] - left[0])


def energyAt(hull, time):

    "Energy of a section at a running time, linearly interpolated on its envelope."

    if time <= hull[0][0]:

        return hull[0][1]

    for left, right in zip(hull, hull[1:]):

        if time <= right[0]:

            return left[1] + slope(left, right)*(time - left[0])

    return hull[-1][1]


def buildLegs(sections, margin):

    """
    One entry per journey section: its envelope and the window the allocation may move
    it in. Sections the sweep could not solve have no curve and are frozen at the time
    the timetable gives them.
    """

    legs = []

    for section in sorted(sections, key=lambda s: s['sectionIdx']):

        usable = [sample for sample in section['samples'] if 'Energy SIM [kWh]' in sample]

        scheduled = int(section['scheduled time [s]'])
        minimum = section['minimum time [s]']

        leg = {
            'name': section['name'],
            'sectionIdx': section['sectionIdx'],
            'scheduled': scheduled,
            'minimum': minimum,
            'frozen': len(usable) < 2,
        }

        if leg['frozen']:

            leg.update(hull=None, lower=scheduled, upper=scheduled)

        else:

            hull = lowerEnvelope(usable)

            # never below the reserve, and never outside the sampled range: beyond it the
            # envelope would be extrapolated, and an extrapolated marginal value is a guess
            lower = max(roundUp(hull[0][0]), roundUp(minimum*(1 + margin)) if minimum else 0)
            upper = max(roundDown(hull[-1][0]), lower)

            leg.update(hull=hull, lower=lower, upper=upper)

        legs.append(leg)

    return legs


def allocate(legs, budget):

    """
    Hand out the running time budget in steps of STEP seconds, each one to the section
    that saves the most energy with it. For convex curves that greedy is exact: after
    every step the marginal values stay ordered, so no reshuffling can do better.
    """

    times = [leg['lower'] for leg in legs]

    remaining = budget - sum(times)

    assert remaining % STEP == 0, f"A budget of {budget} s is not a whole number of {STEP} s steps"

    if remaining < 0:

        raise ValueError(
            f"The trip has {budget} s of running time to give, but its sections need "
            f"{sum(times)} s to keep their reserve. Lower --margin, or give the trip more time.")

    def saving(index):

        "Energy saved by giving section `index` one more step, as a max-heap key."

        time = times[index]

        if time + STEP > legs[index]['upper']:

            return None

        return -(energyAt(legs[index]['hull'], time) - energyAt(legs[index]['hull'], time + STEP))

    heap = [(value, index) for index in range(len(legs))
            if (value := saving(index)) is not None]

    heapify(heap)

    while remaining > 0:

        if not heap:

            raise ValueError(
                f"{remaining} s of running time are left over: every section is at the top of "
                f"its sampled range. Sweep a wider range of running times.")

        _, index = heappop(heap)

        times[index] += STEP
        remaining -= STEP

        if (value := saving(index)) is not None:

            heappush(heap, (value, index))

    return times


def dwellTimes(stops, variant, maxCut):

    """
    The dwell at every stop of the trip, one entry per intermediate stop.

    The 'running' variant keeps the timetable's dwells untouched. The 'dwell' variant
    shortens them, but never below the minimum waiting time the timetable gives for the
    stop and never by more than `maxCut` of what the timetable planned: a dwell well
    above its minimum is usually a connection being held, not time going spare. The
    minimum waiting time also lengthens the few dwells the timetable puts below it.
    """

    scheduled = [int(stop['operational']['dwell_s']) for stop in stops[1:-1]]

    if variant == 'running':

        return scheduled

    return [max(roundUp(stop['min_waiting_time_s']), roundUp(dwell*(1 - maxCut)))
            for stop, dwell in zip(stops[1:-1], scheduled)]


def timingPoints(positions, runningTimes, dwells):

    "The timing point list of a journey, in the convention of journeyFromTimetable.py."

    values = [[positions[0], 0, 0, 0.0, 0.0]]

    departure = 0

    for index, (position, running) in enumerate(zip(positions[1:], runningTimes)):

        arrival = departure + running

        # the lower bound is the previous departure, i.e. non-binding
        values.append([position, departure, arrival, 0.0, 0.0])

        if index < len(dwells):

            departure = arrival + dwells[index]

            values.append([position, departure, departure, 0.0, 0.0])

    return values


def describe(baseline, journeyId, variant, run, margin, legs, times, dwells, scheduledDwells,
             energies, names, trackId):

    """
    The description that travels with the journey file.

    It is written from scratch rather than extended from the description of the journey
    this one is derived from: that one states the running times and dwells of the
    timetable, and those are exactly what changed here.
    """

    total = sum(times) + sum(dwells)
    moved = sum(abs(new - leg['scheduled']) for leg, new in zip(legs, times))//2
    frozen = [leg['name'] for leg in legs if leg['frozen']]
    saved = energies['scheduled'] - energies['new']

    text = [
        f"Line S4, {names[0]} -> {names[-1]}, all {len(names)} stops of track {trackId} as "
        f"{len(names) - 1} journey sections.",

        f"Running times REALLOCATED: they are not the ones of the timetable. Journey "
        f"'{journeyId}' holds those; this one was derived from it by reallocateTimes.py, which "
        f"read sensitivity sweep {run} - the energy of every section as a measured function of "
        f"its running time - and spent the available running time where it saves most, in "
        f"steps of {STEP} s, the grid the operator plans this line on.",

        f"The trip keeps the duration the timetable gives it, {total} s, and every station keeps "
        f"its position and its place in the order. What changed is the split: {sum(times)} s of "
        f"running time and {sum(dwells)} s of dwell, with {moved} s of running time moved between "
        f"sections."
        + ("" if variant == 'running' else
           f" The dwells were cut to the minimum waiting time of each stop, rounded up to "
           f"{STEP} s, which moved {sum(scheduledDwells) - sum(dwells)} s out of dwell and into "
           f"running time; the timetable's own dwells are in '{journeyId}'."),

        f"Every section keeps a reserve of at least {margin:.0%} over the shortest running time it "
        f"could physically do, measured as the time optimal solution of the same sweep, and no "
        f"section was moved outside the range of running times the sweep sampled."
        + (f" {len(frozen)} sections have no measured curve - the solver does not converge on them "
           f"under the settings of the sweep - and keep the running time the timetable gives them: "
           f"{', '.join(frozen)}." if frozen else ""),

        f"Over the {len(energies['legs'])} sections that do have a curve, the reallocation is "
        f"predicted to need {energies['new']:.1f} kWh against {energies['scheduled']:.1f} kWh for "
        f"the timetable allocation, i.e. {saved:.1f} kWh "
        f"({100*saved/abs(energies['scheduled']):.1f}%) less. That is what the measured curves "
        f"predict, not a solved trajectory: run the ocp on this journey to confirm it.",

        "Stopping points carry both speed constraints at 0, so each station appears twice "
        "(arrival then departure) except the two termini. Departures are fixed (lower = upper); "
        "arrivals give the scheduled time as the upper bound and the previous departure as a "
        "non-binding lower bound. Dwell times do not enter the OCP: each journey section is "
        "solved on its own and shifted to t = 0, so only the per-section running time matters.",
    ]

    return ' '.join(text)


def report(direction, variant, legs, times, dwells, scheduledDwells):

    "Per section table of what the reallocation did, printed as it is computed."

    print(f"\n{direction}, variant '{variant}'")
    print("  {:<12} {:>7} {:>7} {:>7} {:>6} {:>9} {:>9}".format(
        'section', 'minimum', 'planned', 'new', 'change', 'dE/dt old', 'energy'))

    for leg, time in zip(legs, times):

        if leg['frozen']:

            print("  {:<12} {:>7} {:>7} {:>7} {:>6} {:>9} {:>9}".format(
                leg['name'], '-', leg['scheduled'], time, 0, '-', 'no curve'))

            continue

        marginal = (energyAt(leg['hull'], leg['scheduled'])
                    - energyAt(leg['hull'], leg['scheduled'] + STEP))/STEP

        print("  {:<12} {:>7.0f} {:>7} {:>7} {:>+6} {:>9.4f} {:>+9.2f} {}".format(
            leg['name'], leg['minimum'], leg['scheduled'], time, time - leg['scheduled'],
            marginal, energyAt(leg['hull'], time) - energyAt(leg['hull'], leg['scheduled']),
            'at the top of the sampled range' if time == leg['upper'] else ''))

    if variant != 'running':

        print(f"  dwell: {sum(scheduledDwells)} s planned -> {sum(dwells)} s, "
              f"{sum(scheduledDwells) - sum(dwells)} s freed for running")


def energiesOf(legs, times):

    "Predicted energy of an allocation, over the sections that have a curve."

    curved = [(leg, time) for leg, time in zip(legs, times) if not leg['frozen']]

    return {
        'legs': [leg['name'] for leg, _ in curved],
        'scheduled': sum(energyAt(leg['hull'], leg['scheduled']) for leg, _ in curved),
        'new': sum(energyAt(leg['hull'], time) for leg, time in curved),
    }


def main():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', default=None,
                        help="name of the sensitivity sweep to read (default: the latest complete one)")
    parser.add_argument('--margin', type=float, default=MARGIN,
                        help=f"reserve every section keeps over its shortest running time "
                             f"(default: {MARGIN:.0%})")
    parser.add_argument('--max-dwell-cut', type=float, default=MAX_DWELL_CUT,
                        help=f"largest share of a dwell the 'dwell' variant may take "
                             f"(default: {MAX_DWELL_CUT:.0%})")
    parser.add_argument('--variants', nargs='*', default=list(SUFFIXES),
                        help=f"which variants to write (default: {' '.join(SUFFIXES)})")
    parser.add_argument('--dry-run', action='store_true',
                        help="report the reallocation, write no journey files")
    args = parser.parse_args()

    config = loadConfig()

    directory = Path(config['directory'])
    journeyDirectory = Path(config['journeyDirectory'])

    run = Path(directory / 'sensitivity' / args.run) if args.run \
        else latestCompleteRun(directory / 'sensitivity')

    sweep = readMetadata(run)

    assert sweep is not None, f"No {METADATA_FILE} in '{run}'"

    with open(directory / config['timetable'], encoding='utf-8') as f:
        timetable = json.load(f)

    print(f"Sweep {run.name}, from ocp run {sweep['baselineRun']}, "
          f"reserve {args.margin:.0%} over the time optimum, {STEP} s steps")

    for variant in args.variants:

        assert variant in SUFFIXES, f"Unknown variant '{variant}', expected one of {list(SUFFIXES)}"

    for direction, journeyId in sweep['inputs']['journeys'].items():

        with open(journeyDirectory / f'{journeyId}.json', encoding='utf-8') as f:
            baseline = json.load(f)

        names = baseline['metadata']['station names']

        # a stop is one position carrying an arrival and a departure, the termini only one
        positions = []

        for values in baseline['timing points']['values']:

            if not positions or values[0] != positions[-1]:

                positions.append(values[0])

        assert len(positions) == len(names), \
            f"Journey '{journeyId}' has {len(positions)} stops for {len(names)} station names"

        stops = stopsOfTimetable(timetable, names, journeyId)

        sections = [section for section in sweep['sections'] if section['direction'] == direction]

        for variant in args.variants:

            legs = buildLegs(sections, args.margin)

            scheduledDwells = dwellTimes(stops, 'running', args.max_dwell_cut)
            dwells = dwellTimes(stops, variant, args.max_dwell_cut)

            budget = sum(leg['scheduled'] for leg in legs) + sum(scheduledDwells) - sum(dwells)

            times = allocate(legs, budget)

            report(direction, variant, legs, times, dwells, scheduledDwells)

            energies = energiesOf(legs, times)

            print(f"  predicted energy {energies['scheduled']:.1f} -> {energies['new']:.1f} kWh "
                  f"over {len(energies['legs'])} of {len(legs)} sections")

            if args.dry_run:

                continue

            newId = journeyId + SUFFIXES[variant]

            journey = {
                'metadata': {
                    'id': newId,
                    'library version': LIBRARY_VERSION,
                    'created by': baseline['metadata']['created by'],
                    'associated track': baseline['metadata']['associated track'],
                    'description': describe(baseline, journeyId, variant, run.name, args.margin,
                                            legs, times, dwells, scheduledDwells, energies,
                                            names, baseline['metadata']['associated track']),
                    'station names': names,
                },
                'timing points': {
                    'units': baseline['timing points']['units'],
                    'values': timingPoints(positions, times, dwells),
                },
            }

            target = journeyDirectory / f'{newId}.json'

            with open(target, 'w', encoding='utf-8') as f:
                json.dump(journey, f, ensure_ascii=False, indent=2)

            print(f"  wrote {target}")


def stopsOfTimetable(timetable, names, journeyId):

    "The commercial stops of the timetable that this journey was built from."

    for entry in timetable['timetables']:

        stops = [point for point in entry['points'] if point['stop_type'] == STOP]

        if [stop['name'] for stop in stops] == names:

            return stops

    raise ValueError(f"No timetable in the extract calls at the stations of journey '{journeyId}'")


if __name__ == '__main__':

    main()
