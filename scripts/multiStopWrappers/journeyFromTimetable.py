"""
Build TTOBench journey files from an SOB operational timetable extract.

The timetable is the JSON produced by `make_s4.py` in the Innocheque data folder
(`timetable/s4_timetable.json`): one canonical timetable per direction, every point
carrying `operational` times with second precision and `commercial` times rounded to
the published minute.

Each `commercialStop` of a timetable becomes a pair of stopping points (arrival and
departure) at the position of the station of the same name on the track, so that the
journey has one section per stop-to-stop leg. Passing points are not carried over:
the track does not say where they are.

Departures are fixed (lower = upper), arrivals give the scheduled time as the upper
bound and the previous departure as a non-binding lower bound. Since each journey
section is solved on its own and shifted to t = 0, the running time of a section is
the arrival time minus the previous departure time - with operational times that is
the running time the timetable actually plans, with commercial times it is the
running time plus the dwell that the minute rounding hides.

A timetable whose stops run against the station order of the track (the return
service) is written with its positions mirrored on the track length, so that it can
be used directly with a reversed track.

Usage (paths and ids come from config.json, see config.example.json):

    python journeyFromTimetable.py                    # both directions, operational
    python journeyFromTimetable.py --basis commercial
    python journeyFromTimetable.py --dry-run
"""

import argparse
import json

from pathlib import Path

from wrapperConfig import loadConfig

TIMETABLE_FILE = 's4_timetable.json'  # default, override with "timetable" in config.json

LIBRARY_VERSION = 'TTOBench v1.5'

STOP = 'commercialStop'


def loadTrack(path):

    "Station names, their positions, the track length and its author, from a TTOBench track file."

    with open(path, encoding='utf-8') as f:
        track = json.load(f)

    names = track['metadata'].get('station names')

    assert names is not None, f"Track '{path}' carries no metadata['station names']"

    positions = track['stops']['values']

    assert track['stops']['unit'] == 'm', f"Expected stop positions in m, got '{track['stops']['unit']}'"
    assert len(names) == len(positions), "Track has {} station names for {} stops".format(len(names), len(positions))

    return names, positions, positions[-1], track['metadata'].get('created by')


def orientation(stopNames, trackNames):

    """
    'forward' if the timetable calls at the stations in the order of the track,
    'reverse' if it runs against it. Anything else is not this track.
    """

    if stopNames == trackNames:

        return 'forward'

    if stopNames == trackNames[::-1]:

        return 'reverse'

    unknown = [name for name in stopNames if name not in trackNames]

    raise ValueError("Timetable stops do not match the stations of the track. "
                     + (f"Unknown to the track: {unknown}." if unknown
                        else "Same stations, but in neither the track order nor its reverse."))


def timingPoints(stops, positions, basis):

    """
    The timing point list of one timetable: two stopping points per intermediate
    station, one for the terminus at either end.

    Times are shifted so that the first departure is t = 0. That matters for the
    commercial basis, whose origin departure is a few seconds negative because the
    published times are rounded to the minute.
    """

    origin = stops[0][basis]['dep_s']

    values = []
    previousDeparture = None

    for stop, position in zip(stops, positions):

        arrival = stop[basis]['arr_s']
        departure = stop[basis]['dep_s']

        assert (arrival is None) == (stop is stops[0]), f"Missing arrival time at {stop['name']}"
        assert (departure is None) == (stop is stops[-1]), f"Missing departure time at {stop['name']}"

        if arrival is not None:

            # the lower bound is the previous departure, i.e. non-binding: it only says
            # that the train cannot arrive before it left the station before
            values.append([float(position), previousDeparture - origin, arrival - origin, 0.0, 0.0])

        if departure is not None:

            values.append([float(position), departure - origin, departure - origin, 0.0, 0.0])

            previousDeparture = departure

    return values


def plural(count, singular, several):

    "'1 other train deviates' vs '3 others deviate'."

    return f"{count} {singular if count == 1 else several}"


def describe(timetable, meta, basis, sense, trackId, source):

    "The description that travels with the journey file, so results can be traced back to it."

    stops = [p for p in timetable['points'] if p['stop_type'] == STOP]

    running = sum(b[basis]['arr_s'] - a[basis]['dep_s'] for a, b in zip(stops, stops[1:]))
    total = stops[-1][basis]['arr_s'] - stops[0][basis]['dep_s']

    text = [
        f"Line {meta.get('line', '?')}, {timetable['origin_name']} -> {timetable['destination_name']}, "
        f"all {len(stops)} stops of track {trackId} as {len(stops) - 1} journey sections.",

        f"Times are the {basis} times of the timetable extract '{source}' "
        f"(timetable '{timetable['id']}', {meta.get('source', 'operator extract')}), "
        f"reference train {timetable['reference_train']}, which "
        f"{timetable['n_trains_exact']} of the {timetable['n_trains']} trains of this direction run "
        f"to the second"
        + (f"; the {plural(len(timetable['variants']), 'other train deviates', 'others deviate')} "
           f"by at most {meta.get('variant_tolerance_s')} s before re-converging."
           if timetable['variants'] else "."),

        f"Times are seconds after the departure at {timetable['origin_name']}, so the trip spans "
        f"0 - {total} s, of which {running} s are running time and {total - running} s dwell.",

        "Stopping points carry both speed constraints at 0, so each station appears twice (arrival "
        "then departure) except the two termini. Departures are fixed (lower = upper); arrivals give "
        "the scheduled time as the upper bound and the previous departure as a non-binding lower bound. "
        "Dwell times do not enter the OCP: each journey section is solved on its own and shifted to "
        "t = 0, so only the per-section running time matters.",
    ]

    if basis == 'operational':

        text.append("The operational times are second-precision and need no running time supplements: "
                    "they are the times the operator plans the train to run to.")

    else:

        text.append("CAVEAT: the commercial times are rounded to whole minutes, so the dwell reads as 0 s "
                    "at the stops whose arrival and departure fall in the same minute, and every section "
                    "running time carries up to +-60 s of rounding error. Prefer the operational basis.")

    if sense == 'reverse':

        text.append(f"This is the return service and it runs against the direction of track {trackId}. "
                    f"Positions are mirrored on the track length, i.e. measured from "
                    f"{timetable['origin_name']}, so the journey belongs to the reversed track. It is a "
                    f"timetable of its own, not the mirror image of the outbound one.")

    return ' '.join(text)


def main():

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--basis', choices=['operational', 'commercial'], default='operational',
                        help="which set of times to use (default: operational)")
    parser.add_argument('--timetable', default=None,
                        help="timetable JSON, relative to the input directory "
                             f"(default: 'timetable' in config.json, else {TIMETABLE_FILE})")
    parser.add_argument('--prefix', default='CH_Journeys',
                        help="journey id prefix, the timetable id is appended (default: CH_Journeys)")
    parser.add_argument('--dry-run', action='store_true', help="report what would be written, write nothing")
    args = parser.parse_args()

    config = loadConfig()

    directory = Path(config['directory'])
    journeyDirectory = Path(config['journeyDirectory'])
    trackId = config['trackId']

    timetablePath = directory / (args.timetable or config.get('timetable', TIMETABLE_FILE))
    trackPath = Path(config['trackDirectory']) / f'{trackId}.json'

    assert timetablePath.is_file(), f"No timetable file '{timetablePath}'"

    with open(timetablePath, encoding='utf-8') as f:
        timetable = json.load(f)

    trackNames, trackPositions, trackLength, author = loadTrack(trackPath)

    for entry in timetable['timetables']:

        stops = [p for p in entry['points'] if p['stop_type'] == STOP]

        names = [p['name'] for p in stops]

        sense = orientation(names, trackNames)

        # the return service runs against the track, so its positions are the ones of the
        # reversed track, measured from what is the far end of the track file
        positions = [trackLength - trackPositions[trackNames.index(name)] if sense == 'reverse'
                     else trackPositions[trackNames.index(name)] for name in names]

        # 'S4:RW-SA' -> 'CH_Journeys_S4_RW_SA'
        journeyId = args.prefix + '_' + entry['id'].replace(':', '_').replace('-', '_')

        journey = {
            'metadata': {
                'id': journeyId,
                'library version': LIBRARY_VERSION,
                'created by': author,
                'associated track': trackId,
                'description': describe(entry, timetable['meta'], args.basis, sense,
                                        trackId, timetablePath.name),
                'station names': names,
            },
            'timing points': {
                'units': {
                    'position': 'm',
                    'lower time constraint': 's',
                    'upper time constraint': 's',
                    'lower speed constraint': 'km/h',
                    'upper speed constraint': 'km/h',
                },
                'values': timingPoints(stops, positions, args.basis),
            },
        }

        target = journeyDirectory / f'{journeyId}.json'

        runs = [b[args.basis]['arr_s'] - a[args.basis]['dep_s'] for a, b in zip(stops, stops[1:])]

        print(f"{journeyId}: {entry['id']}, {args.basis} times, {sense} on {trackId}, "
              f"{len(stops)} stops, {len(runs)} sections, "
              f"running time {sum(runs)} s, shortest section {min(runs)} s")

        if args.dry_run:

            continue

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(journey, f, ensure_ascii=False, indent=2)

        print(f"  wrote {target}")


if __name__ == '__main__':

    main()
