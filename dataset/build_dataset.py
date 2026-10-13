#!/usr/bin/env python3
"""Build dataset.csv from the per-film JSON annotations in movies/.

Source of truth is movies/NN-slug.json; the CSV is a build artifact. Each film
stores its event set once, in fabula (chronological) order, plus a syuzhet_order
permutation giving the told order. The schema is documented in README.md.

Usage:
    python3 build_dataset.py           # validate + rebuild dataset.csv
    python3 build_dataset.py --check   # validate only, write nothing
"""
from __future__ import annotations

import bisect
import csv
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
MOVIES_DIR = HERE / "movies"
OUT_CSV = HERE / "dataset.csv"

COLUMNS = ["title", "release_year", "genre", "source", "fabula", "syuzhet",
           "fabula_order", "syuzhet_order",
           "fabula_scene_order", "syuzhet_scene_order", "tau", "lis",
           "syuzhet_notes", "fabula_notes"]
SEP = " | "
ORDER_SEP = " "
MIN_EVENTS, MAX_EVENTS = 10, 20

# Where the annotated telling comes from. The corpus began as films and grew to take
# in prose and theatre, because reverse chronology and achrony are far commoner on the
# page and the stage than on screen. `release_year` is the year of first release,
# publication or premiere, whichever applies.
SOURCES = {"Film", "Novel", "Short story", "Play", "Musical"}

# Canonical corpus order. The NN prefix on each movies/*.json matches this list.
FILMS = [
    ("Pulp Fiction", 1994),
    ("Memento", 2000),
    ("Eternal Sunshine of the Spotless Mind", 2004),
    ("The Godfather Part II", 1974),
    ("Citizen Kane", 1941),
    ("Reservoir Dogs", 1992),
    ("21 Grams", 2003),
    ("Babel", 2006),
    ("Crash", 2004),
    ("Amores Perros", 2000),
    ("The Usual Suspects", 1995),
    ("Fight Club", 1999),
    ("Irreversible", 2002),
    ("500 Days of Summer", 2009),
    ("Arrival", 2016),
    ("Inception", 2010),
    ("The Sixth Sense", 1999),
    ("Cloud Atlas", 2012),
    ("The Prestige", 2006),
    ("Rashomon", 1950),
    ("Slumdog Millionaire", 2008),
    ("Big Fish", 2003),
    ("The Curious Case of Benjamin Button", 2008),
    ("Vantage Point", 2008),
    ("Go", 1999),
    ("Magnolia", 1999),
    ("Short Cuts", 1993),
    ("Mulholland Drive", 2001),
    ("Lost Highway", 1997),
    ("Donnie Darko", 2001),
    ("Looper", 2012),
    ("Primer", 2004),
    ("Annie Hall", 1977),
    ("The Hours", 2002),
    ("Atonement", 2007),
    ("Once Upon a Time in America", 1984),
    ("Kill Bill: Vol. 1", 2003),
    ("The Girl with the Dragon Tattoo", 2011),
    ("Dunkirk", 2017),
    ("Tenet", 2020),
    ("Everything Everywhere All at Once", 2022),
    ("Nine", 2009),
    ("Sliding Doors", 1998),
    ("Run Lola Run", 1998),
    ("Snatch", 2000),
    ("Traffic", 2000),
    ("Zero Dark Thirty", 2012),
    ("JFK", 1991),
    ("Blue Valentine", 2010),
    ("Requiem for a Dream", 2000),
    # Added to give every nonlinear_category a usable floor of samples. The five thin
    # classes could not be filled from cinema alone, so prose and theatre enter here.
    ("Betrayal", 1978),
    ("The Night Watch", 2006),
    ("Merrily We Roll Along", 1981),
    ("Peppermint Candy", 1999),
    ("5x2", 2004),
    ("Journey Back to the Source", 1944),
    ("Time's Arrow", 1991),
    ("Happy End", 1967),
    ("Two Friends", 1986),
    ("The Last Five Years", 2001),
    ("Arcadia", 1993),
    ("Little Women", 2019),
    ("The God of Small Things", 1997),
    ("Sunset Boulevard", 1950),
    ("Goodfellas", 1990),
    ("Michael Clayton", 2007),
    ("The Prime of Miss Jean Brodie", 1961),
    ("Twelve Monkeys", 1995),
    ("Nashville", 1975),
    ("Grand Hotel", 1932),
    ("Love Actually", 2003),
    ("Syriana", 2005),
    ("Do the Right Thing", 1989),
    ("United 93", 2006),
    ("Slaughterhouse-Five", 1969),
    ("Catch-22", 1961),
    ("The Limey", 1999),
    ("Mirror", 1975),
    ("The Tree of Life", 2011),
    ("Waltz with Bashir", 2008),
    ("Beloved", 1987),
    ("The Burning Plain", 2008),
    ("Je t'aime, je t'aime", 1968),
]


class FilmError(Exception):
    """A film's annotation failed validation."""


# The two deterministic nonlinearity metrics, defined as in README.md so the
# numbers are comparable across the corpus and the benchmark.
def kendall_tau(t: list[int]) -> float:
    """+1.0 when the told order is strictly chronological, -1.0 when reversed."""
    c = d = 0
    for i, j in itertools.combinations(range(len(t)), 2):
        c += t[i] < t[j]
        d += t[i] > t[j]
    return (c - d) / (c + d) if c + d else 1.0


def lis_ratio(t: list[int]) -> float:
    """Longest increasing subsequence over length: 1.0 if the telling never doubles back."""
    p: list[int] = []
    for x in t:
        k = bisect.bisect_left(p, x)
        p[k:k + 1] = [x]
    return len(p) / len(t) if t else 1.0


def validate_scenes(scene_of: list, order: list, n: int) -> None:
    """Raise FilmError unless event_scenes is a usable scene segmentation.

    A scene is a stretch the film presents as one continuous unit, so it must be
    contiguous in *both* orders: an unbroken run of chronological events, told as
    an unbroken run of told positions. That is what keeps the scene sequences a
    clean permutation of 1..M, parseable exactly like the event-level ones.
    """
    if not isinstance(scene_of, list) or len(scene_of) != n:
        raise FilmError(f"event_scenes must be a list of {n} scene ids, "
                        f"got {len(scene_of) if isinstance(scene_of, list) else '?'}")
    if not all(isinstance(s, int) for s in scene_of):
        raise FilmError("event_scenes must contain integers")

    # Chronologically the ids must read 1, 1, 2, 3, 3, ... - starting at 1 and
    # stepping up by exactly one at each boundary, so scenes are numbered by
    # chronological position and each is a contiguous run of events.
    if scene_of[0] != 1:
        raise FilmError(f"event_scenes starts at {scene_of[0]}, must start at 1")
    expected = 1
    for i, s in enumerate(scene_of, 1):
        if s == expected:
            continue
        if s == expected + 1:
            expected = s
            continue
        raise FilmError(f"event_scenes jumps to {s} at event {i}; scene ids must "
                        f"start at 1 and increase by one at each scene boundary")

    # And the told order must enter each scene exactly once, never returning to
    # one it has left, and never running a scene's own events out of order.
    told = [scene_of[i - 1] for i in order]
    entered: list[int] = []
    for s in told:
        if entered and entered[-1] == s:
            continue
        if s in entered:
            raise FilmError(f"scene {s} is split across told order, so it is not "
                            f"one continuous scene")
        entered.append(s)
    for scene_id in range(1, expected + 1):
        members = [i for i, s in enumerate(scene_of, 1) if s == scene_id]
        told_pos = [order.index(i) for i in members]
        if told_pos != sorted(told_pos):
            raise FilmError(f"scene {scene_id} is told out of chronological order "
                            f"internally, so its events are not one continuous run")


def scene_orders(scene_of: list, order: list) -> tuple[list, list]:
    """Per-event scene ids in fabula order and in told order."""
    return scene_of, [scene_of[i - 1] for i in order]


def path_for(index: int) -> Path:
    """movies/NN-slug.json for the 1-based corpus position `index`."""
    title = FILMS[index - 1][0]
    slug = "".join(c if c.isalnum() else "-" for c in title.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return MOVIES_DIR / f"{index:02d}-{slug.strip('-')}.json"


def validate(film: dict, expect_title: str, expect_year: int) -> None:
    """Raise FilmError unless the annotation satisfies the README.md schema."""
    if film.get("title") != expect_title:
        raise FilmError(f"title is {film.get('title')!r}, expected {expect_title!r}")
    if film.get("release_year") != expect_year:
        raise FilmError(f"release_year is {film.get('release_year')!r}, expected {expect_year}")

    if film.get("source") not in SOURCES:
        raise FilmError(f"source is {film.get('source')!r}, expected one of "
                        f"{sorted(SOURCES)}")

    genre = film.get("genre", "")
    tags = [t.strip() for t in genre.split(",")]
    if not (1 <= len(tags) <= 3) or not all(tags):
        raise FilmError(f"genre must be 1-3 non-empty tags, got {genre!r}")

    events = film.get("events_chronological")
    if not isinstance(events, list):
        raise FilmError("events_chronological must be a list")
    n = len(events)
    if not MIN_EVENTS <= n <= MAX_EVENTS:
        raise FilmError(f"{n} events, expected {MIN_EVENTS}-{MAX_EVENTS}")
    for i, ev in enumerate(events, 1):
        if not isinstance(ev, str) or not ev.strip():
            raise FilmError(f"event {i} is empty or not a string")
        if SEP.strip() in ev:
            raise FilmError(f"event {i} contains the '|' separator: {ev!r}")
    if len(set(events)) != n:
        dupes = {e for e in events if events.count(e) > 1}
        raise FilmError(f"duplicate event text: {sorted(dupes)[:1]}")

    order = film.get("syuzhet_order")
    if not isinstance(order, list):
        raise FilmError("syuzhet_order must be a list")
    if sorted(order) != list(range(1, n + 1)):
        raise FilmError(
            f"syuzhet_order is not a permutation of 1..{n} "
            f"(len {len(order)}, {len(set(order))} distinct)"
        )

    validate_scenes(film.get("event_scenes"), order, n)

    for key in ("syuzhet_notes", "fabula_notes"):
        if not isinstance(film.get(key, ""), str):
            raise FilmError(f"{key} must be a string")


def row_for(film: dict) -> list:
    """Flatten a validated film into its CSV row.

    Events are numbered 1..N by chronological position, so fabula_order is the
    identity sequence by construction and syuzhet_order carries the permutation;
    tau and lis are computed from syuzhet_order. The two scene columns label the
    same two bullet lists with scene ids, so they run parallel to the event ones.
    """
    events = film["events_chronological"]
    order = film["syuzhet_order"]
    fabula_order = list(range(1, len(events) + 1))
    fabula_scenes, syuzhet_scenes = scene_orders(film["event_scenes"], order)
    return [
        film["title"],
        film["release_year"],
        film["genre"],
        film["source"],
        SEP.join(events),
        SEP.join(events[i - 1] for i in order),
        ORDER_SEP.join(str(i) for i in fabula_order),
        ORDER_SEP.join(str(i) for i in order),
        ORDER_SEP.join(str(s) for s in fabula_scenes),
        ORDER_SEP.join(str(s) for s in syuzhet_scenes),
        f"{kendall_tau(order):.4f}",
        f"{lis_ratio(order):.4f}",
        film.get("syuzhet_notes", ""),
        film.get("fabula_notes", ""),
    ]


def main() -> int:
    check_only = "--check" in sys.argv
    total = len(FILMS)
    done: list[str] = []
    failed: list[tuple[str, str]] = []

    out = None if check_only else OUT_CSV.open("w", newline="", encoding="utf-8")
    writer = None
    if out is not None:
        writer = csv.writer(out)
        writer.writerow(COLUMNS)
        out.flush()

    try:
        for index, (title, year) in enumerate(FILMS, 1):
            path = path_for(index)
            try:
                if not path.exists():
                    raise FilmError(f"missing annotation file {path.name}")
                film = json.loads(path.read_text(encoding="utf-8"))
                validate(film, title, year)
                if writer is not None:
                    # Flush per film so an interrupted run leaves valid partial output.
                    writer.writerow(row_for(film))
                    out.flush()
            except (FilmError, json.JSONDecodeError) as exc:
                failed.append((f"{title} ({year})", str(exc)))
                print(f"[{index}/{total}] {title} FAILED: {exc}")
                continue
            done.append(title)
            n = len(film["events_chronological"])
            print(f"[{index}/{total}] {title} done ({n} events)")
    finally:
        if out is not None:
            out.close()

    print()
    print(f"Summary: {len(done)}/{total} succeeded, {len(failed)} failed.")
    for name, reason in failed:
        print(f"  FAILED  {name}: {reason}")
    if not check_only and done:
        print(f"Wrote {OUT_CSV} ({len(done)} rows).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
