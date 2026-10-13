#!/usr/bin/env python3
"""Sanity-check both built corpora: round-trip the permutation and report nonlinearity.

Reads dataset.csv and dataset_anonymized.csv exactly the way a consumer would
(splitting on " | "), rebuilds the fabula->syuzhet permutation from the prose columns
alone, and computes the same two deterministic metrics the benchmark uses:

    tau  = Kendall's tau of the told order against chronological order
           (1.0 = perfectly chronological, -1.0 = perfectly reversed)
    lis  = longest-increasing-subsequence ratio
           (1.0 = the told order never doubles back)

The metric functions are deliberately re-implemented here rather than imported from
build_dataset, so that this script is an independent check on the build rather than a
restatement of it. It then confirms that the recomputed permutation and metrics agree
with the stored fabula_order / syuzhet_order / tau / lis columns - a mismatch means
a CSV is stale or was hand-edited.

It also checks the two scene columns, which carry the same two orderings at scene
granularity: one id per bullet, repeated where several bullets belong to one scene.
Permuting fabula_scene_order by the told order must reproduce syuzhet_scene_order,
collapsing each run of repeats must leave a permutation of 1..M - which is only true
if every scene is one unbroken block in both orders - and within a scene the telling
must play the events in their own chronological order.

Where an anonymized row carries a story_graph, its anchors are re-derived too: node ei
must be bullet i, every edge must land on an event that exists, and a REVEALS edge must
arrive late in the telling, checked against the told order recovered from the prose
rather than against the order the build wrote out.

The anonymized corpus is checked by the same code and then joined against the original:
the two files describe the same 83 permutations, so the columns they share must be
byte-identical row-for-row, and anonymization_key.csv must map each story_id back to
the right title. Checking the anonymized file matters most - it is the one that goes to
models under test, so it is the one whose silent corruption would be invisible.

A work in this corpus scoring tau == 1.0 is suspicious: it means the annotation
says the work is told in strict chronological order.

Usage:  python3 verify_dataset.py
"""
from __future__ import annotations

import bisect
import csv
import itertools
import json
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / "dataset.csv"
ANON_CSV_PATH = HERE / "dataset_anonymized.csv"
KEY_PATH = HERE / "anonymization_key.csv"
SEP = " | "

# Everything the two corpora say about the same work in the same words. title and
# release_year live only in the original, story_id and nonlinear_category only in the
# anonymized one; the rest describes structure, which anonymizing must not touch.
SHARED_COLUMNS = ["genre", "source", "fabula_order", "syuzhet_order",
                  "fabula_scene_order", "syuzhet_scene_order", "tau", "lis"]


def kendall_tau(t: list[int]) -> float:
    c = d = 0
    for i, j in itertools.combinations(range(len(t)), 2):
        c += t[i] < t[j]
        d += t[i] > t[j]
    return (c - d) / (c + d) if c + d else 1.0


def lis_ratio(t: list[int]) -> float:
    p: list[int] = []
    for x in t:
        k = bisect.bisect_left(p, x)
        p[k:k + 1] = [x]
    return len(p) / len(t) if t else 1.0


def label_for(row: dict) -> str:
    """The work's name in whichever corpus this row came from."""
    if "title" in row:
        return f"{row['title']} ({row['release_year']})"
    return row["story_id"]


def check_graph(raw: str, n: int, t: list[int], title: str) -> list[str]:
    """Re-derive the story graph's anchors from the CSV alone.

    Restated rather than imported, like the metrics: node ei must be bullet i, every
    edge must land on a node that exists, and a REVEALS edge must arrive late in the
    telling - which is checked here against the told order recovered from the prose
    columns, not against the order the build wrote out.
    """
    problems: list[str] = []
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [f"{title}: story_graph is not valid JSON ({exc})"]

    nodes = graph.get("events", [])
    if len(nodes) != n:
        problems.append(f"{title}: story_graph has {len(nodes)} nodes but the story "
                        f"has {n} bullets")
        return problems
    for i, node in enumerate(nodes, 1):
        if node.get("id") != f"e{i}" or node.get("story_time") != i:
            problems.append(f"{title}: node {i} is {node.get('id')!r} at story_time "
                            f"{node.get('story_time')!r}; node ei must be bullet i")

    ids = {f"e{i}" for i in range(1, n + 1)}
    # Told position of each 1-based chronological event, recovered from the prose.
    pos = {event + 1: k for k, event in enumerate(t)}
    for edge in graph.get("edges", []):
        a, b = edge.get("from"), edge.get("to")
        if a not in ids or b not in ids:
            problems.append(f"{title}: {edge.get('type')} edge {a}->{b} names an event "
                            f"that does not exist")
        elif edge.get("type") == "REVEALS" and pos[int(a[1:])] <= pos[int(b[1:])]:
            problems.append(f"{title}: REVEALS edge {a}->{b} is told explanation-first, "
                            f"so it withholds nothing")
    return problems


def check_corpus(path: Path) -> tuple[list[dict], list[str]]:
    """Round-trip every row of one CSV. Returns its rows and any problems found."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    problems: list[str] = []
    taus: list[float] = []
    scenes: list[tuple[int, int]] = []
    graphed = 0

    print(f"\n{path.name}")
    print(f"{'work':<44} {'n':>3} {'sc':>3} {'tau':>7} {'lis':>6}")
    print("-" * 67)
    for row in rows:
        fabula = row["fabula"].split(SEP)
        syuzhet = row["syuzhet"].split(SEP)
        title = label_for(row)

        if sorted(fabula) != sorted(syuzhet):
            problems.append(f"{title}: fabula and syuzhet are not the same event set")
            continue

        pos = {e: i for i, e in enumerate(fabula)}
        t = [pos[e] for e in syuzhet]
        tau, lis = kendall_tau(t), lis_ratio(t)
        taus.append(tau)

        # The stored columns must agree with what the prose columns imply.
        stored_fabula = [int(x) for x in row["fabula_order"].split()]
        stored_syuzhet = [int(x) for x in row["syuzhet_order"].split()]
        if stored_fabula != list(range(1, len(fabula) + 1)):
            problems.append(f"{title}: fabula_order is not the identity sequence")
        if stored_syuzhet != [i + 1 for i in t]:
            problems.append(f"{title}: syuzhet_order disagrees with the syuzhet text")
        if f"{tau:.4f}" != row["tau"] or f"{lis:.4f}" != row["lis"]:
            problems.append(
                f"{title}: stored tau/lis ({row['tau']}/{row['lis']}) "
                f"disagree with recomputed ({tau:.4f}/{lis:.4f})"
            )

        # The scene columns label the same two bullet lists, so permuting the
        # fabula one by syuzhet_order must reproduce the syuzhet one exactly.
        fab_scenes = [int(x) for x in row["fabula_scene_order"].split()]
        syu_scenes = [int(x) for x in row["syuzhet_scene_order"].split()]
        if len(fab_scenes) != len(fabula) or len(syu_scenes) != len(syuzhet):
            problems.append(f"{title}: scene columns are not one id per bullet")
        elif [fab_scenes[i] for i in t] != syu_scenes:
            problems.append(f"{title}: syuzhet_scene_order disagrees with "
                            f"fabula_scene_order under the permutation")
        elif sorted(set(fab_scenes)) != list(range(1, max(fab_scenes) + 1)):
            problems.append(f"{title}: scene ids are not 1..M")
        else:
            # Collapsing each run of repeats must leave a permutation of 1..M -
            # i.e. every scene is one unbroken block in both orders.
            runs = [s for i, s in enumerate(syu_scenes)
                    if i == 0 or s != syu_scenes[i - 1]]
            if sorted(runs) != list(range(1, max(fab_scenes) + 1)):
                problems.append(f"{title}: a scene is split across told order")
            # Contiguity alone would still allow a scene to play its own events
            # backwards. A scene is a continuous stretch, so it runs forwards.
            told = [i + 1 for i in t]
            for s in range(1, max(fab_scenes) + 1):
                members = [i + 1 for i, x in enumerate(fab_scenes) if x == s]
                if [e for e in told if fab_scenes[e - 1] == s] != members:
                    problems.append(f"{title}: scene {s} plays its own events "
                                    f"out of chronological order")
            scenes.append((max(fab_scenes), len(t)))

        # Anonymized rows only, and only once a graph has been authored for them.
        if row.get("story_graph"):
            problems += check_graph(row["story_graph"], len(fabula), t, title)
            graphed += 1

        flag = "  <- told chronologically?" if tau == 1.0 else ""
        m = max(fab_scenes) if fab_scenes else 0
        print(f"{title:<44} {len(t):>3} {m:>3} {tau:>7.3f} {lis:>6.3f}{flag}")

    print("-" * 67)
    if taus:
        m_total = sum(m for m, _ in scenes)
        n_total = sum(n for _, n in scenes)
        graphs = f", {graphed} carrying a story graph" if graphed else ""
        print(f"{len(rows)} works, mean tau {sum(taus) / len(taus):+.3f}, "
              f"{n_total} events grouped into {m_total} scenes{graphs}")
    else:
        print("no works")
    return rows, problems


def check_join(rows: list[dict], anon_rows: list[dict]) -> list[str]:
    """The two corpora describe the same works, so they must line up row-for-row."""
    problems: list[str] = []
    if len(rows) != len(anon_rows):
        return [f"dataset.csv has {len(rows)} rows but dataset_anonymized.csv has "
                f"{len(anon_rows)} - they cannot be joined"]

    key = {r["story_id"]: r for r in csv.DictReader(KEY_PATH.open(encoding="utf-8"))}
    for i, (row, anon) in enumerate(zip(rows, anon_rows), start=1):
        sid = anon["story_id"]
        title = label_for(row)
        if sid != f"S{i:02d}":
            problems.append(f"row {i}: story_id is {sid}, expected S{i:02d} - the "
                            f"anonymized corpus is out of sequence")
        for column in SHARED_COLUMNS:
            if row[column] != anon[column]:
                problems.append(f"{sid} ({title}): {column} differs between the "
                                f"corpora\n      dataset.csv: {row[column]}\n"
                                f"      anonymized:  {anon[column]}")
        if len(row["fabula"].split(SEP)) != len(anon["fabula"].split(SEP)):
            problems.append(f"{sid} ({title}): the two corpora have different "
                            f"event counts, so their bullets cannot be aligned")
        if sid not in key:
            problems.append(f"{sid} ({title}): missing from anonymization_key.csv")
        elif (key[sid]["original_title"] != row["title"]
                or key[sid]["release_year"] != row["release_year"]):
            problems.append(f"{sid}: the key says {key[sid]['original_title']} "
                            f"({key[sid]['release_year']}) but dataset.csv row {i} "
                            f"is {title}")

    extra = set(key) - {a["story_id"] for a in anon_rows}
    if extra:
        problems.append(f"anonymization_key.csv has ids no corpus row claims: "
                        f"{sorted(extra)}")
    return problems


def main() -> int:
    rows, problems = check_corpus(CSV_PATH)
    anon_rows, anon_problems = check_corpus(ANON_CSV_PATH)
    problems += anon_problems
    problems += check_join(rows, anon_rows)

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("Both corpora round-trip: fabula and syuzhet are exact permutations of each")
    print("other, the stored order/tau/lis columns agree with the prose columns, every")
    print("scene is one unbroken forward-running block in both orders, and the two")
    print("files agree on every column they share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
