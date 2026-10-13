#!/usr/bin/env python3
"""Build dataset_anonymized.csv from the de-identified annotations in movies_anon/.

Purpose: a benchmark corpus whose narrative structure is identical to dataset.csv but
whose surface identity has been stripped, so that an LLM's memorised knowledge of the
source film contributes as little as possible to its performance.

What is preserved, exactly:
  - the event count per story
  - the syuzhet_order permutation, and therefore tau and lis
  - the causal and temporal relationships between events
  - narrative granularity
  - the scene segmentation, and with it the nonlinear_category label describing
    the shape of the told scene order

What is replaced:
  - character names, place names, organisations, in-world titles and quotations,
    distinctive props, signature numbers, real people and real historical events
  - the film title and release year, which are dropped entirely in favour of a
    story_id; the mapping lives in anonymization_key.csv, which must NOT be shown
    to any model under test

Because both the fabula and syuzhet columns are generated from one anonymised event
list, the two can never drift out of permutation with each other.

Usage:
    python3 build_anonymized.py           # validate + rebuild dataset_anonymized.csv
    python3 build_anonymized.py --check   # validate only, write nothing
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from build_dataset import (FILMS, ORDER_SEP, SEP, SOURCES, FilmError, kendall_tau,
                           lis_ratio, path_for, scene_orders, validate_scenes)

HERE = Path(__file__).parent
ANON_DIR = HERE / "movies_anon"
OUT_CSV = HERE / "dataset_anonymized.csv"
KEY_CSV = HERE / "anonymization_key.csv"

COLUMNS = ["story_id", "genre", "source", "fabula", "syuzhet", "fabula_order",
           "syuzhet_order", "fabula_scene_order", "syuzhet_scene_order",
           "tau", "lis", "nonlinear_category", "syuzhet_notes", "fabula_notes",
           "story_graph"]

# The four stored edge types. BEFORE is induced by story_time - every pair of events
# already stands in it - so writing it out would be redundant and is refused.
EDGE_TYPES = {"CAUSES", "REVEALS", "FORESHADOWS", "SHARES"}

# Why an edge of each directed type may not run backwards through the chronology.
BACKWARD_REASON = {
    "CAUSES": "a cause precedes its effect",
    "FORESHADOWS": "the plant precedes the payoff",
    "REVEALS": "the withheld explanation is the earlier of the two events",
}

# World-state flags are values a checker can compare, not prose.
SCALARS = (str, int, float, bool, type(None))

# Flipped to True once all 83 stories carry a graph. Until then a story without one
# still builds, so the corpus stays green while the graphs are authored in batches.
REQUIRE_GRAPH = True

# The seven shapes a told scene order can take. Assigned by reading each story
# against its scene permutation, not derived from the permutation alone: three of
# the classes turn on what the scenes are about, not just where they sit.
CATEGORIES = {"ACHRONY", "CHRONICLE", "RETROGRADE", "ZIGZAG", "ANALEPSIS",
              "PROLEPSIS", "SYLLEPSIS"}

# Capitalised words that are generic enough to survive anonymisation: they identify
# no particular film. Anything else carried over from the source is a leak.
ALLOWED = {
    # sentence connectives and articles that can begin a clause mid-bullet
    "A", "An", "The", "In", "On", "At", "By", "As", "After", "Before", "During",
    "When", "While", "Then", "Now", "Later", "Years", "Days", "Day", "One", "Two",
    "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Twelve",
    "Told", "It", "He", "She", "They", "His", "Her", "Their", "There", "That",
    "This", "Because", "But", "And", "Or", "If", "So", "To", "For", "From", "Of",
    "With", "Without", "Over", "Under", "Through", "Into", "Out", "Up", "Down",
    "Back", "Only", "Every", "Each", "Both", "All", "No", "Not", "Its", "Him",
    # generic nouns that happen to be capitalised at a bullet start
    "Story", "Chapter", "Part", "Volume",
}

# Terms that are generic vocabulary rather than identifying, even though they are
# capitalised in the source and would otherwise be flagged.
ALLOWED_GENERIC = {
    "Christmas", "New", "Year", "Eve", "Communion", "Prohibition", "Depression",
    "Underground", "Federal", "Reserve", "President", "Senate", "Marines", "Army",
    "Navy", "Hindu", "Muslim", "Jews", "Hasidic", "Cambodians", "Korean",
    "American", "British", "Spanish", "Japanese", "Chinese", "German", "French",
    "Mexican", "Italian", "Russian", "Sicilian", "Irish", "Travellers", "Scots",
    "Bible", "God", "Jesus", "Catholic", "Metro", "Underground",
    "Christmases", "Romantic", "X-ray", "T-shirt", "T-shirts",
}


def scene_permutation(scene_of: list, order: list) -> list[int]:
    """The told scene order with repeats collapsed: a permutation of 1..M."""
    told = [scene_of[i - 1] for i in order]
    return [s for i, s in enumerate(told) if i == 0 or s != told[i - 1]]


def tokens(text: str) -> set[str]:
    """Capitalised tokens that are not sentence-initial - i.e. probable proper nouns."""
    found = set()
    # Split on sentence ends, allowing a closing quote after the terminator, so that
    # a word opening a sentence after `..."` is not mistaken for a proper noun.
    for sentence in re.split(r'(?:(?<=[.!?])|(?<=[.!?]"))\s+|^', text):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        for i, w in enumerate(words):
            if i == 0:
                continue
            if w[0].isupper():
                found.add(w)
    return found


def identifying_terms(index: int) -> set[str]:
    """Proper nouns and long numbers from the source film that must not survive."""
    src = json.loads(path_for(index).read_text(encoding="utf-8"))
    title, year = FILMS[index - 1]
    blob = " ".join(src["events_chronological"]) + " " + \
        src.get("syuzhet_notes", "") + " " + src.get("fabula_notes", "")

    terms = tokens(blob)
    # Title words count even when they would look sentence-initial.
    terms |= {w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", title) if w[0].isupper()}
    terms -= ALLOWED | ALLOWED_GENERIC
    terms |= {str(year)}
    # Four-digit numbers (years) and long figures are strong fingerprints.
    terms |= set(re.findall(r"\b\d{4,}\b", blob))
    return {t for t in terms if len(t) > 2}


def graph_text(graph: dict) -> str:
    """Every authored string in a story graph, flattened for the leak guard.

    The graph is a large new prose surface - character names, locations, free-text
    content - so it has to be scanned for source proper nouns exactly like the
    bullets are. Underscores become spaces so that snake_case state flags are
    scanned as words rather than hiding a term inside a single token.
    """
    parts: list[str] = list(graph.get("characters", {}))
    for node in graph.get("events", []):
        if not isinstance(node, dict):
            continue
        parts.extend(str(x) for x in node.get("core", []))
        parts.append(str(node.get("content", "")))
        parts.extend(str(p) for p in node.get("participants", []))
        parts.append(str(node.get("location", "")))
        parts.extend(str(k) for k in node.get("pre", {}))
        parts.extend(str(k) for k in node.get("post", {}))
        for who, facts in node.get("learns", {}).items():
            parts.append(str(who))
            parts.extend(str(f) for f in facts)
        parts.extend(str(v) for v in node.get("questions", {}).values())
    for edge in graph.get("edges", []):
        if isinstance(edge, dict) and edge.get("on"):
            parts.append(str(edge["on"]))
    return " ".join(parts).replace("_", " ")


def validate_graph(graph: dict, n: int, order: list[int]) -> None:
    """Raise FilmError unless story_graph is a usable gold fabula graph.

    Node ei is bullet i: the corpus already numbers events by chronological
    position, so anchoring the graph to that numbering is what keeps it joinable to
    the permutation columns and lets told_at stay derived rather than stored.

    Two of the checks here exist only because this corpus has a hand-annotated
    syuzhet as well as a fabula: a REVEALS edge has to arrive late in the telling,
    and a question has to be answered after it is asked. Both are vacuous without a
    known told order, and both are where the non-linearity actually lives.
    """
    if not isinstance(graph, dict):
        raise FilmError("story_graph must be an object")

    chars = graph.get("characters")
    if not isinstance(chars, dict) or not chars:
        raise FilmError("story_graph.characters must be a non-empty object")
    for name, attrs in chars.items():
        if not name.strip():
            raise FilmError("story_graph.characters has an empty name")
        if not isinstance(attrs, dict):
            raise FilmError(f"character {name!r} must map to an object of attributes")

    nodes = graph.get("events")
    if not isinstance(nodes, list) or len(nodes) != n:
        got = len(nodes) if isinstance(nodes, list) else "?"
        raise FilmError(f"story_graph has {got} nodes; there must be exactly one per "
                        f"bullet ({n})")

    known = set(chars) | {"reader"}
    questions: dict[str, str] = {}
    raised: dict[str, int] = {}
    resolved: dict[str, int] = {}
    state: dict[str, object] = {}

    for i, node in enumerate(nodes, 1):
        if not isinstance(node, dict):
            raise FilmError(f"node {i} is not an object")
        if node.get("id") != f"e{i}":
            raise FilmError(f"node {i} has id {node.get('id')!r}, expected 'e{i}': node "
                            f"ei is bullet i")
        if node.get("story_time") != i:
            raise FilmError(f"e{i} has story_time {node.get('story_time')!r}, expected "
                            f"{i}: events are numbered by chronological position")
        # told_at is the syuzhet position, and the syuzhet position is the thing the
        # benchmark asks a model to recover. Storing it would ship the answer.
        if node.get("told_at") not in (None, "", []):
            raise FilmError(f"e{i} sets told_at; it is empty in a gold graph and is "
                            f"derived from syuzhet_order")

        core = node.get("core")
        if (not isinstance(core, list) or len(core) != 3
                or not all(isinstance(x, str) and x.strip() for x in core)):
            raise FilmError(f"e{i}: core must be a subject-verb-object triple of "
                            f"non-empty strings, got {core!r}")
        if not isinstance(node.get("content"), str) or not node["content"].strip():
            raise FilmError(f"e{i}: content must be a non-empty string")

        parts = node.get("participants")
        if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
            raise FilmError(f"e{i}: participants must be a list of character names")
        undeclared = [p for p in parts if p not in chars]
        if undeclared:
            raise FilmError(f"e{i}: participants not declared in characters: "
                            f"{undeclared}")
        if not isinstance(node.get("location"), str) or not node["location"].strip():
            raise FilmError(f"e{i}: location must be a non-empty string")

        for field in ("pre", "post"):
            flags = node.get(field)
            if not isinstance(flags, dict):
                raise FilmError(f"e{i}: {field} must be an object of world-state flags")
            for key, value in flags.items():
                if not key.strip():
                    raise FilmError(f"e{i}: {field} has an empty flag name")
                if not isinstance(value, SCALARS):
                    raise FilmError(f"e{i}: {field}[{key!r}] must be a scalar a checker "
                                    f"can compare, got {value!r}")

        # The point of pre/post: an event is legal at its own story time iff its
        # preconditions already hold in the state every earlier event has produced.
        for key, value in node["pre"].items():
            if key not in state:
                raise FilmError(f"e{i}: pre[{key!r}] is never established by an earlier "
                                f"event's post, so the precondition cannot hold")
            if state[key] != value:
                raise FilmError(f"e{i}: pre[{key!r}] expects {value!r} but the state at "
                                f"story_time {i} has {state[key]!r}")
        state.update(node["post"])

        learns = node.get("learns")
        if not isinstance(learns, dict) or not learns:
            raise FilmError(f"e{i}: learns must record what someone comes to know; the "
                            f"epistemic layer is what non-linearity acts on")
        for who, facts in learns.items():
            if who not in known:
                raise FilmError(f"e{i}: learns names {who!r}, who is neither a declared "
                                f"character nor 'reader'")
            if (not isinstance(facts, list) or not facts
                    or not all(isinstance(f, str) and f.strip() for f in facts)):
                raise FilmError(f"e{i}: learns[{who!r}] must be a non-empty list of "
                                f"propositions")

        for qid, text in node.get("questions", {}).items():
            if qid in questions:
                raise FilmError(f"e{i}: question {qid!r} is defined twice")
            if not isinstance(text, str) or not text.strip():
                raise FilmError(f"e{i}: question {qid!r} has no text")
            questions[qid] = text

        for field, seen in (("raises", raised), ("resolves", resolved)):
            ids = node.get(field, [])
            if not isinstance(ids, list) or not all(isinstance(q, str) for q in ids):
                raise FilmError(f"e{i}: {field} must be a list of question ids")
            for qid in ids:
                if qid in seen:
                    raise FilmError(f"e{i}: question {qid!r} already appears in the "
                                    f"{field} of e{seen[qid]}; exactly one event may "
                                    f"{field.rstrip('s')} it")
                seen[qid] = i

    undefined = sorted((set(raised) | set(resolved)) - set(questions))
    if undefined:
        raise FilmError(f"raises/resolves name questions with no text: {undefined}")
    never_raised = sorted(set(questions) - set(raised))
    if never_raised:
        raise FilmError(f"questions nobody raises: {never_raised}")
    never_resolved = sorted(set(raised) - set(resolved))
    if never_resolved:
        raise FilmError(f"questions never resolved: {never_resolved}")

    # Told position of each chronological event, 0-based.
    pos = {event: p for p, event in enumerate(order)}

    edges = graph.get("edges")
    if not isinstance(edges, list) or not edges:
        raise FilmError("story_graph.edges must be a non-empty list")

    ids = {f"e{i}" for i in range(1, n + 1)}
    seen_edges: set[tuple[str, str, str]] = set()
    shared_pairs: set[frozenset[str]] = set()
    reveals: list[tuple[int, int]] = []
    linked: set[str] = set()

    for edge in edges:
        if not isinstance(edge, dict):
            raise FilmError(f"edge {edge!r} is not an object")
        etype, a, b = edge.get("type"), edge.get("from"), edge.get("to")
        if etype == "BEFORE":
            raise FilmError("BEFORE edges are induced by story_time and must not be "
                            "written out")
        if etype not in EDGE_TYPES:
            raise FilmError(f"edge type is {etype!r}, expected one of "
                            f"{sorted(EDGE_TYPES)}")
        if a not in ids or b not in ids:
            raise FilmError(f"{etype} edge {a!r}->{b!r} names an event that does not "
                            f"exist")
        if a == b:
            raise FilmError(f"{etype} edge on {a} points at itself")
        if (etype, a, b) in seen_edges:
            raise FilmError(f"duplicate {etype} edge {a}->{b}")
        seen_edges.add((etype, a, b))
        linked.update((a, b))

        i, j = int(a[1:]), int(b[1:])
        if etype in BACKWARD_REASON and i >= j:
            raise FilmError(f"{etype} edge {a}->{b} runs backwards through the "
                            f"chronology; {BACKWARD_REASON[etype]}")

        if etype == "REVEALS":
            # The whole content of the claim: the telling shows the effect, then the
            # explanation. Told the other way round, nothing was ever withheld.
            if pos[i] <= pos[j]:
                raise FilmError(f"REVEALS edge {a}->{b} is told explanation-first "
                                f"(positions {pos[i] + 1} then {pos[j] + 1}), so it "
                                f"reveals nothing")
            reveals.append((i, j))

        if etype == "SHARES":
            pair = frozenset((a, b))
            if pair in shared_pairs:
                raise FilmError(f"SHARES edge {a}-{b} is given twice; it is undirected")
            shared_pairs.add(pair)
            if i > j:
                raise FilmError(f"SHARES edge {a}->{b} is undirected and must be stored "
                                f"with the earlier event first")
            common = set(nodes[i - 1]["participants"]) & set(nodes[j - 1]["participants"])
            on = edge.get("on")
            if not common and not (isinstance(on, str) and on.strip()):
                raise FilmError(f"SHARES edge {a}-{b} shares no participant, so it must "
                                f"name what it does share in an 'on' field")

    orphans = sorted(ids - linked, key=lambda e: int(e[1:]))
    if orphans:
        raise FilmError(f"events joined to nothing: {orphans[:6]}")

    # A REVEALS edge and a raises/resolves pair are two statements of one fact, so
    # they must agree; otherwise the reveal machinery and the question machinery
    # drift apart and the structure_targets the kit derives go stale.
    for i, j in reveals:
        if not any(asker == j and resolved.get(qid) == i for qid, asker in raised.items()):
            raise FilmError(f"REVEALS edge e{i}->e{j} has no matching question: e{j} "
                            f"must raise a question that e{i} resolves")

    # The same rule the REVEALS edges are held to, applied to every question pair -
    # including the ones carried by raises/resolves alone, where the answer is the
    # later event and so no REVEALS edge can legally exist. Checked after the edges
    # so that a mis-told REVEALS reports as an edge fault rather than as this.
    for qid, asker in raised.items():
        answerer = resolved[qid]
        if pos[answerer] <= pos[asker]:
            raise FilmError(f"question {qid!r} is raised by e{asker} at told position "
                            f"{pos[asker] + 1} but resolved by e{answerer} at told "
                            f"position {pos[answerer] + 1}; an answer that arrives "
                            f"first withholds nothing")


def validate(anon: dict, index: int) -> None:
    """Raise FilmError unless the anonymised story preserves the source's structure."""
    src = json.loads(path_for(index).read_text(encoding="utf-8"))
    n_src = len(src["events_chronological"])

    if anon.get("source_index") != index:
        raise FilmError(f"source_index is {anon.get('source_index')!r}, expected {index}")
    if anon.get("story_id") != f"S{index:02d}":
        raise FilmError(f"story_id is {anon.get('story_id')!r}, expected S{index:02d}")
    if anon.get("genre") != src["genre"]:
        raise FilmError(f"genre {anon.get('genre')!r} differs from source {src['genre']!r}")
    # `source` names the medium the telling was read off, which anonymising does not
    # change - a play stays a play - so it has to survive the rewrite untouched.
    if anon.get("source") != src.get("source"):
        raise FilmError(f"source {anon.get('source')!r} differs from source file's "
                        f"{src.get('source')!r}")
    if anon.get("source") not in SOURCES:
        raise FilmError(f"source is {anon.get('source')!r}, expected one of "
                        f"{sorted(SOURCES)}")

    events = anon.get("events_chronological")
    if not isinstance(events, list) or len(events) != n_src:
        raise FilmError(f"{len(events) if isinstance(events, list) else '?'} events, "
                        f"source has {n_src}")
    for i, ev in enumerate(events, 1):
        if not isinstance(ev, str) or not ev.strip():
            raise FilmError(f"event {i} is empty or not a string")
        if "|" in ev:
            raise FilmError(f"event {i} contains the '|' separator")
    if len(set(events)) != n_src:
        raise FilmError("duplicate event text")

    if anon.get("syuzhet_order") != src["syuzhet_order"]:
        raise FilmError("syuzhet_order differs from the source - the permutation, and "
                        "therefore tau and lis, would no longer match dataset.csv")

    # Scene structure is narrative, not surface identity, so anonymising must not
    # touch it: the two corpora have to segment identically to stay joinable.
    if anon.get("event_scenes") != src.get("event_scenes"):
        raise FilmError("event_scenes differs from the source - the scene columns "
                        "would no longer match dataset.csv row-for-row")
    validate_scenes(anon.get("event_scenes"), anon["syuzhet_order"], n_src)

    # The classification is a judgement, so the build cannot check it - but it can
    # check the two things the rubric states categorically, and it can insist that
    # the reason exists, since a label nobody can audit is worse than none.
    category = anon.get("nonlinear_category")
    if category not in CATEGORIES:
        raise FilmError(f"nonlinear_category is {category!r}, expected one of "
                        f"{sorted(CATEGORIES)}")
    reason = anon.get("nonlinear_category_reason", "")
    if not isinstance(reason, str) or not reason.strip():
        raise FilmError("nonlinear_category_reason must say why this label was "
                        "chosen; SYLLEPSIS in particular has to name its principle")
    scenes = scene_permutation(anon["event_scenes"], anon["syuzhet_order"])
    tau = kendall_tau(scenes)
    if category == "RETROGRADE" and tau >= 0:
        raise FilmError(f"RETROGRADE claims the scenes run n..1, but their tau is "
                        f"{tau:+.4f}")
    # This one is a floor, not a proof: CHRONICLE tolerates simultaneous scenes
    # arriving in either order, and with concurrent strands that can displace a
    # scene a long way, so only a negative tau is conclusively wrong.
    if category == "CHRONICLE" and tau <= 0:
        raise FilmError(f"CHRONICLE claims the scenes run 1..n, but their tau is "
                        f"{tau:+.4f}")

    graph = anon.get("story_graph")
    if graph is None:
        if REQUIRE_GRAPH:
            raise FilmError("story_graph is missing")
    else:
        validate_graph(graph, n_src, anon["syuzhet_order"])

    # The point of the exercise: no source proper noun may survive anywhere - which
    # now includes everywhere in the graph, not just the bullets and the notes.
    banned = identifying_terms(index)
    blob = " ".join(events) + " " + anon.get("syuzhet_notes", "") + " " + \
        anon.get("fabula_notes", "") + " " + graph_text(graph or {})
    present = sorted(t for t in banned if re.search(rf"\b{re.escape(t)}\b", blob))
    if present:
        raise FilmError(f"identifying terms leaked from the source: {present[:6]}")

    # A bullet with no proper nouns can still be an exact-string retrieval cue, so
    # every sentence must be rewritten even when there was no name in it to change.
    verbatim = sorted(set(events) & set(src["events_chronological"]))
    if verbatim:
        raise FilmError(f"{len(verbatim)} bullet(s) carried over verbatim: "
                        f"{verbatim[0][:60]!r}")


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
            story_id = f"S{index:02d}"
            path = ANON_DIR / f"{index:02d}.json"
            try:
                if not path.exists():
                    raise FilmError(f"missing annotation file {path.name}")
                anon = json.loads(path.read_text(encoding="utf-8"))
                validate(anon, index)
                if writer is not None:
                    events = anon["events_chronological"]
                    order = anon["syuzhet_order"]
                    fab_scenes, syu_scenes = scene_orders(anon["event_scenes"], order)
                    writer.writerow([
                        story_id,
                        anon["genre"],
                        anon["source"],
                        SEP.join(events),
                        SEP.join(events[i - 1] for i in order),
                        ORDER_SEP.join(str(i) for i in range(1, len(events) + 1)),
                        ORDER_SEP.join(str(i) for i in order),
                        ORDER_SEP.join(str(s) for s in fab_scenes),
                        ORDER_SEP.join(str(s) for s in syu_scenes),
                        f"{kendall_tau(order):.4f}",
                        f"{lis_ratio(order):.4f}",
                        anon["nonlinear_category"],
                        anon.get("syuzhet_notes", ""),
                        anon.get("fabula_notes", ""),
                        # Compact: no newlines, so the row stays one logical CSV line.
                        json.dumps(anon["story_graph"], separators=(",", ":"),
                                   ensure_ascii=False) if anon.get("story_graph") else "",
                    ])
                    out.flush()
            except (FilmError, json.JSONDecodeError) as exc:
                failed.append((f"{story_id} ({title})", str(exc)))
                print(f"[{index}/{total}] {story_id} FAILED: {exc}")
                continue
            done.append(story_id)
            print(f"[{index}/{total}] {story_id} done ({len(anon['events_chronological'])} events)")
    finally:
        if out is not None:
            out.close()

    if not check_only and done:
        with KEY_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["story_id", "original_title", "release_year"])
            for index, (title, year) in enumerate(FILMS, 1):
                w.writerow([f"S{index:02d}", title, year])

    print()
    print(f"Summary: {len(done)}/{total} succeeded, {len(failed)} failed.")
    for name, reason in failed:
        print(f"  FAILED  {name}: {reason}")
    if not check_only and done:
        print(f"Wrote {OUT_CSV} ({len(done)} rows) and {KEY_CSV}.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
