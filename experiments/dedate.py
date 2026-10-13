"""
dedate.py -- strip absolute time anchors from the event list shown to a solver.

    python dedate.py                 # build experiments/dedated_events.json
    python dedate.py --audit         # report what the built file still contains

Why this exists. The corpus deliberately permits absolute time anchors in bullets
("Years earlier", "Day 218", "Decades later") so that a bullet reads correctly in
either ordering -- see the corpus notes in README.md. That is right for the
corpus and wrong for the measurement: a solver can sort the bullets cold, and the
no-text control showed it does, at tau = 0.833 against 0.877 with the telling.

So the fix belongs on the evaluation side, not in the corpus. This produces a
parallel event list with the dating removed and everything else -- who, what, to
whom, where -- untouched, and leaves `movies_anon/` alone.

Rewriting is done by a model because temporal anchoring is expressed too many
ways for a regex to catch ("the following autumn", "by then", "as a boy", "still
in uniform"). A regex is used afterwards, as an *audit*, to report residue rather
than to do the work. Three deterministic guards run before anything is written:
the event count must match, no bullet may become empty or a duplicate, and the
proper nouns in each bullet must survive, so the rewrite cannot quietly change
who the event is about.
"""
from __future__ import annotations

import argparse, json, os, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tasks as T
from judge import judge_call, parse_json_obj
from runner import APIError, load_corpus

HERE = Path(__file__).parent
OUT = HERE / "dedated_events.json"

SYS = ('You remove time information from story event descriptions. Reply with a single '
       'JSON object and nothing else: {"1": "...", "2": "..."} with one rewritten string '
       'per input number, and exactly as many entries as you were given.')

INSTR = """Rewrite each event below so that it no longer says WHEN it happens, while
changing nothing else.

REMOVE: dates and years; day/week/month/year counters ("Day 218", "Five years later");
relative anchors ("earlier", "later", "afterwards", "by then", "that night", "the next
morning", "the following autumn", "eventually", "finally", "already", "still", "now",
"once more", "again", "for the last time"); era markers ("between the wars", "after the
collapse", "centuries hence"); and age or life-stage anchors that fix position in a
lifetime ("as a boy", "as a young man", "decades on", "now old").

KEEP EXACTLY AS THEY ARE: every character name, place name and object; who does what to
whom; the outcome; causal words that are not temporal ("because", "so that", "in order
to"); and ordinary past-tense narration.

Do not merge, split, reorder, summarise or explain. One rewritten sentence per input.
If an event contains no time information, return it unchanged.

EVENTS
"""

# Audit only. Deliberately blunt: it is meant to over-report so residue is visible.
ANCHOR = re.compile(
    r"\b("
    r"day\s+\d+|days\s+\d+|year\s+\d+|in\s+\d{4}|\d{4}s?"
    r"|earlier|later|afterwards?|subsequently|previously|beforehand"
    r"|next\s+(?:day|morning|night|week|month|year)"
    r"|that\s+(?:day|morning|night|evening|week|year)"
    r"|the\s+following\s+\w+|by\s+then|since\s+then|until\s+then"
    r"|decades?|centuries|century|years\s+(?:on|before|after)"
    r"|as\s+a\s+(?:boy|girl|child|young\s+man|young\s+woman)"
    r"|between\s+the\s+wars|eventually|finally|already|no\s+longer"
    r")\b", re.I)


def audit(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in ANCHOR.finditer(text)})


# Capitalised words that ARE time anchors, so the rewrite is allowed to drop them.
# Without this the guard fires on exactly the words the rewrite is supposed to
# remove -- "Christmas Eve", "New Year", "Tuesday" -- and rejects good output.
CALENDAR = {
    "christmas", "christmases", "eve", "new", "year", "years", "easter",
    "thanksgiving", "halloween", "passover", "ramadan", "diwali", "hanukkah",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "spring", "summer", "autumn", "winter", "fall",
    "day", "days", "night", "morning", "afternoon", "evening", "midnight", "noon",
}


def proper_nouns(text: str) -> set[str]:
    """Capitalised non-sentence-initial tokens -- the names that must survive.

    Calendar terms are excluded: they are capitalised, but they are time anchors,
    which is precisely what the rewrite is asked to remove.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    return {w for i, w in enumerate(words)
            if i and w[0].isupper() and w.lower() not in CALENDAR}


def rewrite_story(job, key):
    g = job["gold"]
    ev = T.split_events(g["fabula"])
    out = {"story_id": g["story_id"], "n": len(ev)}
    try:
        listing = "\n".join(f"{i}. {e}" for i, e in enumerate(ev, 1))
        raw, u = judge_call(job["model_id"], SYS, INSTR + listing, key, max_tokens=8000)
        obj = parse_json_obj(raw)
        new = [str(obj[str(i)]).strip() for i in range(1, len(ev) + 1)]

        if len(new) != len(ev):
            raise ValueError(f"got {len(new)} rewrites for {len(ev)} events")
        if any(not s for s in new):
            raise ValueError("a rewritten event is empty")
        if len(set(new)) != len(new):
            raise ValueError("rewrites collapsed two events into the same text")
        # Recurring names must survive somewhere in the rewrite. Two refinements
        # matter. Position is ignored, because deleting a leading anchor promotes
        # the next word to sentence-initial ("Day 1: Nolan sees" -> "Nolan sees")
        # and a position-sensitive check reads that as a loss. And a token that
        # occurs in only one event of the story is not a recurring name -- it is
        # usually the era marker being removed ("After the Collapse") -- so it is
        # allowed to go.
        recurring = {w for w in {x for e in ev for x in proper_nouns(e)}
                     if sum(bool(re.search(rf"\b{re.escape(w)}\b", e)) for e in ev) > 1}
        for i, (a, b) in enumerate(zip(ev, new), 1):
            lost = {w for w in proper_nouns(a) & recurring
                    if not re.search(rf"\b{re.escape(w)}\b", b)}
            if lost:
                raise ValueError(f"event {i} lost proper noun(s) {sorted(lost)}")

        out.update({"status": "ok", "events": new, "cost": u.get("cost") or 0.0,
                    "residue": {i: audit(s) for i, s in enumerate(new, 1) if audit(s)}})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:220]}"})
    return out


def build(model_name, models_path, workers):
    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(models_path, encoding="utf-8"))}
    mid = specs[model_name]["id"]
    rows = load_corpus()

    have = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [{"gold": g, "model_id": mid} for g in rows if g["story_id"] not in have]
    print(f"de-dating with {model_name}: {len(todo)} stor(ies) to do, {len(have)} cached")

    cost, bad = 0.0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(rewrite_story, j, key) for j in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
            except APIError as e:
                print(f"!! aborting: {e}")
                break
            cost += rec.get("cost") or 0.0
            if rec["status"] == "ok":
                have[rec["story_id"]] = rec["events"]
            else:
                bad += 1
                print(f"  {rec['story_id']} FAILED: {rec['error']}")
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  kept={len(have)} fail={bad} ${cost:.4f}", flush=True)
                OUT.write_text(json.dumps(have, indent=1, ensure_ascii=False), encoding="utf-8")
    OUT.write_text(json.dumps(have, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} with {len(have)} stor(ies); spend ${cost:.4f}")


def report():
    if not OUT.exists():
        sys.exit("nothing built yet")
    ded = json.loads(OUT.read_text(encoding="utf-8"))
    rows = {g["story_id"]: g for g in load_corpus()}
    before = after = 0
    n_ev = 0
    examples = []
    for sid, evs in ded.items():
        orig = T.split_events(rows[sid]["fabula"])
        for a, b in zip(orig, evs):
            n_ev += 1
            ha, hb = bool(audit(a)), bool(audit(b))
            before += ha
            after += hb
            if ha and not hb and len(examples) < 6:
                examples.append((sid, a, b))
    print(f"stories de-dated: {len(ded)}/83   events: {n_ev}")
    print(f"events containing a detectable time anchor: "
          f"{before} ({before/n_ev:.1%}) -> {after} ({after/n_ev:.1%})")
    print("\nexamples:")
    for sid, a, b in examples:
        print(f"  [{sid}] - {a}\n         + {b}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Claude Sonnet 5")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.audit:
        return report()
    build(a.model, a.models, a.workers)
    report()


if __name__ == "__main__":
    raise SystemExit(main())
