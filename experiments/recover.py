"""
recover.py -- what does the telling actually contribute to chronology recovery?

    python recover.py --only "Claude Opus 5,Gemini 3.1 Pro,Grok 4.5,Llama 4 Maverick"

A 2x2 over one task. The solver is always shown the same n events and always asked
for the chronological order; only two things vary.

                      | events as written     | events de-dated
    ------------------+-----------------------+---------------------
    shuffled, no order| R0                    | R2   <- the floor
    in told order     | R1  (= the T3 task)   | R3   <- the real task

R1 - R0 is what the *told order* is worth on top of bullets that partly date
themselves. R3 - R2 is what it is worth once that shortcut is removed. The gap
between those two differences is the size of the artefact.

All four cells are run here rather than reusing the earlier T3 numbers, so every
comparison is within-experiment and no difference can come from a prompt that
drifted between runs.

Scoring is exact match first and Kendall tau second. Tau is inflatable by
date-sorting; exact match is not, because one misplaced event costs the item.
"""
import argparse, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import metrics as M
import tasks as T
from judge import judge_call
from runner import RAW, APIError, append, load_corpus

HERE = Path(__file__).parent
DEDATED = HERE / "dedated_events.json"

CELLS = {                       # (use de-dated events, present in told order)
    "R0_orig_notext":    (False, False),
    "R1_orig_told":      (False, True),
    "R2_dedated_notext": (True, False),
    "R3_dedated_told":   (True, True),
}

SYS = ("You are a narrative analyst. You reconstruct the true chronological order of a "
       "story's events. Reply with a single line of space-separated numbers and nothing "
       "else.")


def run_one(job, key):
    g, cell = job["gold"], job["cell"]
    dedated, told = CELLS[cell]
    ev = job["dedated"] if dedated else T.split_events(g["fabula"])
    n = len(ev)
    out = {"story_id": g["story_id"], "category": g["category"], "cell": cell,
           "model": job["model"], "n_events": n, "dedated": dedated, "told": told}
    try:
        if told:
            shown = list(g["gold_order"])          # the telling, in bullet form
            preamble = ("SCENES, IN THE ORDER THE STORY TELLS THEM\n"
                        + "\n".join(f"{k}. {ev[e-1]}" for k, e in enumerate(shown, 1)))
        else:
            rng = random.Random(f"none|{g['story_id']}")
            shown = list(range(1, n + 1))
            rng.shuffle(shown)
            preamble = ("SCENES, IN NO PARTICULAR ORDER\n"
                        + "\n".join(f"{k}. {ev[e-1]}" for k, e in enumerate(shown, 1)))

        ask = (f"\n\nWork out the order in which these events actually happened in the "
               f"story world, earliest first.\nOutput exactly {n} space-separated numbers: "
               f"a permutation of {' '.join(str(i) for i in range(1, n+1))}.\n"
               f"Each number refers to a scene above. Output the numbers only.")
        raw, u = judge_call(job["model_id"], SYS, preamble + ask, key, max_tokens=6000)
        order, mode = T.parse_permutation(raw, n)
        chrono = [shown[i - 1] for i in order]
        out.update({"status": "ok", "tau": M.kendall_tau(chrono),
                    "exact": int(chrono == sorted(chrono)),
                    "adjacent": sum(chrono[i+1] > chrono[i] for i in range(n-1)) / (n-1),
                    "parse_mode": mode, "cost": u.get("cost") or 0.0})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def done(path):
    if not path.exists():
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.add((r["story_id"], r["model"], r["cell"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, help="comma-separated model names")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(a.models, encoding="utf-8"))}
    names = [s.strip() for s in a.only.split(",") if s.strip()]
    ded = json.loads(DEDATED.read_text(encoding="utf-8"))
    rows = [g for g in load_corpus() if g["story_id"] in ded]
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]

    path = RAW / "recover.jsonl"
    seen = done(path)
    jobs = [{"gold": g, "cell": c, "model": nm, "model_id": specs[nm]["id"],
             "dedated": ded[g["story_id"]]}
            for nm in names for c in cells for g in rows
            if (g["story_id"], nm, c) not in seen]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(names)} model(s) x {len(cells)} cell(s) x {len(rows)} stories "
          f"-> {len(jobs)} pending")

    cost, ok, bad = 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, j, key) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
            except APIError as e:
                print(f"!! aborting: {e}")
                raise
            append(path, rec)
            cost += rec.get("cost") or 0.0
            ok += rec["status"] == "ok"
            bad += rec["status"] != "ok"
            if i % 40 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ok={ok} fail={bad} ${cost:.4f}", flush=True)
    print(f"done. ok={ok} failed={bad} spend=${cost:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
