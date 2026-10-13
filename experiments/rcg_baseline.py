"""
rcg_baseline.py -- the no-text null model for every recoverability score.

    python rcg_baseline.py --solver "Grok 4.5"

Every reconstruction score in this project -- T3's tau_rec, and both halves of
RCG -- asks a solver to put a story's events into chronological order. In each
case the solver is shown the events as the corpus writes them, and the corpus's
authoring rules explicitly permit absolute time anchors ("Years earlier", "Day
218", "Decades later") because they make a bullet readable in either ordering.

That permission has a consequence nobody has measured: a solver may be able to
sort the bullets **without reading the story at all**. If so, tau_rec is partly a
measure of how self-dating the annotation is, not of how recoverable the telling
is, and RCG has almost no headroom to move because the pre-reveal score already
starts near the ceiling.

This script measures that directly. Same solver, same shuffled event list, same
instruction -- and no story. Whatever it scores is the floor that every other
recoverability number has to be read against.
"""
import argparse, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import metrics as M
import tasks as T
from judge import judge_call
from rcg import SOLVE_SYS
from runner import RAW, APIError, append, load_corpus

HERE = Path(__file__).parent


def run_one(job, key):
    g = job["gold"]
    ev = T.split_events(g["fabula"])
    n = len(ev)
    out = {"story_id": g["story_id"], "category": g["category"], "n_events": n,
           "solver": job["solver"]}
    try:
        rng = random.Random(f"none|{g['story_id']}")
        shown = list(range(1, n + 1))
        rng.shuffle(shown)
        listing = "\n".join(f"{k}. {ev[e - 1]}" for k, e in enumerate(shown, 1))
        ask = (f"EVENTS (in no particular order)\n{listing}\n\n"
               f"Put these {n} events into the order in which they actually happened, "
               f"earliest first. Output exactly {n} space-separated numbers.")
        raw, u = judge_call(job["solver_id"], SOLVE_SYS,
                            f"STORY\n(none supplied)\n\n{ask}", key, max_tokens=5000)
        order, _ = T.parse_permutation(raw, n)
        chrono = [shown[i - 1] for i in order]
        out.update({"status": "ok", "tau_none": M.kendall_tau(chrono),
                    "exact": int(chrono == sorted(chrono)),
                    "cost": u.get("cost") or 0.0})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="Grok 4.5")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(a.models, encoding="utf-8"))}
    sid = specs[a.solver]["id"]
    rows = load_corpus(a.limit)

    path = RAW / f"rcg_none_{re.sub(r'[^A-Za-z0-9._-]', '_', a.solver)}.jsonl"
    seen = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                seen.add(r["story_id"])

    jobs = [{"gold": g, "solver": a.solver, "solver_id": sid}
            for g in rows if g["story_id"] not in seen]
    print(f"no-text baseline: solver={a.solver}  pending={len(jobs)}")
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
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ok={ok} fail={bad} ${cost:.4f}", flush=True)
    print(f"done. ok={ok} failed={bad} spend=${cost:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
