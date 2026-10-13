"""
judge_grapharm.py -- score the graph-scaffolding arm (G0 / G1 / G2).

    python judge_grapharm.py --judge "Gemini 3.1 Pro"

The T5 judge pairs one condition against one other. This arm has three, sampled
twice each, and the two comparisons that matter are not symmetric:

    G1 vs G0   does the graph help at all?
    G1 vs G2   does it help for a reason other than handing over the told order?

G2 is an oracle condition -- it is given the gold telling outright -- so it is a
ceiling on the ordering component, not a baseline a system could ship. Reading
G1-vs-G0 without G1-vs-G2 beside it would credit the graph for information the
told order alone supplies.

Alignment and PDS run on every generated story; the pairwise blocks run on the
first replicate only, which is where the budget goes furthest.
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tasks as T
from judge import PAIR_SYS, do_align, do_pds, judge_call, parse_json_obj
from runner import RAW, APIError, append, load_corpus, out_path

HERE = Path(__file__).parent
ABLATIONS = ["A1_no_causal", "A2_no_state", "A3_no_epistemic",
             "A4_no_questions", "A5_no_grounding"]
CONDITIONS = ["G0_fabula_only", "G1_full_graph", "G2_order_only"] + ABLATIONS
# Each ablation is paired against the full graph, never against G0: the question
# an ablation answers is "what does removing this layer cost", which is a
# comparison with the graph that still has it.
PAIRS = ([("G1_full_graph", "G0_fabula_only"), ("G1_full_graph", "G2_order_only")]
         + [("G1_full_graph", a) for a in ABLATIONS])


def read_prose(model, task, rep):
    p = out_path(model, task, rep)
    if not p.exists():
        return {}
    best = {}
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok" and r.get("text"):
            best[r["story_id"]] = r
    return best


def do_pair(job, key):
    """A vs B on the same fabula, judged twice with the positions swapped.

    Disagreement between the two passes is recorded as a tie, so a win has to
    survive being shown in the other order.
    """
    g = job["gold"]
    out = {"story_id": g["story_id"], "generator": job["gen"], "judge": job["judge"],
           "a": job["a_name"], "b": job["b_name"], "replicate": job["replicate"],
           "instrument": "pairwise", "category": g["category"]}
    q = ("Which is the better story overall -- more engaging, better told, "
         "more satisfying to read?")
    try:
        cost, verdicts = 0.0, []
        for x, y in ((job["a"], job["b"]), (job["b"], job["a"])):
            raw, u = judge_call(job["judge_id"], PAIR_SYS,
                                f"{q}\n\nSTORY A:\n{x}\n\nSTORY B:\n{y}", key,
                                max_tokens=800)
            verdicts.append(parse_json_obj(raw).get("winner"))
            cost += u.get("cost") or 0.0
        flip = {"A": "B", "B": "A", "tie": "tie"}
        agreed = verdicts[0] if verdicts[0] == flip.get(verdicts[1], "tie") else "tie"
        out.update({"status": "ok", "cost": cost, "raw_verdicts": verdicts,
                    "winner": {"A": job["a_name"], "B": job["b_name"]}.get(agreed, "tie")})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def _stamped(fn):
    """do_align/do_pds predate replicates and do not record one. Wrap rather than
    change them, so the T5 arm keeps behaving exactly as it did when it was run."""
    def inner(job, key):
        rec = fn(job, key)
        rec["replicate"] = job.get("replicate", 1)
        return rec
    return inner


def done_keys(path, fields):
    if not path.exists():
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.add(tuple(r.get(f) for f in fields))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Gemini 3.1 Pro")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(a.models, encoding="utf-8"))}
    judge_id = specs[a.judge]["id"]
    gold = {g["story_id"]: g for g in load_corpus()}

    gens = sorted({p.name.split("__")[0] for p in RAW.glob("*__G1_full_graph*.jsonl")})
    name_of = {re.sub(r"[^A-Za-z0-9._-]", "_", n): n for n in specs}
    gens = [name_of[g] for g in gens if g in name_of]

    tag = re.sub(r"[^A-Za-z0-9._-]", "_", a.judge)
    paths = {k: RAW / f"gjudge_{tag}__{k}.jsonl" for k in ("align", "pds", "pairwise")}
    seen = {
        "align": done_keys(paths["align"], ("story_id", "generator", "task", "replicate")),
        "pds": done_keys(paths["pds"], ("story_id", "generator", "task", "replicate")),
        "pairwise": done_keys(paths["pairwise"], ("story_id", "generator", "a", "b", "replicate")),
    }

    jobs = []
    for gen in gens:
        for rep in range(1, a.replicates + 1):
            recs = {c: read_prose(gen, c, rep) for c in CONDITIONS}
            for cond, rows in recs.items():
                for sid, r in sorted(rows.items()):
                    base = {"gen": gen, "judge": a.judge, "judge_id": judge_id,
                            "task": cond, "gold": gold[sid], "rec": r, "replicate": rep}
                    if (sid, gen, cond, rep) not in seen["align"]:
                        jobs.append(("align", base))
                    if (sid, gen, cond, rep) not in seen["pds"]:
                        jobs.append(("pds", base))
            if rep == 1:
                for an, bn in PAIRS:
                    for sid in sorted(set(recs[an]) & set(recs[bn])):
                        if (sid, gen, an, bn, rep) in seen["pairwise"]:
                            continue
                        jobs.append(("pairwise", {
                            "gen": gen, "judge": a.judge, "judge_id": judge_id,
                            "gold": gold[sid], "replicate": rep,
                            "a": recs[an][sid]["text"], "b": recs[bn][sid]["text"],
                            "a_name": an, "b_name": bn}))

    print(f"judge={a.judge}  generators={len(gens)}  pending={len(jobs)}")
    fns = {"align": _stamped(do_align), "pds": _stamped(do_pds), "pairwise": do_pair}
    cost, ok, bad = 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fns[k], j, key): k for k, j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            kind = futs[f]
            try:
                rec = f.result()
            except APIError as e:
                print(f"!! aborting: {e}")
                raise
            append(paths[kind], rec)
            cost += rec.get("cost") or 0.0
            ok += rec["status"] == "ok"
            bad += rec["status"] != "ok"
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ok={ok} fail={bad} ${cost:.4f}", flush=True)
    print(f"done. ok={ok} failed={bad} spend=${cost:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
