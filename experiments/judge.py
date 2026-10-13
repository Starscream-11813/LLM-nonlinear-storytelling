"""
judge.py -- score the prose arm (T5) with an LLM judge.

    python judge.py --judge "Gemini 3.1 Pro" [--pds] [--pairwise]

Three instruments, matched to question type exactly as the study specifies
(pairwise for "is A better than B", Likert only for PDS, alignment for order):

  align    prose -> the told order of the fabula events, so a written story
           yields the same permutation the T1/T2 tasks yield directly.
  pds      the five Psychological Depth Scale components, 1-5.
  pairwise cued vs free telling of the same fabula, judged twice with the
           positions swapped; a disagreement between the two passes is a tie.

Alignment is an *extractor*, so its error propagates into every prose tau.
`--judge2` re-aligns a subsample with a second judge and the two are compared
in score_prose.py -- the study's Stage 0.5 gate, measured rather than assumed.
"""
import argparse, json, os, random, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tasks as T
from runner import (RAW, APIError, append, call, content_of, load_corpus,
                    out_path)

HERE = Path(__file__).parent
_lock = threading.Lock()

PDS_SYS = ('You rate stories on the Psychological Depth Scale. Reply with a single JSON '
           'object and nothing else: {"authenticity":n,"empathy":n,"engagement":n,'
           '"emotion_provocation":n,"narrative_complexity":n} with each n an integer 1-5.')
PAIR_SYS = ('You compare two stories built from the same events. Reply with a single JSON '
            'object and nothing else: {"winner":"A"|"B"|"tie"}.')


def read_prose(model, task):
    p = out_path(model, task)
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


def judge_call(judge_id, system, user, key, max_tokens=1200, temperature=0.0):
    r = call(judge_id, system, user, key, max_tokens=max_tokens,
             temperature=temperature, extra={"reasoning": {"effort": "low"}})
    return content_of(r), (r.get("usage") or {})


def parse_json_obj(raw):
    txt = re.sub(r"```(?:json)?", " ", str(raw))
    i = txt.index("{")
    depth = 0
    for j, ch in enumerate(txt[i:], i):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(txt[i:j + 1])
    raise ValueError(f"no JSON object: {str(raw)[:160]}")


def do_align(job, key):
    gen, judge_name, judge_id, task = job["gen"], job["judge"], job["judge_id"], job["task"]
    g, rec = job["gold"], job["rec"]
    events = T.split_events(g["fabula"])
    out = {"story_id": g["story_id"], "generator": gen, "judge": judge_name,
           "task": task, "instrument": "align", "category": g["category"]}
    try:
        raw, u = judge_call(judge_id, T.SOLVE_SYS,
                            T.prompt_align(events, rec["text"]), key, max_tokens=4000)
        order, mode = T.parse_permutation(raw, len(events))
        out.update({"status": "ok", "order": order, "parse_mode": mode,
                    "cost": u.get("cost"), "word_count": rec.get("word_count")})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def do_pds(job, key):
    g, rec = job["gold"], job["rec"]
    out = {"story_id": g["story_id"], "generator": job["gen"], "judge": job["judge"],
           "task": job["task"], "instrument": "pds", "category": g["category"]}
    try:
        raw, u = judge_call(job["judge_id"], PDS_SYS, f"STORY:\n{rec['text']}", key,
                            max_tokens=1000)
        d = parse_json_obj(raw)
        out.update({"status": "ok", "cost": u.get("cost"),
                    **{f"pds_{k}": d.get(k) for k in
                       ("authenticity", "empathy", "engagement",
                        "emotion_provocation", "narrative_complexity")}})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def do_pairwise(job, key):
    """Cued telling vs the same model's free telling of the same fabula.
    Judged twice with the positions swapped; disagreement is recorded as a tie,
    so presentation-order bias cannot manufacture a win."""
    g, a, b = job["gold"], job["cued"], job["free"]
    out = {"story_id": g["story_id"], "generator": job["gen"], "judge": job["judge"],
           "task": "T5_vs_T5b", "instrument": "pairwise", "category": g["category"]}
    q = ("Which is the better story overall -- more engaging, better told, "
         "more satisfying to read?")
    try:
        cost = 0.0
        verdicts = []
        for x, y in ((a, b), (b, a)):
            raw, u = judge_call(job["judge_id"], PAIR_SYS,
                                f"{q}\n\nSTORY A:\n{x}\n\nSTORY B:\n{y}", key,
                                max_tokens=800)
            verdicts.append(parse_json_obj(raw).get("winner"))
            cost += u.get("cost") or 0.0
        flip = {"A": "B", "B": "A", "tie": "tie"}
        agreed = verdicts[0] if verdicts[0] == flip.get(verdicts[1], "tie") else "tie"
        out.update({"status": "ok", "cost": cost, "raw_verdicts": verdicts,
                    "winner": {"A": "cued", "B": "free"}.get(agreed, "tie")})
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return out


def done_ids(path, keyfields):
    if not path.exists():
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out.add(tuple(r.get(k) for k in keyfields))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="Gemini 3.1 Pro")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--generators", default=None)
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--pds", action="store_true")
    ap.add_argument("--pairwise", action="store_true")
    ap.add_argument("--subsample", type=int, default=None,
                    help="align only N stories per generator (judge-noise check)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if not (a.align or a.pds or a.pairwise):
        a.align = a.pds = a.pairwise = True

    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(a.models, encoding="utf-8"))}
    judge_id = specs[a.judge]["id"]
    gold = {g["story_id"]: g for g in load_corpus()}
    gens = ([s.strip() for s in a.generators.split(",")] if a.generators
            else sorted({p.name.split("__")[0] for p in RAW.glob("*__T5*.jsonl")}))
    gens = [g for g in specs if re.sub(r"[^A-Za-z0-9._-]", "_", g) in
            {re.sub(r"[^A-Za-z0-9._-]", "_", x) for x in gens}]

    jobs = []
    for gen in gens:
        cued, free = read_prose(gen, "T5_prose_cued"), read_prose(gen, "T5b_prose_free")
        for task, recs in (("T5_prose_cued", cued), ("T5b_prose_free", free)):
            ids = sorted(recs)
            if a.subsample:
                ids = ids[:a.subsample]
            for sid in ids:
                base = {"gen": gen, "judge": a.judge, "judge_id": judge_id,
                        "task": task, "gold": gold[sid], "rec": recs[sid]}
                if a.align:
                    jobs.append(("align", base))
                if a.pds:
                    jobs.append(("pds", base))
        if a.pairwise:
            for sid in sorted(set(cued) & set(free)):
                jobs.append(("pairwise", {"gen": gen, "judge": a.judge, "judge_id": judge_id,
                                          "gold": gold[sid], "cued": cued[sid]["text"],
                                          "free": free[sid]["text"]}))

    paths = {k: RAW / f"judge_{re.sub(r'[^A-Za-z0-9._-]', '_', a.judge)}__{k}.jsonl"
             for k in ("align", "pds", "pairwise")}
    seen = {"align": done_ids(paths["align"], ("story_id", "generator", "task")),
            "pds": done_ids(paths["pds"], ("story_id", "generator", "task")),
            "pairwise": done_ids(paths["pairwise"], ("story_id", "generator"))}
    fns = {"align": do_align, "pds": do_pds, "pairwise": do_pairwise}

    todo = []
    for kind, job in jobs:
        k = ((job["gold"]["story_id"], job["gen"], job.get("task"))
             if kind != "pairwise" else (job["gold"]["story_id"], job["gen"]))
        if k not in seen[kind]:
            todo.append((kind, job))
    print(f"judge={a.judge}  generators={len(gens)}  pending judgements={len(todo)}")

    cost, n_ok, n_bad = 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fns[k], j, key): k for k, j in todo}
        for i, f in enumerate(as_completed(futs), 1):
            kind = futs[f]
            try:
                rec = f.result()
            except APIError as e:
                print(f"!! aborting: {e}")
                raise
            append(paths[kind], rec)
            cost += rec.get("cost") or 0.0
            n_ok += rec["status"] == "ok"
            n_bad += rec["status"] != "ok"
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  ok={n_ok} fail={n_bad} ${cost:.4f}", flush=True)
    print(f"judging done. ok={n_ok} failed={n_bad} spend=${cost:.4f}")


if __name__ == "__main__":
    main()
