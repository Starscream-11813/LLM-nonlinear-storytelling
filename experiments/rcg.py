"""
rcg.py -- retrospective coherence gain and reveal surprisal.

    python rcg.py --solver "Grok 4.5" --segmenter "Gemini 3.1 Pro"

The framework's named novel measure, and the one thing in it that has never been
computed. For a designated reveal r:

    RCG = tau_rec(full text) - tau_rec(text up to just before r)

on the *same* event subset, so the only difference between the two scores is how
much of the story the solver was allowed to read. Paired with a guess task at the
cut point -- can the solver name the withheld cause before it lands? -- this puts
each story on the framework's diagnostic plane:

    low guess accuracy + high RCG   a well-made enigma: unguessable, then inevitable
    high guess accuracy + high RCG  a transparent telling: nothing was withheld
    low guess accuracy + low RCG    a confusing mess: withheld, never paid off
    high guess accuracy + low RCG   a broken reveal: contradicts what was clear

This is the measure that can explain why cued nonlinear tellings lose their
pairwise comparison against the model's own chronological telling. Losing while
landing upper-left would be a judge artefact; losing while landing lower-left
means the models produce confusion without payoff, which is a finding.

Relation to prior work: Bai et al. (arXiv 2507.13841) measure a similar
surprise/coherence pair reference-lessly, by simulating readers guessing a
culprit in detective fiction. This corpus has a gold chronology and gold reveal
annotations, so tau_rec here is a real reconstruction score against a known
answer rather than a proxy, and it applies to all seven structural categories
rather than to whodunits.

Three model calls per story, plus one segmentation call:

  segment   which sentence first narrates each event (needs the aligner's skill,
            so it uses the extractor already validated at r = 0.895)
  pre       reconstruct the chronology of the prefix events, seeing the prefix
  post      reconstruct the same subset, seeing everything
  guess     name the withheld cause, seeing only the prefix

The solver is drawn from a family that generated none of the stories.
"""
import argparse, json, os, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import metrics as M
import tasks as T
from judge import judge_call, parse_json_obj
from runner import RAW, APIError, append, load_corpus, out_path

HERE = Path(__file__).parent
CONDITIONS = ["G0_fabula_only", "G1_full_graph", "G2_order_only"]
MIN_PREFIX = 3          # below this, a tau on the prefix is noise

SEG_SYS = ('You map events to the sentence that first narrates them. Reply with a single '
           'JSON object and nothing else: {"1": s, "2": s, ...} where each key is an event '
           'number and each value is the 1-based index of the sentence in which that event '
           'is first narrated. Every event must appear exactly once.')
SOLVE_SYS = ("You are a narrative analyst. You reconstruct the true chronological order of "
             "story events from a telling that may present them out of order. Reply with a "
             "single line of space-separated event numbers, earliest first, and nothing else.")
GUESS_SYS = ('You are a reader partway through a story, working out what is being withheld. '
             'Reply with a single JSON object and nothing else: {"choice": n} where n is the '
             'number of the option you think is correct.')


def sentences(text):
    """Split prose into sentences. Deliberately simple and deterministic, so the
    same story always cuts at the same place across conditions and reruns."""
    parts = re.split(r'(?<=[.!?])["”\')]*\s+', text.replace("\n", " "))
    return [s.strip() for s in parts if s.strip()]


def designated_reveal(gold):
    """The story's climactic withheld explanation, and the question it answers.

    Where a graph carries several REVEALS edges, take the one the human telling
    places latest: that is the payoff the structure is built around.
    """
    g = gold["story_graph"]
    pos = {e: p + 1 for p, e in enumerate(gold["gold_order"])}
    raised = {q: int(n["id"][1:]) for n in g["events"] for q in n["raises"]}
    resolved = {q: int(n["id"][1:]) for n in g["events"] for q in n["resolves"]}
    qtext = {q: t for n in g["events"] for q, t in n["questions"].items()}
    best = None
    for e in g["edges"]:
        if e["type"] != "REVEALS":
            continue
        i, j = int(e["from"][1:]), int(e["to"][1:])
        qs = [q for q, a in raised.items() if a == j and resolved.get(q) == i]
        if not qs:
            continue
        if best is None or pos[i] > pos[best[0]]:
            best = (i, j, qs[0], qtext.get(qs[0], ""))
    return best


def run_story(job, key):
    """Segment, solve twice, guess once. Returns one record."""
    gold, text, cond, model = job["gold"], job["text"], job["condition"], job["model"]
    ev = T.split_events(gold["fabula"])
    n = len(ev)
    out = {"story_id": gold["story_id"], "model": model, "condition": cond,
           "category": gold["category"], "replicate": job.get("replicate", 1)}
    try:
        rev = designated_reveal(gold)
        if not rev:
            out.update({"status": "skipped", "reason": "graph designates no reveal"})
            return out
        r_ev, asker, qid, qtext = rev
        out.update({"reveal_event": r_ev, "question": qtext})

        sents = sentences(text)
        seg_raw, u0 = judge_call(job["seg_id"], SEG_SYS,
                                 f"EVENTS\n{T._numbered(ev)}\n\nSTORY, ONE SENTENCE PER LINE\n"
                                 + "\n".join(f"{i}. {s}" for i, s in enumerate(sents, 1)),
                                 key, max_tokens=6000)
        seg = {int(k): int(v) for k, v in parse_json_obj(seg_raw).items()}
        cost = u0.get("cost") or 0.0
        if r_ev not in seg:
            out.update({"status": "failed", "error": "segmenter omitted the reveal event",
                        "cost": cost})
            return out

        cut = max(1, min(len(sents), seg[r_ev]))          # sentence the reveal lands in
        prefix_ids = sorted(e for e, s in seg.items() if s < cut)
        out.update({"reveal_sentence": cut, "n_sentences": len(sents),
                    "reveal_told_frac": round((cut - 1) / max(1, len(sents)), 3),
                    "n_prefix_events": len(prefix_ids)})
        if len(prefix_ids) < MIN_PREFIX:
            out.update({"status": "skipped", "cost": cost,
                        "reason": f"only {len(prefix_ids)} events before the reveal"})
            return out

        prefix_text = " ".join(sents[:cut - 1])
        rng = random.Random(f"{gold['story_id']}|{model}|{cond}")
        shown = prefix_ids[:]
        rng.shuffle(shown)
        listing = "\n".join(f"{k}. {ev[e - 1]}" for k, e in enumerate(shown, 1))
        ask = (f"EVENTS (in no particular order)\n{listing}\n\n"
               f"Put these {len(shown)} events into the order in which they actually "
               f"happened in the story world, earliest first. Output exactly {len(shown)} "
               f"space-separated numbers from the list above.")

        def solve(story_text, label):
            raw, u = judge_call(job["solver_id"], SOLVE_SYS,
                                f"STORY\n{story_text}\n\n{ask}", key, max_tokens=5000)
            order, _ = T.parse_permutation(raw, len(shown))
            chrono = [shown[i - 1] for i in order]        # solver's claimed chronology
            return M.kendall_tau([sorted(prefix_ids).index(e) + 1 for e in chrono]), \
                (u.get("cost") or 0.0)

        tau_pre, c1 = solve(prefix_text, "pre")
        tau_post, c2 = solve(text, "post")
        cost += c1 + c2

        # Guess task. Distractors are drawn from events the prefix has NOT already
        # narrated, so every option is equally unseen -- otherwise the solver can
        # eliminate options simply by remembering it has read them.
        unseen = [e for e in range(1, n + 1) if e != r_ev and e not in prefix_ids]
        pool = unseen or [e for e in range(1, n + 1) if e != r_ev]
        rng2 = random.Random(f"guess|{gold['story_id']}|{model}|{cond}")
        distract = rng2.sample(pool, k=min(4, len(pool)))
        options = distract + [r_ev]
        rng2.shuffle(options)
        opts = "\n".join(f"{i}. {ev[e - 1]}" for i, e in enumerate(options, 1))
        g_raw, u3 = judge_call(job["solver_id"], GUESS_SYS,
                               f"STORY SO FAR\n{prefix_text}\n\nOPEN QUESTION\n{qtext}\n\n"
                               f"OPTIONS\n{opts}\n\nWhich option answers the open question?",
                               key, max_tokens=3000)
        pick = int(parse_json_obj(g_raw).get("choice", 0))
        cost += u3.get("cost") or 0.0

        out.update({"status": "ok", "tau_pre": tau_pre, "tau_post": tau_post,
                    "rcg": tau_post - tau_pre,
                    "guess_correct": int(1 <= pick <= len(options)
                                         and options[pick - 1] == r_ev),
                    "n_options": len(options), "cost": cost})
        return out
    except Exception as e:
        out.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:250]}"})
        return out


def read_prose(model, task, rep=1):
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
            best[r["story_id"]] = r["text"]
    return best


def done(path):
    if not path.exists():
        return set()
    out = set()
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") in ("ok", "skipped"):
            out.add((r["story_id"], r["model"], r["condition"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="Grok 4.5",
                    help="held out from every generating family")
    ap.add_argument("--segmenter", default="Gemini 3.1 Pro")
    ap.add_argument("--models", default=str(HERE / "models.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only this many jobs")
    a = ap.parse_args()

    key = os.environ["OPENROUTER_API_KEY"]
    specs = {m["name"]: m for m in json.load(open(a.models, encoding="utf-8"))}
    solver_id, seg_id = specs[a.solver]["id"], specs[a.segmenter]["id"]
    gold = {g["story_id"]: g for g in load_corpus()}

    gens = sorted({p.name.split("__")[0] for p in RAW.glob("*__G1_full_graph.jsonl")})
    name_of = {re.sub(r"[^A-Za-z0-9._-]", "_", n): n for n in specs}
    gens = [name_of[g] for g in gens if g in name_of]

    path = RAW / f"rcg_{re.sub(r'[^A-Za-z0-9._-]', '_', a.solver)}.jsonl"
    seen = done(path)

    jobs = []
    for model in gens:
        for cond in CONDITIONS:
            for sid, text in read_prose(model, cond).items():
                if (sid, model, cond) not in seen:
                    jobs.append({"gold": gold[sid], "text": text, "condition": cond,
                                 "model": model, "solver_id": solver_id, "seg_id": seg_id})
    # The human telling, as the reference point the plane is read against. It is
    # the corpus's event bullets in told order, not prose, so it is labelled
    # separately rather than pooled with the generated stories.
    sids = sorted({j["gold"]["story_id"] for j in jobs})
    for sid in sids:
        if (sid, "[reference] human (gold)", "H_gold_telling") not in seen:
            g = gold[sid]
            jobs.append({"gold": g, "text": " ".join(T.split_events(g["syuzhet"])),
                         "condition": "H_gold_telling", "model": "[reference] human (gold)",
                         "solver_id": solver_id, "seg_id": seg_id})

    if a.limit:
        jobs = jobs[:a.limit]
    print(f"solver={a.solver} segmenter={a.segmenter}  pending={len(jobs)}")
    cost, ok, skip, bad = 0.0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_story, j, key) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
            except APIError as e:
                print(f"!! aborting: {e}")
                raise
            append(path, rec)
            cost += rec.get("cost") or 0.0
            ok += rec["status"] == "ok"
            skip += rec["status"] == "skipped"
            bad += rec["status"] == "failed"
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  ok={ok} skip={skip} fail={bad} ${cost:.4f}",
                      flush=True)
    print(f"done. ok={ok} skipped={skip} failed={bad} spend=${cost:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
