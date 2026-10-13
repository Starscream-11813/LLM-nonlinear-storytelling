"""
score.py -- turn experiments/raw/*.jsonl into the eval tables.

    python score.py            # writes experiments/results/*.csv and prints the tables

Nothing here calls an API. Every number is recomputed from the stored model
replies and the gold corpus, so the tables can be rebuilt at any time and a
disagreement between a table and the raw JSONL is a bug in this file, not a
missing experiment.
"""
import json, random, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import metrics as M
from runner import load_corpus, out_path

HERE = Path(__file__).parent
RAW, RES = HERE / "raw", HERE / "results"
RES.mkdir(exist_ok=True)

GOLD = {r["story_id"]: r for r in load_corpus()}
# Bands are built from tau/LIS recomputed from the gold permutations, not from
# the CSV's 4-dp rounded columns: with the rounded values a gold telling can
# fall a hair outside its own category's band.
BANDS = M.build_bands([{"category": r["category"],
                        "tau": M.kendall_tau(r["gold_order"]),
                        "lis": M.lis_ratio(r["gold_order"])} for r in GOLD.values()])
def _load_rosters():
    """models.json is the headline roster; models_*.json hold variants (a model
    re-run under a different decoding setting) that must be scored the same way
    but kept out of the headline table."""
    specs = []
    for p in sorted(HERE.glob("models*.json")):
        specs += json.load(open(p, encoding="utf-8"))
    return specs


META = {m["name"]: m for m in _load_rosters()}
MODELS = list(META)


def read(model, task):
    """Last successful record per story (a resumed run may hold earlier failures)."""
    p = out_path(model, task)
    if not p.exists():
        return {}
    best = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r["story_id"]
            if r.get("status") == "ok" or k not in best:
                best[k] = r
    return best


REFUSAL = re.compile(
    r"\b(i'?m sorry|i am sorry|i can(?:'|no)?t (?:assist|help|comply)|"
    r"unable to (?:assist|help|comply)|i (?:won'?t|will not) (?:be able to )?(?:assist|help)|"
    r"against my|violates? (?:the )?(?:content|usage) polic)", re.I)


def failure_kind(r):
    """Why a row has no usable answer. Refusals are a model behaviour and belong
    in the table; a truncated reasoning budget is a harness fact and belongs in a
    footnote. Keeping them apart stops one being read as the other."""
    if r.get("status") == "ok":
        return None
    raw = str(r.get("raw") or "")
    err = str(r.get("error") or "")
    if raw and REFUSAL.search(raw[:400]):
        return "refusal"
    if "HTTP" in err or "unreachable" in err or "timeout" in err:
        return "api_error"
    if not raw or "empty content" in err:
        return "empty"
    return "malformed"


def gold_scene_ranks(g, told):
    seq, contig = M.scene_sequence(told, g["gold_scene_order"])
    return M.scene_ranks(seq), contig


# ------------------------------------------------------- classifier calibration
def calibrate_classifier():
    """The structural classifier's accuracy on the 83 gold tellings. This is the
    noise floor of every structure-hit number below, and is reported as such."""
    rows, conf = [], defaultdict(Counter)
    for g in GOLD.values():
        ranks, _ = gold_scene_ranks(g, g["gold_order"])
        pred = M.classify(ranks)
        conf[g["category"]][pred] += 1
        rows.append({"story_id": g["story_id"], "gold": g["category"], "pred": pred,
                     "hit": pred == g["category"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "classifier_calibration.csv", index=False)
    acc = df["hit"].mean()
    dec = df[df["gold"].isin(M.STRUCTURAL_CATEGORIES)]
    return {"overall_acc": acc, "structural_acc": dec["hit"].mean(),
            "per_class": {c: round(df[df.gold == c]["hit"].mean(), 3)
                          for c in M.CATEGORIES if (df.gold == c).any()},
            "confusion": {k: dict(v) for k, v in conf.items()}}


# --------------------------------------------------------------- per-row scoring
def score_ordering(model, task):
    recs, out = read(model, task), []
    for sid, r in recs.items():
        g = GOLD[sid]
        base = {"model": model, "task": task, "story_id": sid,
                "category": g["category"], "source": g["source"],
                "n_events": len(g["gold_order"]),
                "status": r.get("status"), "parse_mode": r.get("parse_mode"),
                "failure_kind": failure_kind(r),
                "latency_s": r.get("latency_s"), "cost": r.get("cost"),
                "completion_tokens": r.get("completion_tokens")}
        if r.get("status") != "ok" or "order" not in r:
            out.append({**base, "valid": 0}); continue
        t = r["order"]
        tau, lis = M.kendall_tau(t), M.lis_ratio(t)
        ranks, contig = gold_scene_ranks(g, t)
        row = {**base, "valid": 1, "tau": tau, "lis": lis,
               "displacement": M.displacement(t),
               "identity": int(M.is_identity(t)), "reverse": int(M.is_reverse(t)),
               "near_chronological": int(tau >= 0.95),
               "scene_contiguity": contig,
               "pred_category": M.classify(ranks),
               "pred_category_event": M.classify(t),
               # how close the model's telling is to the one the human annotator
               # read off the actual work, on the same events
               "agree_with_gold": M.kendall_tau([g["gold_order"].index(x) + 1 for x in t]),
               "tau_gold": g["tau"], "lis_gold": g["lis"],
               "tau_err": abs(tau - g["tau"])}
        if task == "T1_transpose_cued":
            row["band_hit"] = M.in_band(BANDS, g["category"], tau, lis)
            row["struct_hit"] = int(row["pred_category"] == g["category"])
            row["decidable"] = int(g["category"] in M.STRUCTURAL_CATEGORIES)
        out.append(row)
    return out


def score_reconstruct(model):
    recs, out = read(model, "T3_reconstruct"), []
    for sid, r in recs.items():
        g = GOLD[sid]
        base = {"model": model, "task": "T3_reconstruct", "story_id": sid,
                "category": g["category"], "source": g["source"],
                "n_events": len(g["gold_order"]), "status": r.get("status"),
                "failure_kind": failure_kind(r), "tau_told": g["tau"], "latency_s": r.get("latency_s"),
                "cost": r.get("cost"), "completion_tokens": r.get("completion_tokens")}
        if r.get("status") != "ok" or "order" not in r:
            out.append({**base, "valid": 0}); continue
        # The model returns told-positions in claimed chronological order.
        # Mapping each back through the gold told order gives chronological
        # ranks; perfect recovery is 1, 2, ... n.
        chrono = [g["gold_order"][p - 1] for p in r["order"]]
        out.append({**base, "valid": 1,
                    "tau_rec": M.kendall_tau(chrono),
                    "spearman_rec": M.spearman(chrono, list(range(1, len(chrono) + 1))),
                    "exact": int(chrono == sorted(chrono)),
                    "adjacent_acc": sum(1 for i in range(len(chrono) - 1)
                                        if chrono[i + 1] > chrono[i]) / (len(chrono) - 1)})
    return out


def score_classify(model):
    recs, out = read(model, "T4_classify"), []
    for sid, r in recs.items():
        g = GOLD[sid]
        out.append({"model": model, "task": "T4_classify", "story_id": sid,
                    "category": g["category"], "source": g["source"],
                    "status": r.get("status"), "valid": int(r.get("status") == "ok"),
                    "failure_kind": failure_kind(r),
                    "pred": r.get("label"), "hit": int(r.get("label") == g["category"]),
                    "tau_told": g["tau"], "latency_s": r.get("latency_s"),
                    "cost": r.get("cost"), "completion_tokens": r.get("completion_tokens")})
    return out


# ------------------------------------------------------------------- baselines
def baselines(seed=0):
    """Reference rows: what the metrics say for a model that does nothing, a model
    that reverses everything, a coin flip, and the human annotation itself."""
    rng = random.Random(seed)
    rows = []
    for name, make in [
        ("[baseline] identity", lambda n, g: list(range(1, n + 1))),
        ("[baseline] reverse",  lambda n, g: list(range(n, 0, -1))),
        ("[baseline] random",   lambda n, g: rng.sample(range(1, n + 1), n)),
        ("[reference] human (gold)", lambda n, g: list(g["gold_order"])),
    ]:
        for g in GOLD.values():
            n = len(g["gold_order"])
            t = make(n, g)
            tau, lis = M.kendall_tau(t), M.lis_ratio(t)
            ranks, contig = gold_scene_ranks(g, t)
            rows.append({"model": name, "task": "T1_transpose_cued", "story_id": g["story_id"],
                         "category": g["category"], "source": g["source"], "valid": 1,
                         "tau": tau, "lis": lis, "displacement": M.displacement(t),
                         "identity": int(M.is_identity(t)), "reverse": int(M.is_reverse(t)),
                         "near_chronological": int(tau >= 0.95), "scene_contiguity": contig,
                         "pred_category": M.classify(ranks), "pred_category_event": M.classify(t),
                         "agree_with_gold": M.kendall_tau(
                             [g["gold_order"].index(x) + 1 for x in t]),
                         "tau_gold": g["tau"], "lis_gold": g["lis"],
                         "tau_err": abs(tau - g["tau"]),
                         "band_hit": M.in_band(BANDS, g["category"], tau, lis),
                         "struct_hit": int(M.classify(ranks) == g["category"]),
                         "decidable": int(g["category"] in M.STRUCTURAL_CATEGORIES)})
    return rows


# ---------------------------------------------------------------------- tables
def agg_ordering(df, task):
    d = df[df.task == task]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(is_refusal=(d.failure_kind == "refusal").astype(float),
                 is_malformed=(d.failure_kind == "malformed").astype(float))
    g = d.groupby("model")
    ok = d[d.valid == 1].groupby("model")
    # A CHRONICLE target is *supposed* to come back chronological, so counting
    # those as degenerate would charge a model for obeying the instruction.
    nonchron = d[(d.valid == 1) & (d.category != "CHRONICLE")].groupby("model")
    out = pd.DataFrame({
        "n": g.size(),
        "valid_rate": g["valid"].mean(),
        "refusal_rate": g["is_refusal"].mean(),
        "malformed_rate": g["is_malformed"].mean(),
        "tau": ok["tau"].mean(),
        "lis": ok["lis"].mean(),
        "identity_rate": nonchron["identity"].mean(),
        "near_chrono_rate": nonchron["near_chronological"].mean(),
        "scene_contiguity": ok["scene_contiguity"].mean(),
        "agree_with_gold": ok["agree_with_gold"].mean(),
        "tau_err_vs_gold": ok["tau_err"].mean(),
    })
    if task == "T1_transpose_cued":
        dec = d[(d.valid == 1) & (d.decidable == 1)].groupby("model")
        out["band_hit"] = ok["band_hit"].mean()
        out["struct_hit_decidable"] = dec["struct_hit"].mean()
    out["cost_usd"] = g["cost"].sum()
    out["med_latency_s"] = g["latency_s"].median()
    return out.round(4)


def agg_reconstruct(df):
    d = df[df.task == "T3_reconstruct"]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(is_refusal=(d.failure_kind == "refusal").astype(float))
    g, ok = d.groupby("model"), d[d.valid == 1].groupby("model")
    return pd.DataFrame({
        "n": g.size(), "valid_rate": g["valid"].mean(),
        "refusal_rate": g["is_refusal"].mean(),
        "tau_rec": ok["tau_rec"].mean(), "spearman_rec": ok["spearman_rec"].mean(),
        "exact_rate": ok["exact"].mean(), "adjacent_acc": ok["adjacent_acc"].mean(),
        "cost_usd": g["cost"].sum(), "med_latency_s": g["latency_s"].median(),
    }).round(4)


def agg_classify(df):
    d = df[df.task == "T4_classify"]
    if d.empty:
        return pd.DataFrame()
    rows = {}
    for m, sub in d.groupby("model"):
        valid = sub[sub.valid == 1]
        f1s = []
        for c in M.CATEGORIES:
            tp = ((valid.pred == c) & (valid.category == c)).sum()
            fp = ((valid.pred == c) & (valid.category != c)).sum()
            fn = ((valid.pred != c) & (valid.category == c)).sum()
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        rows[m] = {"n": len(sub), "valid_rate": sub["valid"].mean(),
                   "refusal_rate": (sub["failure_kind"] == "refusal").mean(),
                   "accuracy": valid["hit"].mean() if len(valid) else 0.0,
                   "macro_f1": sum(f1s) / len(f1s),
                   "n_distinct_labels": valid["pred"].nunique(),
                   "cost_usd": sub["cost"].sum(),
                   "med_latency_s": sub["latency_s"].median()}
    return pd.DataFrame(rows).T.round(4)


def main():
    rows = []
    for m in MODELS:
        rows += score_ordering(m, "T1_transpose_cued")
        rows += score_ordering(m, "T2_transpose_free")
        rows += score_reconstruct(m)
        rows += score_classify(m)
    rows += baselines()
    df = pd.DataFrame(rows)
    df.to_csv(RES / "per_row.csv", index=False)

    cal = calibrate_classifier()
    json.dump({"classifier_calibration": cal,
               "gold_bands": BANDS}, open(RES / "calibration.json", "w"), indent=1)

    tables = {"T1_transpose_cued": agg_ordering(df, "T1_transpose_cued"),
              "T2_transpose_free": agg_ordering(df, "T2_transpose_free"),
              "T3_reconstruct": agg_reconstruct(df),
              "T4_classify": agg_classify(df)}
    for k, t in tables.items():
        if not t.empty:
            t.to_csv(RES / f"table_{k}.csv")
            print(f"\n=== {k} ===")
            print(t.to_string())

    print("\n=== structural classifier calibrated on the 83 gold tellings ===")
    print(f"overall {cal['overall_acc']:.3f}  |  five decidable classes "
          f"{cal['structural_acc']:.3f}")
    print(" per class:", cal["per_class"])
    return df, tables


if __name__ == "__main__":
    main()
