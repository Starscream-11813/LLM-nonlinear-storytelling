"""
score_prose.py -- tables for the prose arm (T5 / T5b) and the judge itself.

    python score_prose.py

Prose tau is *judge-extracted*, not read off a permutation the model handed us,
so it is silver where T1/T2 are gold. Every table here carries that distinction,
and `--judge2` output (if present) is turned into an explicit inter-judge
agreement number rather than a footnote.
"""
import json, re
from pathlib import Path

import pandas as pd

import metrics as M
from runner import RAW, load_corpus

HERE = Path(__file__).parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
GOLD = {g["story_id"]: g for g in load_corpus()}
BANDS = M.build_bands([{"category": g["category"], "tau": M.kendall_tau(g["gold_order"]),
                        "lis": M.lis_ratio(g["gold_order"])} for g in GOLD.values()])
META = {m["name"]: m for m in json.load(open(HERE / "models.json", encoding="utf-8"))}


def load(kind, judge=None):
    pat = f"judge_*__{kind}.jsonl" if judge is None else \
          f"judge_{re.sub(r'[^A-Za-z0-9._-]', '_', judge)}__{kind}.jsonl"
    rows = []
    for p in RAW.glob(pat):
        for line in open(p, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def align_rows(judge=None):
    out, seen = [], set()
    for r in sorted(load("align", judge), key=lambda x: x.get("status") != "ok"):
        k = (r["story_id"], r["generator"], r["task"], r["judge"])
        if k in seen or r.get("status") != "ok":
            continue
        seen.add(k)
        g = GOLD[r["story_id"]]
        t = r["order"]
        tau, lis = M.kendall_tau(t), M.lis_ratio(t)
        _, contig = M.scene_sequence(t, g["gold_scene_order"])
        out.append({"model": r["generator"], "judge": r["judge"], "task": r["task"],
                    "story_id": r["story_id"], "category": g["category"],
                    "tau": tau, "lis": lis, "identity": int(M.is_identity(t)),
                    "near_chronological": int(tau >= 0.95),
                    "scene_contiguity": contig,
                    "band_hit": M.in_band(BANDS, g["category"], tau, lis),
                    "agree_with_gold": M.kendall_tau(
                        [g["gold_order"].index(x) + 1 for x in t]),
                    "word_count": r.get("word_count")})
    return pd.DataFrame(out)


def pds_rows(judge=None):
    cols = ["authenticity", "empathy", "engagement", "emotion_provocation",
            "narrative_complexity"]
    out, seen = [], set()
    for r in load("pds", judge):
        k = (r["story_id"], r["generator"], r["task"], r["judge"])
        if k in seen or r.get("status") != "ok":
            continue
        seen.add(k)
        vals = {c: r.get(f"pds_{c}") for c in cols}
        if any(v is None for v in vals.values()):
            continue
        out.append({"model": r["generator"], "task": r["task"], "story_id": r["story_id"],
                    "category": r["category"], **vals,
                    "pds_mean": sum(vals.values()) / len(vals)})
    return pd.DataFrame(out)


def pairwise_rows(judge=None):
    out, seen = [], set()
    for r in load("pairwise", judge):
        k = (r["story_id"], r["generator"], r["judge"])
        if k in seen or r.get("status") != "ok":
            continue
        seen.add(k)
        out.append({"model": r["generator"], "story_id": r["story_id"],
                    "category": r["category"], "winner": r["winner"]})
    return pd.DataFrame(out)


def interjudge(a_df):
    """Same story, same generator, two judges -> how much of the prose tau is the
    story and how much is the extractor."""
    if a_df.empty or a_df["judge"].nunique() < 2:
        return None
    piv = a_df.pivot_table(index=["model", "task", "story_id"], columns="judge",
                           values="tau")
    piv = piv.dropna()
    if len(piv) < 5 or piv.shape[1] < 2:
        return None
    j1, j2 = piv.columns[:2]
    return {"n_pairs": int(len(piv)), "judge_a": j1, "judge_b": j2,
            "pearson_tau": float(piv[j1].corr(piv[j2])),
            "mean_abs_diff": float((piv[j1] - piv[j2]).abs().mean()),
            "mean_a": float(piv[j1].mean()), "mean_b": float(piv[j2].mean())}


def main():
    al, pd_, pw = align_rows(), pds_rows(), pairwise_rows()
    if al.empty:
        print("no alignment judgements yet")
        return

    ij = interjudge(al)
    primary = al["judge"].value_counts().idxmax()
    a = al[al.judge == primary]

    t = a.pivot_table(index="model", columns="task",
                      values=["tau", "lis", "identity", "near_chronological",
                              "band_hit", "scene_contiguity", "word_count"],
                      aggfunc="mean")
    t.columns = [f"{b.replace('T5_prose_cued','cued').replace('T5b_prose_free','free')}_{x}"
                 for x, b in t.columns]
    t = t.round(4)
    if "cued_tau" in t and "free_tau" in t:
        t["tau_shift_cued_minus_free"] = (t["cued_tau"] - t["free_tau"]).round(4)
    t.to_csv(RES / "table_T5_prose.csv")
    print("\n=== T5 prose arm (order extracted by judge: %s) ===" % primary)
    print(t.to_string())

    if not pd_.empty:
        p = pd_.pivot_table(index="model", columns="task", values="pds_mean",
                            aggfunc="mean").round(3)
        p.columns = [c.replace("T5_prose_cued", "cued_pds").replace("T5b_prose_free",
                                                                   "free_pds")
                     for c in p.columns]
        comp = pd_[pd_.task == "T5_prose_cued"].groupby("model")[
            ["authenticity", "empathy", "engagement", "emotion_provocation",
             "narrative_complexity"]].mean().round(3)
        out = p.join(comp)
        out.to_csv(RES / "table_T5_pds.csv")
        print("\n=== Psychological Depth Scale (1-5; components are the cued condition) ===")
        print(out.to_string())

    if not pw.empty:
        w = pw.groupby("model")["winner"].value_counts(normalize=True).unstack().fillna(0)
        w["n"] = pw.groupby("model").size()
        w = w.round(3)
        w.to_csv(RES / "table_T5_pairwise.csv")
        print("\n=== Pairwise: structured telling vs the model's own free telling ===")
        print(w.to_string())

    if ij:
        json.dump(ij, open(RES / "interjudge.json", "w"), indent=1)
        print("\n=== extractor (judge) noise on prose tau ===")
        print(f"  {ij['judge_a']} vs {ij['judge_b']}: n={ij['n_pairs']} "
              f"r={ij['pearson_tau']:.3f} mean|diff|={ij['mean_abs_diff']:.3f} "
              f"(means {ij['mean_a']:.3f} / {ij['mean_b']:.3f})")

    al.to_csv(RES / "prose_align_rows.csv", index=False)
    if not pd_.empty:
        pd_.to_csv(RES / "prose_pds_rows.csv", index=False)
    if not pw.empty:
        pw.to_csv(RES / "prose_pairwise_rows.csv", index=False)


if __name__ == "__main__":
    main()
