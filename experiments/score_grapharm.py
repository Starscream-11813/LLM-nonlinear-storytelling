"""
score_grapharm.py -- tables for the graph-scaffolding arm.

    python score_grapharm.py

Reads the judge output for G0/G1/G2 and reports, per condition:

  structure   tau / LIS of the told order the judge recovered from the prose,
              and whether it lands inside the human band for the requested
              category (band_hit) -- the measure the arm exists to move
  fidelity    scene contiguity and agreement with the gold telling
  quality     the PDS five-vector
  preference  blind pairwise win rate, positions swapped

Read the two pairwise blocks together. G2 is an oracle -- it is handed the gold
told order -- so G1 losing to G2 bounds how much of the graph's benefit is
ordering information rather than semantics, and G1 beating G0 while merely
matching G2 would mean the graph is worth no more than the permutation.
"""
import json
import re
from pathlib import Path

import pandas as pd

import metrics as M
from runner import RAW, load_corpus

HERE = Path(__file__).parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)

GOLD = {g["story_id"]: g for g in load_corpus()}
BANDS = M.build_bands([{"category": g["category"],
                        "tau": M.kendall_tau(g["gold_order"]),
                        "lis": M.lis_ratio(g["gold_order"])} for g in GOLD.values()])
LABEL = {"G0_fabula_only": "G0 fabula only",
         "G1_full_graph": "G1 full graph",
         "G2_order_only": "G2 told order (oracle)",
         "A1_no_causal": "A1 graph -CAUSES/-FORESHADOWS",
         "A2_no_state": "A2 graph -pre/post",
         "A3_no_epistemic": "A3 graph -learns",
         "A4_no_questions": "A4 graph -questions/-REVEALS",
         "A5_no_grounding": "A5 graph -participants/-location"}


def load(kind):
    rows = []
    for p in RAW.glob(f"gjudge_*__{kind}.jsonl"):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                rows.append(r)
    return rows


def align_frame():
    seen, out = set(), []
    for r in load("align"):
        k = (r["story_id"], r["generator"], r["task"], r.get("replicate", 1))
        if k in seen:
            continue
        seen.add(k)
        g = GOLD[r["story_id"]]
        t = r["order"]
        tau, lis = M.kendall_tau(t), M.lis_ratio(t)
        _, contig = M.scene_sequence(t, g["gold_scene_order"])
        out.append({
            "model": r["generator"], "condition": r["task"],
            "replicate": r.get("replicate", 1), "story_id": r["story_id"],
            "category": g["category"], "tau": tau, "lis": lis,
            "band_hit": M.in_band(BANDS, g["category"], tau, lis),
            "identity": int(M.is_identity(t)),
            "near_chronological": int(tau >= 0.95),
            "scene_contiguity": contig,
            "agree_with_gold": M.kendall_tau([g["gold_order"].index(x) + 1 for x in t]),
            "tau_err": abs(tau - M.kendall_tau(g["gold_order"])),
            "word_count": r.get("word_count"),
        })
    return pd.DataFrame(out)


def pds_frame():
    cols = ["authenticity", "empathy", "engagement", "emotion_provocation",
            "narrative_complexity"]
    seen, out = set(), []
    for r in load("pds"):
        k = (r["story_id"], r["generator"], r["task"], r.get("replicate", 1))
        if k in seen:
            continue
        seen.add(k)
        vals = {c: r.get(f"pds_{c}") for c in cols}
        if any(v is None for v in vals.values()):
            continue
        out.append({"model": r["generator"], "condition": r["task"],
                    "replicate": r.get("replicate", 1), "story_id": r["story_id"],
                    **vals, "pds_mean": sum(vals.values()) / len(vals)})
    return pd.DataFrame(out)


def pairwise_frame():
    seen, out = set(), []
    for r in load("pairwise"):
        k = (r["story_id"], r["generator"], r["a"], r["b"], r.get("replicate", 1))
        if k in seen:
            continue
        seen.add(k)
        out.append({"model": r["generator"], "story_id": r["story_id"],
                    "category": r["category"], "a": r["a"], "b": r["b"],
                    "winner": r["winner"]})
    return pd.DataFrame(out)


def main():
    al, pd_, pw = align_frame(), pds_frame(), pairwise_frame()
    if al.empty:
        print("no graph-arm judgements yet")
        return

    # The ablations ran on a subset of the roster, so a mean pooled over all rows
    # would compare conditions on different model sets -- and the models differ
    # more than the conditions do. Restrict to models carrying every condition
    # before reading any cross-condition difference.
    have = al.groupby("model")["condition"].nunique()
    full = sorted(have[have == al.condition.nunique()].index)
    if full and len(full) < al.model.nunique():
        sub = al[al.model.isin(full)]
        print(f"\n=== structure, by condition, common models only "
              f"({', '.join(full)}; n={len(sub)}) ===")
        c = sub.groupby("condition")[["tau", "band_hit", "agree_with_gold",
                                      "scene_contiguity", "word_count"]].mean().round(4)
        if not pd_.empty:
            c["pds_mean"] = (pd_[pd_.model.isin(full)].groupby("condition")["pds_mean"]
                             .mean().round(3))
        c.index = [LABEL.get(i, i) for i in c.index]
        print(c.to_string())
        c.to_csv(RES / "grapharm_structure_common.csv")

    print(f"\n=== structure, by condition, all rows "
          f"(n={len(al)} across {al.model.nunique()} models; "
          f"NOT comparable across conditions -- see the table above) ===")
    t = al.groupby("condition")[["tau", "lis", "band_hit", "scene_contiguity",
                                 "agree_with_gold", "near_chronological",
                                 "word_count"]].mean().round(4)
    t.index = [LABEL.get(i, i) for i in t.index]
    print(t.to_string())
    t.to_csv(RES / "grapharm_structure.csv")

    print("\n=== structure, per model x condition (band_hit) ===")
    bh = al.pivot_table(index="model", columns="condition", values="band_hit").round(3)
    bh.columns = [LABEL.get(c, c) for c in bh.columns]
    if "G1 full graph" in bh and "G0 fabula only" in bh:
        bh["delta G1-G0"] = (bh["G1 full graph"] - bh["G0 fabula only"]).round(3)
    print(bh.to_string())
    bh.to_csv(RES / "grapharm_bandhit_by_model.csv")

    print("\n=== paired by story: does the graph move the same story? ===")
    piv = al.pivot_table(index=["model", "story_id", "replicate"],
                         columns="condition", values="band_hit")
    if {"G0_fabula_only", "G1_full_graph"} <= set(piv.columns):
        p = piv.dropna(subset=["G0_fabula_only", "G1_full_graph"])
        win = (p["G1_full_graph"] > p["G0_fabula_only"]).sum()
        loss = (p["G1_full_graph"] < p["G0_fabula_only"]).sum()
        print(f"  band_hit  G1 better on {win} pairs, worse on {loss}, "
              f"tied on {len(p) - win - loss} (n={len(p)})")
    tp = al.pivot_table(index=["model", "story_id", "replicate"],
                        columns="condition", values="agree_with_gold")
    if {"G0_fabula_only", "G1_full_graph"} <= set(tp.columns):
        q = tp.dropna(subset=["G0_fabula_only", "G1_full_graph"])
        print(f"  agreement with the gold telling: G0 {q['G0_fabula_only'].mean():+.3f} "
              f"-> G1 {q['G1_full_graph'].mean():+.3f} "
              f"(mean paired delta {(q['G1_full_graph'] - q['G0_fabula_only']).mean():+.3f})")

    if not pd_.empty:
        print("\n=== Psychological Depth Scale, by condition (1-5) ===")
        cols = ["authenticity", "empathy", "engagement", "emotion_provocation",
                "narrative_complexity", "pds_mean"]
        p = pd_.groupby("condition")[cols].mean().round(3)
        p.index = [LABEL.get(i, i) for i in p.index]
        print(p.to_string())
        p.to_csv(RES / "grapharm_pds.csv")

    if not pw.empty:
        print("\n=== blind pairwise, positions swapped ===")
        for (an, bn), sub in pw.groupby(["a", "b"]):
            share = sub["winner"].value_counts(normalize=True)
            n = len(sub)
            print(f"  {LABEL.get(an, an)} vs {LABEL.get(bn, bn)}  (n={n})")
            for k in [an, bn, "tie"]:
                print(f"      {LABEL.get(k, k):<26} {share.get(k, 0):.3f}")
        pw.to_csv(RES / "grapharm_pairwise_rows.csv", index=False)
        w = pw.groupby(["a", "b", "model"])["winner"].value_counts(normalize=True).unstack().fillna(0)
        w.round(3).to_csv(RES / "grapharm_pairwise.csv")

    al.to_csv(RES / "grapharm_align_rows.csv", index=False)
    if not pd_.empty:
        pd_.to_csv(RES / "grapharm_pds_rows.csv", index=False)
    print(f"\nwrote tables to {RES}")


if __name__ == "__main__":
    main()
