"""
analysis.py -- the secondary cuts that turn four tables into a result section.

    python analysis.py

Reads experiments/results/per_row.csv (written by score.py) and writes the
breakdowns the write-up argues from: which structures are hard, whether
comprehension and generation move together, whether open weights trail closed
ones, where the classifier confuses itself, and what any of it cost.
"""
import json
from pathlib import Path

import pandas as pd

import metrics as M
from score import BANDS

HERE = Path(__file__).parent
RES = HERE / "results"
def _load_rosters():
    specs = []
    for p in sorted(HERE.glob("models*.json")):
        specs += json.load(open(p, encoding="utf-8"))
    return specs


META = {m["name"]: m for m in _load_rosters()}
# The headline roster: no baseline/reference pseudo-rows, and no decoding variants.
REAL = [n for n, m in META.items() if not m.get("variant")]
VARIANTS = [n for n, m in META.items() if m.get("variant")]


def effort_ablation(df):
    """Each variant model against the identical model at the default setting."""
    pairs = [(META[v]["base"], v) for v in VARIANTS if META[v].get("base") in META]
    rows = []
    for base, var in pairs:
        for task, col in [("T1_transpose_cued", "band_hit"), ("T1_transpose_cued", "tau"),
                          ("T3_reconstruct", "tau_rec"), ("T3_reconstruct", "exact"),
                          ("T4_classify", "hit")]:
            d = df[(df.task == task) & (df.valid == 1)]
            if col not in d or d.empty:
                continue
            lo = d[d.model == base][col]
            hi = d[d.model == var][col]
            if lo.empty or hi.empty:
                continue
            shared = set(d[d.model == base].story_id) & set(d[d.model == var].story_id)
            lo = d[(d.model == base) & d.story_id.isin(shared)][col]
            hi = d[(d.model == var) & d.story_id.isin(shared)][col]
            rows.append({"model": base, "task": task, "metric": col, "n": len(shared),
                         "low_effort": lo.mean(), "high_effort": hi.mean(),
                         "delta": hi.mean() - lo.mean()})
    t = pd.DataFrame(rows)
    if not t.empty:
        t.round(4).to_csv(RES / "breakdown_reasoning_effort.csv", index=False)
    return t.round(4)


def _tag(df):
    df = df.copy()
    df["access"] = df["model"].map(lambda m: META.get(m, {}).get("access", "reference"))
    df["lab"] = df["model"].map(lambda m: META.get(m, {}).get("lab", "-"))
    df["tier"] = df["model"].map(lambda m: META.get(m, {}).get("tier", "-"))
    return df


def per_category(df):
    """T1 broken out by requested structure: the per-structure profile the study
    asks for, since averaging tau across structures is meaningless (reverse
    chronology targets -1, chronicle targets +1)."""
    d = df[(df.task == "T1_transpose_cued") & (df.valid == 1) & df.model.isin(REAL)]
    t = d.pivot_table(index="category", columns=None,
                      values=["tau", "lis", "band_hit", "agree_with_gold",
                              "scene_contiguity"], aggfunc="mean")
    band = d.pivot_table(index="model", columns="category", values="band_hit",
                         aggfunc="mean")
    tau = d.pivot_table(index="model", columns="category", values="tau", aggfunc="mean")
    t.to_csv(RES / "breakdown_category_overall.csv")
    band.round(3).to_csv(RES / "breakdown_category_bandhit.csv")
    tau.round(3).to_csv(RES / "breakdown_category_tau.csv")
    return t.round(3), band.round(3), tau.round(3)


def gold_reference_by_category(gold_rows):
    r = pd.DataFrame(gold_rows).groupby("category")[["tau", "lis"]].agg(["mean", "min", "max"])
    return r.round(3)


def access_split(df):
    """Open weights vs closed API, per task. Reported because 'can an open model
    do this' is the question a lab without a budget actually has."""
    d = _tag(df[df.model.isin(REAL)])
    out = {}
    for task, cols in [("T1_transpose_cued", ["tau", "band_hit", "agree_with_gold",
                                              "scene_contiguity"]),
                       ("T2_transpose_free", ["tau", "near_chronological", "lis"]),
                       ("T3_reconstruct", ["tau_rec", "exact"]),
                       ("T4_classify", ["hit"])]:
        sub = d[(d.task == task) & (d.valid == 1)]
        if sub.empty:
            continue
        cols = [c for c in cols if c in sub]
        out[task] = sub.groupby("access")[cols].mean().round(4)
    return out


def comprehension_vs_generation(df):
    """Does a model that can *read* nonlinearity also *write* it? One number per
    model on each axis, then the correlation across models."""
    d = df[df.model.isin(REAL) & (df.valid == 1)]
    gen = d[d.task == "T1_transpose_cued"].groupby("model")["band_hit"].mean()
    genT = d[d.task == "T1_transpose_cued"].groupby("model")["tau"].mean()
    comp = d[d.task == "T3_reconstruct"].groupby("model")["tau_rec"].mean()
    cls = d[d.task == "T4_classify"].groupby("model")["hit"].mean()
    free = d[d.task == "T2_transpose_free"].groupby("model")["tau"].mean()
    t = pd.DataFrame({"gen_band_hit": gen, "gen_tau": genT, "recover_tau_rec": comp,
                      "classify_acc": cls, "free_tau": free}).dropna()
    t.round(4).to_csv(RES / "breakdown_axes.csv")
    return t.round(4), t.corr(method="spearman").round(3)


def difficulty(df):
    """Is reconstruction harder for more scrambled tellings? Bin by the gold tau
    of the telling the model was shown."""
    d = df[(df.task == "T3_reconstruct") & (df.valid == 1) & df.model.isin(REAL)].copy()
    if d.empty:
        return pd.DataFrame()
    d["told_tau_bin"] = pd.cut(d["tau_told"], [-1.01, -0.3, 0.2, 0.5, 0.75, 1.01],
                               labels=["<=-0.3 (reversed)", "-0.3..0.2", "0.2..0.5",
                                       "0.5..0.75", ">0.75 (near-chrono)"])
    t = d.groupby("told_tau_bin", observed=True).agg(
        n=("tau_rec", "size"), tau_rec=("tau_rec", "mean"), exact=("exact", "mean"))
    by_cat = d.groupby("category").agg(n=("tau_rec", "size"),
                                       tau_rec=("tau_rec", "mean"),
                                       exact=("exact", "mean"))
    t.round(4).to_csv(RES / "breakdown_reconstruct_difficulty.csv")
    by_cat.round(4).to_csv(RES / "breakdown_reconstruct_by_category.csv")
    return t.round(4), by_cat.round(4)


def classify_confusion(df):
    """Where the models' narratological labels go wrong, pooled across models."""
    d = df[(df.task == "T4_classify") & (df.valid == 1) & df.model.isin(REAL)]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    conf = pd.crosstab(d["category"], d["pred"], normalize="index").round(3)
    per_cat = d.groupby("category")["hit"].agg(["size", "mean"]).round(3)
    conf.to_csv(RES / "breakdown_classify_confusion.csv")
    per_cat.to_csv(RES / "breakdown_classify_per_category.csv")
    return conf, per_cat


def band_miss_reason(df, bands):
    """A band miss can mean two opposite things, and the difference is the whole
    result: the telling is not structured enough, or it is structured *more
    perfectly than any human work of that category*. Split every miss into which
    coordinate went outside and in which direction.
    """
    d = df[(df.task == "T1_transpose_cued") & (df.valid == 1) & df.model.isin(REAL)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()

    def why(r):
        b = bands.get(r["category"])
        if not b:
            return "no_band"
        if b["tau_min"] <= r["tau"] <= b["tau_max"] and b["lis_min"] <= r["lis"] <= b["lis_max"]:
            return "hit"
        bits = []
        if r["tau"] > b["tau_max"]:
            bits.append("tau_too_chronological")
        if r["tau"] < b["tau_min"]:
            bits.append("tau_too_inverted")
        if r["lis"] > b["lis_max"]:
            bits.append("lis_too_monotone")
        if r["lis"] < b["lis_min"]:
            bits.append("lis_too_fragmented")
        return "+".join(bits)

    d["miss"] = d.apply(why, axis=1)
    by_cat = pd.crosstab(d["category"], d["miss"], normalize="index").round(3)
    by_cat.to_csv(RES / "breakdown_band_miss.csv")

    # Mechanical execution: did the model hand back the textbook permutation?
    mech = d.groupby("category").agg(
        n=("tau", "size"),
        exact_reverse=("reverse", "mean"),
        exact_identity=("identity", "mean"),
        mean_tau=("tau", "mean"), mean_lis=("lis", "mean"))
    gold = df[df.model == "[reference] human (gold)"].groupby("category").agg(
        human_tau=("tau", "mean"), human_lis=("lis", "mean"))
    mech = mech.join(gold).round(3)
    mech.to_csv(RES / "breakdown_mechanical.csv")
    return by_cat, mech


def common_subset(df, task, cols):
    """Same table, restricted to the stories *every* model answered.

    Needed because refusal is not uniform: a model that declines the hardest
    13 stories is otherwise credited with a mean taken over an easier corpus
    than its neighbours. The restricted table is the fair head-to-head; the
    difference between the two is itself the size of that confound.
    """
    d = df[(df.task == task) & df.model.isin(REAL)]
    if d.empty:
        return pd.DataFrame(), 0
    answered = d[d.valid == 1].groupby("model")["story_id"].apply(set)
    if answered.empty:
        return pd.DataFrame(), 0
    shared = set.intersection(*answered.tolist())
    sub = d[(d.valid == 1) & d.story_id.isin(shared)]
    cols = [c for c in cols if c in sub]
    return sub.groupby("model")[cols].mean().round(4), len(shared)


def spend(df):
    d = _tag(df[df.model.isin(REAL)])
    t = d.groupby("model").agg(rows=("cost", "size"), cost_usd=("cost", "sum"),
                               med_latency_s=("latency_s", "median"),
                               mean_completion_tokens=("completion_tokens", "mean"))
    t.round(4).to_csv(RES / "breakdown_cost.csv")
    return t.round(4)


def main():
    df = pd.read_csv(RES / "per_row.csv")

    cat, band, tau = per_category(df)
    print("\n=== T1 by requested structure (all models pooled) ===")
    print(cat.to_string())
    print("\n=== T1 band-hit rate per model x structure ===")
    print(band.to_string())

    miss, mech = band_miss_reason(df, BANDS)
    if not miss.empty:
        print("\n=== why a telling misses the human band (row-normalised) ===")
        print(miss.to_string())
        print("\n=== mechanical execution vs the human band ===")
        print(mech.to_string())

    print("\n=== open weights vs closed API ===")
    for task, t in access_split(df).items():
        print(f"\n-- {task}")
        print(t.to_string())

    axes, corr = comprehension_vs_generation(df)
    print("\n=== the four axes, per model ===")
    print(axes.sort_values("gen_band_hit", ascending=False).to_string())
    print("\n=== Spearman correlation between axes (across models) ===")
    print(corr.to_string())

    diff, bycat = difficulty(df)
    if not diff.empty:
        print("\n=== T3 recoverability vs how scrambled the telling was ===")
        print(diff.to_string())
        print("\n-- T3 by gold category")
        print(bycat.to_string())

    conf, percat = classify_confusion(df)
    if not conf.empty:
        print("\n=== T4 confusion (row = gold, col = predicted, row-normalised) ===")
        print(conf.to_string())
        print("\n-- T4 accuracy per gold category")
        print(percat.to_string())

    print("\n=== restricted to the stories every model answered ===")
    for task, cols in [("T1_transpose_cued", ["tau", "band_hit", "agree_with_gold"]),
                       ("T2_transpose_free", ["tau", "near_chronological"]),
                       ("T3_reconstruct", ["tau_rec", "exact"]),
                       ("T4_classify", ["hit"])]:
        t, n = common_subset(df, task, cols)
        if not t.empty:
            print(f"\n-- {task}  (n={n} stories)")
            print(t.to_string())
            t.to_csv(RES / f"common_subset_{task}.csv")

    eff = effort_ablation(df)
    if not eff.empty:
        print("\n=== reasoning effort: low (default here) vs high, same stories ===")
        print(eff.to_string(index=False))

    print("\n=== spend and latency ===")
    print(spend(df).to_string())


if __name__ == "__main__":
    main()
