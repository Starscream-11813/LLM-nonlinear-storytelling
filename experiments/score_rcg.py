"""
score_rcg.py -- the diagnostic plane, and the tables behind it.

    python score_rcg.py

Two numbers per story:

  guess accuracy at the cut   how predictable the withheld cause was before it landed
  RCG = tau_post - tau_pre    how much the reveal repaired the reader's chronology

Neither is good or bad alone. A story that is unguessable and never resolves is
noise; one that resolves but was obvious was never withholding anything. Only the
pair identifies deliberate, successful confusion, which is why the framework
reports them as a plane rather than a score.

Also emits `figures/rcg_plane.svg`, quadrant-labelled, with the human telling
plotted alongside the generated conditions.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW, RES = HERE / "raw", HERE / "results"
FIG = HERE.parent / "dataset" / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

LABEL = {"G0_fabula_only": "G0 fabula only", "G1_full_graph": "G1 full graph",
         "G2_order_only": "G2 told order (oracle)",
         "H_gold_telling": "H human telling (reference)"}
# Palette slots 1/2/3 + neutral, matching the corpus figures.
COLOR = {"G0_fabula_only": "#52514e", "G1_full_graph": "#2a78d6",
         "G2_order_only": "#eb6834", "H_gold_telling": "#1baf7a"}


def frame():
    rows = []
    for p in RAW.glob("rcg_*.jsonl"):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # keep the last attempt per cell; a retried failure is superseded by its success
    df["rank"] = (df.status == "ok").astype(int)
    df = (df.sort_values("rank").groupby(["story_id", "model", "condition"], as_index=False)
            .last())
    return df


def quadrant(row, rcg_mid, guess_mid):
    if row.guess_correct <= guess_mid and row.rcg > rcg_mid:
        return "well-made enigma"
    if row.guess_correct > guess_mid and row.rcg > rcg_mid:
        return "transparent telling"
    if row.guess_correct <= guess_mid and row.rcg <= rcg_mid:
        return "confusing mess"
    return "broken reveal"


def plane_svg(agg, w=520, h=430):
    """Scatter of condition means on the guess-accuracy / RCG plane."""
    ml, mr, mt, mb = 66, 18, 40, 62
    pw, ph = w - ml - mr, h - mt - mb
    xs = [0.0, 1.0]
    lo = min(-0.05, agg["rcg"].min() - 0.03)
    hi = max(0.20, agg["rcg"].max() + 0.05)

    def X(v):
        return ml + (v - xs[0]) / (xs[1] - xs[0]) * pw

    def Y(v):
        return mt + (1 - (v - lo) / (hi - lo)) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" '
         f'height="{h}" role="img" class="f" aria-label="Retrospective coherence plane">',
         "<style>.f{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#dedcd6}"
         "@media(prefers-color-scheme:dark){:root:where(:not([data-theme='light'])) .f"
         "{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#3a3a38}}"
         ":root[data-theme='dark'] .f{--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;"
         "--grid:#3a3a38}.f text{font-family:ui-sans-serif,system-ui,sans-serif}</style>",
         f'<rect width="{w}" height="{h}" fill="var(--surface)"/>',
         f'<text x="{ml}" y="22" font-size="13" font-weight="600" fill="var(--ink)">'
         f'Was the disorder worth it?</text>']

    gm, rm = 0.5, 0.0
    p.append(f'<line x1="{X(gm):.1f}" y1="{mt}" x2="{X(gm):.1f}" y2="{mt+ph}" '
             f'stroke="var(--grid)" stroke-dasharray="4 4"/>')
    p.append(f'<line x1="{ml}" y1="{Y(rm):.1f}" x2="{ml+pw}" y2="{Y(rm):.1f}" '
             f'stroke="var(--grid)" stroke-dasharray="4 4"/>')
    for tx, ty, anc, txt in [(ml + 6, mt + 14, "start", "well-made enigma"),
                             (ml + pw - 6, mt + 14, "end", "transparent telling"),
                             (ml + 6, mt + ph - 8, "start", "confusing mess"),
                             (ml + pw - 6, mt + ph - 8, "end", "broken reveal")]:
        p.append(f'<text x="{tx}" y="{ty}" font-size="9.5" text-anchor="{anc}" '
                 f'fill="var(--grid)">{txt}</text>')

    for _, r in agg.iterrows():
        c = COLOR.get(r["condition"], "#52514e")
        p.append(f'<circle cx="{X(r.guess_correct):.1f}" cy="{Y(r.rcg):.1f}" r="6" '
                 f'fill="{c}" stroke="var(--surface)" stroke-width="2"/>')
        p.append(f'<text x="{X(r.guess_correct):.1f}" y="{Y(r.rcg)-11:.1f}" font-size="10" '
                 f'text-anchor="middle" fill="var(--ink)">{LABEL.get(r["condition"], "")}</text>')

    p.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="var(--ink2)"/>')
    p.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="var(--ink2)"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        p.append(f'<text x="{X(v):.1f}" y="{mt+ph+16}" font-size="9.5" text-anchor="middle" '
                 f'fill="var(--ink2)">{v:.2f}</text>')
    p.append(f'<text x="{ml+pw/2}" y="{mt+ph+34}" font-size="10.5" text-anchor="middle" '
             f'fill="var(--ink2)">pre-reveal guess accuracy &#8594; more predictable</text>')
    p.append(f'<text transform="translate(16,{mt+ph/2}) rotate(-90)" font-size="10.5" '
             f'text-anchor="middle" fill="var(--ink2)">retrospective coherence gain &#8594;</text>')
    p.append(f'<text x="{ml}" y="{h-8}" font-size="9" fill="var(--grid)">'
             f'dashed lines: chance guessing (1/5) is left of 0.5; zero gain is the horizontal</text>')
    p.append("</svg>")
    return "".join(p)


def main():
    df = frame()
    if df.empty:
        print("no RCG records yet")
        return
    ok = df[df.status == "ok"].copy()
    print(f"records: {len(df)}  usable: {len(ok)}  "
          f"skipped: {(df.status=='skipped').sum()}  failed: {(df.status=='failed').sum()}")
    if (df.status == "skipped").any():
        print("  skip reasons:", dict(Counter(df[df.status == "skipped"].reason)))
    if ok.empty:
        return

    # A skip is usually "the reveal arrived before anything had been withheld",
    # which is a botched reveal rather than missing data. Report the rate per
    # condition instead of letting it disappear into the denominator.
    early = df[df.status.isin(["ok", "skipped"])].copy()
    early["reveal_too_early"] = (early.status == "skipped").astype(int)
    er = early.groupby("condition")["reveal_too_early"].agg(["mean", "size"]).round(3)
    er.index = [LABEL.get(i, i) for i in er.index]
    print("\n=== reveal narrated too early to withhold anything ===")
    print(er.rename(columns={"mean": "rate", "size": "n"}).to_string())
    er.to_csv(RES / "rcg_early_reveal.csv")

    print("\n=== the plane, by condition ===")
    agg = (ok.groupby("condition")
             .agg(n=("rcg", "size"), tau_pre=("tau_pre", "mean"),
                  tau_post=("tau_post", "mean"), rcg=("rcg", "mean"),
                  guess_correct=("guess_correct", "mean"),
                  reveal_told_frac=("reveal_told_frac", "mean"),
                  prefix_events=("n_prefix_events", "mean"))
             .round(4).reset_index())
    show = agg.copy()
    show["condition"] = show["condition"].map(lambda c: LABEL.get(c, c))
    print(show.to_string(index=False))
    agg.to_csv(RES / "rcg_by_condition.csv", index=False)

    print("\n=== quadrant membership (per story, split at guess 0.5 / RCG 0) ===")
    ok["quadrant"] = ok.apply(lambda r: quadrant(r, 0.0, 0.5), axis=1)
    q = pd.crosstab(ok.condition, ok.quadrant, normalize="index").round(3)
    q.index = [LABEL.get(i, i) for i in q.index]
    print(q.to_string())
    q.to_csv(RES / "rcg_quadrants.csv")

    print("\n=== by model (generated conditions only) ===")
    gen = ok[ok.condition != "H_gold_telling"]
    m = gen.groupby("model")[["tau_pre", "tau_post", "rcg", "guess_correct"]].mean().round(4)
    print(m.to_string())
    m.to_csv(RES / "rcg_by_model.csv")

    print("\n=== by nonlinear category (generated only) ===")
    c = gen.groupby("category")[["rcg", "guess_correct", "reveal_told_frac"]].mean().round(4)
    c["n"] = gen.groupby("category").size()
    print(c.to_string())
    c.to_csv(RES / "rcg_by_category.csv")

    (FIG / "rcg_plane.svg").write_text(plane_svg(agg), encoding="utf-8")
    ok.to_csv(RES / "rcg_rows.csv", index=False)
    print(f"\nwrote tables to {RES} and {FIG/'rcg_plane.svg'}")


if __name__ == "__main__":
    main()
