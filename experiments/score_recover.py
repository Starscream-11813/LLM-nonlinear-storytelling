"""
score_recover.py -- how much of chronology recovery is the telling, and how much
is the annotation dating itself?

    python score_recover.py
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW, RES = HERE / "raw", HERE / "results"
RES.mkdir(exist_ok=True)

ORDER = ["R0_orig_notext", "R1_orig_told", "R2_dedated_notext", "R3_dedated_told"]
LABEL = {"R0_orig_notext": "R0  events as written, shuffled",
         "R1_orig_told": "R1  events as written, told order",
         "R2_dedated_notext": "R2  de-dated, shuffled",
         "R3_dedated_told": "R3  de-dated, told order"}


def frame():
    rows = []
    p = RAW / "recover.jsonl"
    if not p.exists():
        return pd.DataFrame()
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    df = pd.DataFrame(rows)
    df["rank"] = (df.status == "ok").astype(int)
    return (df.sort_values("rank").groupby(["story_id", "model", "cell"], as_index=False)
              .last())


def main():
    df = frame()
    if df.empty:
        print("no records yet")
        return
    ok = df[df.status == "ok"]
    print(f"records {len(df)}  usable {len(ok)}  failed {(df.status=='failed').sum()}")

    full = [m for m, s in ok.groupby("model")["cell"].nunique().items() if s == 4]
    print(f"models with all four cells: {', '.join(full) or 'none yet'}\n")
    sub = ok[ok.model.isin(full)] if full else ok

    print("=== pooled over models with the complete 2x2 ===")
    t = sub.groupby("cell")[["tau", "exact", "adjacent"]].mean().reindex(ORDER).round(4)
    t.index = [LABEL[i] for i in t.index]
    print(t.to_string())
    t.to_csv(RES / "recover_pooled.csv")

    if len(full):
        g = sub.pivot_table(index="model", columns="cell", values=["tau", "exact"])
        print("\n=== what the told order is worth, per model ===")
        rows = []
        for m in full:
            r = {"model": m}
            for k in ("tau", "exact"):
                r[f"{k}: as written (R1-R0)"] = round(
                    g.loc[m, (k, "R1_orig_told")] - g.loc[m, (k, "R0_orig_notext")], 4)
                r[f"{k}: de-dated (R3-R2)"] = round(
                    g.loc[m, (k, "R3_dedated_told")] - g.loc[m, (k, "R2_dedated_notext")], 4)
            rows.append(r)
        d = pd.DataFrame(rows).set_index("model")
        print(d.to_string())
        d.to_csv(RES / "recover_told_order_value.csv")

        print("\n=== absolute scores per model x cell ===")
        flat = sub.pivot_table(index="model", columns="cell",
                               values=["tau", "exact"]).round(4)
        flat = flat.reindex(columns=pd.MultiIndex.from_product([["tau", "exact"], ORDER]))
        print(flat.to_string())
        flat.to_csv(RES / "recover_by_model.csv")

    print("\n=== by nonlinear category (de-dated cells) ===")
    d2 = sub[sub.cell.isin(["R2_dedated_notext", "R3_dedated_told"])]
    if not d2.empty:
        c = d2.pivot_table(index="category", columns="cell",
                           values=["tau", "exact"]).round(3)
        print(c.to_string())
        c.to_csv(RES / "recover_by_category.csv")

    sub.to_csv(RES / "recover_rows.csv", index=False)
    print(f"\nwrote tables to {RES}")


if __name__ == "__main__":
    main()
