"""Recall and precision by amount bucket (AMOUNT_BUCKETS) for each model.

The recall-by-bucket chart is the project's headline result: it shows that
threshold rules miss the sub-threshold fragments that provenance recovers.

Run:  python -m eval.buckets [--pred-dir DIR] [--out-dir DIR]
Out:  buckets.csv, buckets.md, buckets_recall.html, buckets_precision.html
Buckets are [low, high). Action-level flag = HOLD/BLOCK/ESCALATE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import config as C
from eval import _common as H
from eval import metrics as M


def bucket_table(preds, truth=None) -> pd.DataFrame:
    truth = M.load_truth() if truth is None else truth
    rows = []
    for name, pred in preds.items():
        df = M.align(pred, truth, name)
        idx = df["amount"].map(C.bucket_index)
        for b in range(len(C.AMOUNT_BUCKETS)):
            d = df[idx == b]
            tp = int((d["flag"] & (d["label"] == 1)).sum())
            fp = int((d["flag"] & (d["label"] == 0)).sum())
            npos = int((d["label"] == 1).sum())
            nflag = tp + fp
            rows.append(dict(model=name, bucket=C.bucket_label(b), n_txn=len(d), n_pos=npos, flagged=nflag, tp=tp, fp=fp,
                             recall=tp / npos if npos else float("nan"),
                             precision=tp / nflag if nflag else float("nan")))
    return pd.DataFrame(rows)


def make_chart(df: pd.DataFrame, metric: str) -> go.Figure:
    fig = go.Figure()
    for m, g in df.groupby("model", sort=False):
        fig.add_bar(x=g["bucket"], y=g[metric], name=H.MODEL_LABELS.get(m, m),
                    text=g[metric].round(3), textposition="outside")
    fig.update_layout(barmode="group", template="plotly_white", yaxis=dict(range=[0, 1.1], title=metric),
                      xaxis_title="Transaction amount bucket",
                      title=f"{metric.capitalize()} by amount bucket (test split, action-level flag)")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, default=C.EVAL_OUT)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out_dir or a.pred_dir
    preds = H.load_all_predictions(a.pred_dir)
    if not preds:
        raise SystemExit("no prediction files found")
    df = bucket_table(preds)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "buckets.csv", index=False)
    (out / "buckets.md").write_text("# Recall / precision by amount bucket (test split)\n\n" + H.df_to_md(df))
    H.save_fig(make_chart(df, "recall"), "buckets_recall", out)
    H.save_fig(make_chart(df, "precision"), "buckets_precision", out)
    print(H.df_to_md(df))
    return df


if __name__ == "__main__":
    main()
