"""Ablation: compare the four prediction files (A rules -> D + provenance).

Run:  python -m eval.ablation [--pred-dir DIR] [--out-dir DIR]
Out:  ablation.csv, ablation.md, ablation.html (plotly)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

import config as C
from eval import _common as H
from eval import metrics as M

TABLE_COLS = ["model", "precision", "recall", "f1", "fpr", "friction_rate", "pr_auc",
              "scenario_recall", "det_latency_median_min", "det_latency_p95_min",
              "latency_ms_p50", "latency_ms_p95", "tp", "fp", "fn", "tn"]


def build_table(preds, truth=None) -> pd.DataFrame:
    df = M.summarize(preds, truth)[TABLE_COLS].copy()
    df.insert(1, "label", df["model"].map(H.MODEL_LABELS))
    return df


def make_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
                      ("friction_rate", "Friction rate (legit)"), ("scenario_recall", "Scenario recall")]:
        fig.add_bar(x=df["label"], y=df[col], name=name, text=df[col].round(3), textposition="outside")
    fig.update_layout(barmode="group", title="Ablation: A rules -> B behavior/anomaly -> C graph -> D provenance (test split)",
                      yaxis=dict(title="value", range=[0, 1.1]), legend_title="metric", template="plotly_white")
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
    df = build_table(preds)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "ablation.csv", index=False)
    (out / "ablation.md").write_text(
        "# Ablation (test split)\n\nflag = HOLD/BLOCK/ESCALATE. friction = STEP_UP/HOLD/BLOCK on legitimate txns. "
        "Latencies: detection in minutes (detected scenarios only), engine in ms.\n\n" + H.df_to_md(df))
    H.save_fig(make_chart(df), "ablation", out)
    print(H.df_to_md(df[["label", "precision", "recall", "f1", "fpr", "friction_rate", "pr_auc", "scenario_recall"]]))
    return df


if __name__ == "__main__":
    main()
