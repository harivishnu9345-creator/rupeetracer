"""Latency summary: engine latency_ms (p50/p95/p99) and scenario detection latency.

Run:  python -m eval.latency [--pred-dir DIR] [--out-dir DIR]
Out:  latency.csv, latency.md, latency.html
Detection latency = minutes from a scenario's first label==1 test txn to its first
flagged txn (HOLD/BLOCK/ESCALATE); computed over detected scenarios only, with the
detected share reported as scenario_recall.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
from eval import _common as H
from eval import metrics as M

COLS = ["model", "latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "latency_ms_mean",
        "scenarios_total", "scenarios_detected", "scenario_recall",
        "det_latency_median_min", "det_latency_p95_min"]


def build_table(preds, truth=None) -> pd.DataFrame:
    return M.summarize(preds, truth)[COLS]


def make_chart(df: pd.DataFrame) -> go.Figure:
    lab = df["model"].map(lambda m: H.MODEL_LABELS.get(m, m))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Engine latency per txn (ms)", "Scenario detection latency (min)"))
    fig.add_bar(x=lab, y=df["latency_ms_p50"], name="p50 ms", row=1, col=1)
    fig.add_bar(x=lab, y=df["latency_ms_p95"], name="p95 ms", row=1, col=1)
    fig.add_bar(x=lab, y=df["det_latency_median_min"], name="median min", row=1, col=2)
    fig.add_bar(x=lab, y=df["det_latency_p95_min"], name="p95 min", row=1, col=2)
    fig.update_layout(barmode="group", template="plotly_white", title="Latency (test split)")
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
    df.to_csv(out / "latency.csv", index=False)
    (out / "latency.md").write_text("# Latency summary (test split)\n\n" + H.df_to_md(df))
    H.save_fig(make_chart(df), "latency", out)
    print(H.df_to_md(df))
    return df


if __name__ == "__main__":
    main()
