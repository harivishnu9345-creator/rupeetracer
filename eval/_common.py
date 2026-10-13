"""Shared helpers for eval scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

import config as C
from data import schemas as S

MODEL_LABELS = {
    "A_rules": "A: Rules",
    "B_behavior_anomaly": "B: + Behavior/Anomaly",
    "C_plus_graph": "C: + Graph",
    "D_plus_provenance": "D: + Provenance",
}


def pred_path(model: str, pred_dir: Optional[Path] = None) -> Path:
    return Path(pred_dir or C.EVAL_OUT) / f"predictions_{model}.csv"


def load_all_predictions(pred_dir: Optional[Path] = None, models=None) -> Dict[str, pd.DataFrame]:
    """Load whichever of the four prediction files exist; warn about missing ones."""
    out = {}
    for m in models or C.PREDICTION_MODELS:
        p = pred_path(m, pred_dir)
        if p.exists():
            out[m] = S.load_predictions(p)
        else:
            print(f"[warn] missing {p.name}; skipping {m}")
    return out


def df_to_md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def save_fig(fig, stem: str, out_dir: Optional[Path] = None) -> Path:
    out_dir = Path(out_dir or C.EVAL_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"{stem}.html"
    fig.write_html(str(html), include_plotlyjs="cdn")
    try:  # PNG only if kaleido is installed
        fig.write_image(str(out_dir / f"{stem}.png"), scale=2)
    except Exception:  # noqa: BLE001
        pass
    return html
