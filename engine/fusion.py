"""Score fusion: behavior, anomaly, graph, provenance -> one risk_score in [0, 1].

* Trained mode  : non-negative logistic-regression stacker fitted on training-split component scores.
* Fallback mode : fixed weighted average (DEFAULT_WEIGHTS). Used when untrained, and per call when any
                  input is missing / NaN (weights are re-normalised over the inputs that are present).
* Feedback loop : cases.py keeps per-component weight scales (SCALE_MIN..SCALE_MAX) and passes them into
                  score(components, scale). Effective weight = learned weight x scale; nothing is stored here.
* HARD RULE     : provenance is one input of four and cannot push a transaction into HOLD alone.
                  The final score is capped at (HOLD floor - 0.01) + CORROBORATION_GAIN * (m - OTHERS_LOW)+,
                  where m is the strongest of the other components (missing counts as 0). So while every other
                  component is below OTHERS_LOW the score stays under HOLD, and the cap relaxes smoothly (no cliff)
                  as the others corroborate. Applied after scaling, so feedback cannot defeat it.
* Contributions : per-component share of the final score; they sum to risk_score (within rounding).
                  When the cap binds, provenance is credited only with the lift it added over the other components.
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from engine.orchestrator import BAND_FLOORS

log = logging.getLogger("rupeetrace")

COMPONENTS = ["behavior", "anomaly", "graph", "provenance"]
SCALE_MIN, SCALE_MAX = 0.5, 1.5
DEFAULT_WEIGHTS = {"behavior": 0.35, "anomaly": 0.25, "graph": 0.25, "provenance": 0.15}

OTHERS_LOW = 0.30
CORROBORATION_GAIN = 1.5
PROVENANCE_GUARD_CAP = BAND_FLOORS["HOLD"] - 0.01

MIN_FIT_ROWS, MIN_FIT_CLASS = 50, 5
POS_WEIGHT_POWER = 0.5   # positives weighted (neg/pos)**0.5 : tempered class balancing
POS_WEIGHT_MAX = 50.0
W_MAX = 12.0
L2 = 1e-3
ITERS, LR = 800, 0.1
FORMAT_VERSION = 1


def _clean(v) -> float | None:
    """Component score -> float in [0,1], or None if missing / NaN / inf / not numeric."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, x)) if math.isfinite(x) else None


def _sigmoid(z: float) -> float:
    return 0.5 * (1.0 + math.tanh(z / 2.0))


def _scale_for(scale, c: str) -> float:
    try:
        s = float((scale or {}).get(c, 1.0))
    except (TypeError, ValueError, AttributeError):
        return 1.0
    return min(SCALE_MAX, max(SCALE_MIN, s)) if math.isfinite(s) else 1.0


class Fusion:
    def __init__(self, path: str | os.PathLike | None = None, components: list[str] | None = None):
        self.components = list(components) if components else list(COMPONENTS)
        bad = [c for c in self.components if c not in COMPONENTS]
        if bad:
            raise ValueError(f"unknown components: {bad}")
        tot = sum(DEFAULT_WEIGHTS[c] for c in self.components)
        self.base_weights = {c: DEFAULT_WEIGHTS[c] / tot for c in self.components}
        self.path = Path(path) if path else None
        self.weights = dict(self.base_weights)
        self.intercept = 0.0
        self.trained = False
        self.meta: dict = {}
        if self.path and self.path.exists():
            self.load()

    @property
    def mode(self) -> str:
        return "logistic" if self.trained else "weighted_average"

    # ------------------------------------------------------------------ training
    def _untrained(self, reason: str, **extra) -> dict:
        self.weights, self.intercept, self.trained = dict(self.base_weights), 0.0, False
        self.meta = {"trained": False, "reason": reason, **extra}
        return dict(self.meta)

    def fit(self, X, y) -> dict:
        """Fit the stacker. X: (n, len(components)) in self.components order; y: 0/1 labels. Returns metadata."""
        k = len(self.components)
        X = np.asarray(X, dtype=float)
        y = np.nan_to_num(np.asarray(y, dtype=float).ravel(), nan=0.0)
        if X.ndim != 2 or X.shape[1] != k or len(X) != len(y):
            raise ValueError(f"X must be (n, {k}) and match len(y)")
        for j in range(k):  # impute NaN/inf with the column mean of finite values
            col = X[:, j]
            ok = np.isfinite(col)
            col[~ok] = col[ok].mean() if ok.any() else 0.0
        X = np.clip(X, 0.0, 1.0)
        y = (y > 0.5).astype(float)
        n, pos = len(y), int(y.sum())
        neg = n - pos
        if n < MIN_FIT_ROWS or pos < MIN_FIT_CLASS or neg < MIN_FIT_CLASS:
            return self._untrained("too few rows / positives to train", n=n, n_pos=pos)

        sw = np.ones(n)
        sw[y == 1] = min(POS_WEIGHT_MAX, max(1.0, (neg / pos) ** POS_WEIGHT_POWER))
        sw /= sw.mean()
        w, b = np.ones(k), -2.0
        mw, vw, mb, vb = np.zeros(k), np.zeros(k), 0.0, 0.0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, ITERS + 1):  # Adam with projection onto w >= 0 (monotone stacker)
            p = 0.5 * (1.0 + np.tanh((X @ w + b) / 2.0))
            r = sw * (p - y) / n
            gw, gb = X.T @ r + L2 * w, r.sum()
            mw, vw = b1 * mw + (1 - b1) * gw, b2 * vw + (1 - b2) * gw * gw
            mb, vb = b1 * mb + (1 - b1) * gb, b2 * vb + (1 - b2) * gb * gb
            w = np.clip(w - LR * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps), 0.0, W_MAX)
            b -= LR * (mb / (1 - b1 ** t)) / (math.sqrt(vb / (1 - b2 ** t)) + eps)
        if float(w.sum()) < 1e-6:
            return self._untrained("no usable signal in component scores", n=n, n_pos=pos)

        p = np.clip(0.5 * (1.0 + np.tanh((X @ w + b) / 2.0)), 1e-9, 1 - 1e-9)
        logloss = float(-np.mean(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))))
        self.weights = {c: float(w[i]) for i, c in enumerate(self.components)}
        self.intercept, self.trained = float(b), True
        self.meta = {"trained": True, "n": n, "n_pos": pos, "train_weighted_logloss": round(logloss, 6)}
        return dict(self.meta)

    # ------------------------------------------------------------------- scoring
    def effective_weights(self, scale: Mapping | None = None) -> dict:
        return {c: self.weights[c] * _scale_for(scale, c) for c in self.components}

    def score(self, components: Mapping, scale: Mapping | None = None) -> dict:
        """-> {risk_score, contributions, mode, gated, missing}. Never raises on bad input; always within [0, 1]."""
        comps = self.components
        src = components if isinstance(components, Mapping) else {}
        vals = {c: _clean(src.get(c)) for c in comps}
        missing = [c for c in comps if vals[c] is None]
        sc = {c: _scale_for(scale, c) for c in comps}
        logit = self.trained and not missing

        def combine(v: dict) -> tuple[float, dict]:
            if logit:
                z = {c: self.weights[c] * sc[c] * v[c] for c in comps}
                return _sigmoid(self.intercept + sum(z.values())), z
            present = [c for c in comps if vals[c] is not None]
            if not present:
                return 0.0, {c: 0.0 for c in comps}
            den = sum(self.base_weights[c] for c in present)
            z = {c: (self.base_weights[c] * sc[c] * v[c] / den if c in present else 0.0) for c in comps}
            return min(1.0, sum(z.values())), z

        v = {c: (vals[c] if vals[c] is not None else 0.0) for c in comps}
        risk, z = combine(v)
        gated, risk_others = False, 0.0
        if "provenance" in comps and len(comps) > 1:
            others = [c for c in comps if c != "provenance"]
            risk_others, _ = combine({**v, "provenance": 0.0})
            m = max((vals[c] or 0.0) for c in others)
            cap = PROVENANCE_GUARD_CAP + CORROBORATION_GAIN * max(0.0, m - OTHERS_LOW)
            if risk > cap:
                risk, gated = cap, True
        risk = min(1.0, max(0.0, risk))

        contrib = self._contributions(risk, z, gated, risk_others)
        return {"risk_score": round(risk, 6), "contributions": {c: round(x, 6) for c, x in contrib.items()},
                "mode": "logistic" if logit else "weighted_average", "gated": gated, "missing": missing}

    def _contributions(self, risk: float, z: dict, gated: bool, risk_others: float) -> dict:
        comps = self.components
        if risk <= 0.0:
            return {c: 0.0 for c in comps}

        def split(names: list[str], total: float) -> dict:
            s = sum(z[c] for c in names)
            if s <= 1e-12:
                w = {c: self.base_weights[c] for c in names}
                s = sum(w.values())
                return {c: total * w[c] / s for c in names}
            return {c: total * z[c] / s for c in names}

        if gated and "provenance" in comps:
            prov = min(risk, max(0.0, risk - risk_others))
            out = split([c for c in comps if c != "provenance"], risk - prov)
            out["provenance"] = prov
            return {c: out[c] for c in comps}
        return split(comps, risk)

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"version": FORMAT_VERSION, "components": self.components, "trained": self.trained,
                "weights": self.weights, "intercept": self.intercept, "meta": self.meta}

    def save(self) -> Path | None:
        if not self.path:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        os.replace(tmp, self.path)
        return self.path

    def load(self) -> bool:
        """Load persisted weights. Corrupt / mismatched files are ignored (stays in the current state)."""
        try:
            d = json.loads(self.path.read_text())
            if d.get("components") != self.components:
                raise ValueError(f"components mismatch: {d.get('components')}")
            w = {c: float(d["weights"][c]) for c in self.components}
            if not all(math.isfinite(x) and x >= 0 for x in w.values()):
                raise ValueError("bad weights")
            b = float(d.get("intercept", 0.0))
            if not math.isfinite(b):
                raise ValueError("bad intercept")
        except Exception as e:
            log.warning("fusion weights not loaded from %s (%s)", self.path, e)
            return False
        self.trained = bool(d.get("trained", False)) and sum(w.values()) > 1e-6
        self.weights = w if self.trained else dict(self.base_weights)
        self.intercept = b if self.trained else 0.0
        self.meta = d.get("meta", {}) if isinstance(d.get("meta"), dict) else {}
        return True

    def describe(self) -> dict:
        return {"mode": self.mode, "trained": self.trained, "components": self.components,
                "weights": dict(self.weights), "base_weights": dict(self.base_weights),
                "intercept": self.intercept, "others_low": OTHERS_LOW, "corroboration_gain": CORROBORATION_GAIN,
                "provenance_guard_cap": PROVENANCE_GUARD_CAP, "scale_range": [SCALE_MIN, SCALE_MAX], "meta": dict(self.meta)}
