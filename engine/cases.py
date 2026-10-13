"""Case management + analyst feedback loop.

* A case opens on HOLD / BLOCK / ESCALATE. Transactions sharing a root_txn (lineage) - or, with no
  lineage, a cluster / source-account key - are grouped into ONE case, never one per transaction.
* Status: open | confirmed | false_positive, analyst notes, persisted to JSON (atomic write).
* Feedback:  false_positive -> lower the weight scale of the component that drove the case and the
  segment offset;  confirmed -> raise both and add involved accounts to a watchlist that boosts later scores.
  Adjustments are persisted and every feedback returns before/after.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

from engine.fusion import COMPONENTS, SCALE_MAX, SCALE_MIN
from engine.orchestrator import OFFSET_LIMIT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "state" / "cases.json"
SCALE_STEP = 0.10      # +-10% on the driving component weight per feedback
OFFSET_STEP = 0.02     # +-0.02 on the segment's risk offset per feedback
WATCHLIST_BOOST = 0.12
MAX_TXNS_STORED = 3000
STATUSES = ("open", "confirmed", "false_positive")


def _fresh_adjustments() -> dict:
    return {"component_scale": {c: 1.0 for c in COMPONENTS}, "segment_offsets": {}, "watchlist": {}, "history": []}


class CaseManager:
    def __init__(self, path: str | os.PathLike | None = None, persist: bool = True):
        self.path = Path(path) if path else DEFAULT_PATH
        self.persist = persist
        self._lock = threading.RLock()
        self._dirty, self._last_save = False, 0.0
        self.cases: dict[str, dict] = {}
        self.adjustments = _fresh_adjustments()
        self._by_key: dict[str, str] = {}
        self._seen: set[str] = set()   # txn_ids already recorded -> replays are idempotent
        self._seq = 0
        self._load()

    # --------------------------------------------------------------- persistence
    def _load(self) -> None:
        if not (self.persist and self.path.exists()):
            return
        try:
            d = json.loads(self.path.read_text())
            self.cases, self._seq = d.get("cases", {}), d.get("seq", 0)
            self.adjustments = {**_fresh_adjustments(), **d.get("adjustments", {})}
            self._by_key = {c["group_key"]: cid for cid, c in self.cases.items()}
            self._seen = {t["txn_id"] for c in self.cases.values() for t in c.get("txns", [])}
        except Exception:
            pass

    def flush(self) -> None:
        with self._lock:
            if not self.persist:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"seq": self._seq, "cases": self.cases, "adjustments": self.adjustments}, indent=1, default=str))
            os.replace(tmp, self.path)
            self._dirty, self._last_save = False, time.time()

    def _touch(self, force: bool = False) -> None:
        self._dirty = True
        if force or time.time() - self._last_save > 2.0:
            self.flush()

    def reset(self) -> None:
        with self._lock:
            self.cases, self.adjustments, self._by_key, self._seq = {}, _fresh_adjustments(), {}, 0
            self._seen = set()
            self._touch(force=True)

    # -------------------------------------------------------------------- state
    def scale(self) -> dict:
        return dict(self.adjustments["component_scale"])

    def offsets(self) -> dict:
        return dict(self.adjustments["segment_offsets"])

    def watchlist(self) -> set:
        return set(self.adjustments["watchlist"])

    def watchlist_hit(self, *accounts) -> bool:
        w = self.adjustments["watchlist"]
        return any(a in w for a in accounts)

    # --------------------------------------------------------------- open/update
    @staticmethod
    def group_key(txn: dict, lineage: dict | None, features: dict | None) -> str:
        if lineage and lineage.get("root_txn_id"):
            return f"root:{lineage['root_txn_id']}"
        cid = (features or {}).get("cluster_id")
        return f"cluster:{cid}" if cid else f"acct:{txn['src_account']}"

    def record(self, txn: dict, decision: dict, lineage: dict | None, features: dict | None, driver: str, comps: dict, raw_components: dict) -> str | None:
        """Add a HOLD/BLOCK/ESCALATE decision to its case (creating it if needed). Returns case_id."""
        if not decision.get("opens_case"):
            return None
        with self._lock:
            key = self.group_key(txn, lineage, features)
            cid = self._by_key.get(key)
            if txn["txn_id"] in self._seen and cid:
                return cid
            self._seen.add(txn["txn_id"])
            if cid is None:
                self._seq += 1
                cid = f"CASE-{self._seq:04d}"
                self._by_key[key] = cid
                self.cases[cid] = {
                    "case_id": cid, "group_key": key, "status": "open", "opened_ts": txn["ts"], "last_ts": txn["ts"],
                    "root_txn_id": (lineage or {}).get("root_txn_id"), "root_amount": (lineage or {}).get("root_aggregate_amount"),
                    "origin_type": (lineage or {}).get("origin_type"), "txn_count": 0, "total_amount": 0.0, "accounts": [],
                    "txns": [], "max_risk": 0.0, "max_cluster_risk": 0.0, "actions": {}, "reasons": {}, "driver_totals": {c: 0.0 for c in COMPONENTS},
                    "driver": driver, "segment": decision["segment"], "peak": None, "notes": [], "feedback": []}
            c = self.cases[cid]
            c["txn_count"] += 1
            c["total_amount"] = round(c["total_amount"] + float(txn["amount"]), 2)
            c["last_ts"] = txn["ts"]
            c["actions"][decision["action"]] = c["actions"].get(decision["action"], 0) + 1
            for a in (txn["src_account"], txn["dst_account"]):
                if a not in c["accounts"] and len(c["accounts"]) < 200:
                    c["accounts"].append(a)
            for r in decision["reasons"]:
                c["reasons"][r] = c["reasons"].get(r, 0) + 1
            for k, v in comps.items():
                c["driver_totals"][k] = round(c["driver_totals"].get(k, 0.0) + v, 4)
            c["driver"] = max(c["driver_totals"], key=c["driver_totals"].get)
            if lineage and lineage.get("fragment_count", 0) >= (c.get("fragments") or 0):
                c["fragments"], c["hops"], c["root_amount"] = lineage.get("fragment_count"), lineage.get("hops"), lineage.get("root_aggregate_amount")
            if len(c["txns"]) < MAX_TXNS_STORED:
                c["txns"].append({"txn_id": txn["txn_id"], "ts": txn["ts"], "src": txn["src_account"], "dst": txn["dst_account"],
                                  "amount": float(txn["amount"]), "risk": decision["risk_score"], "action": decision["action"],
                                  "device_id": txn.get("device_id")})
            c["max_risk"] = max(c["max_risk"], decision["risk_score"])
            c["max_cluster_risk"] = max(c["max_cluster_risk"], decision["cluster_risk"])
            if c["peak"] is None or decision["risk_score"] >= c["peak"]["risk_score"]:
                c["peak"] = {"txn_id": txn["txn_id"], "risk_score": decision["risk_score"], "cluster_risk": decision["cluster_risk"],
                             "components": raw_components, "account_type": decision.get("account_type"), "kyc_tier": decision.get("kyc_tier"),
                             "base_risk_score": decision["base_risk_score"], "action": decision["action"]}
            self._touch()
            return cid

    # ------------------------------------------------------------------- queries
    def list(self, status: str | None = None) -> list[dict]:
        with self._lock:
            out = [self.summary(c) for c in self.cases.values() if status in (None, c["status"])]
        return sorted(out, key=lambda c: (c["status"] != "open", -c["max_risk"], -c["txn_count"]))

    @staticmethod
    def summary(c: dict) -> dict:
        keys = ("case_id", "status", "root_txn_id", "root_amount", "txn_count", "total_amount", "max_risk", "max_cluster_risk",
                "driver", "segment", "opened_ts", "last_ts", "actions", "fragments", "hops")
        s = {k: c.get(k) for k in keys}
        s["account_count"] = len(c["accounts"])
        s["top_reason"] = max(c["reasons"], key=c["reasons"].get) if c["reasons"] else ""
        return s

    def get(self, case_id: str) -> dict | None:
        with self._lock:
            c = self.cases.get(case_id)
            return json.loads(json.dumps(c)) if c else None

    def add_note(self, case_id: str, note: str, analyst: str = "analyst") -> None:
        with self._lock:
            self.cases[case_id]["notes"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "analyst": analyst, "note": note})
            self._touch(force=True)

    # ------------------------------------------------------------------ feedback
    def feedback(self, case_id: str, label: str, note: str = "", analyst: str = "analyst") -> dict:
        """label: 'confirmed' | 'false_positive'. Returns before/after of everything adjusted."""
        if label not in ("confirmed", "false_positive"):
            raise ValueError("label must be 'confirmed' or 'false_positive'")
        with self._lock:
            c = self.cases[case_id]
            before = self._snapshot(c)
            if c["feedback"] and c["status"] == label:  # idempotent: same label twice does nothing
                return {"case_id": case_id, "status": c["status"], "before": before, "after": before, "changed": False}
            if c["feedback"]:  # label flipped: undo previous effect first
                self._revert(c)
            sign = 1 if label == "confirmed" else -1
            drv, seg = c["driver"], c["segment"]
            sc = self.adjustments["component_scale"]
            factor = 1 + sign * SCALE_STEP
            new_scale = min(SCALE_MAX, max(SCALE_MIN, sc.get(drv, 1.0) * factor))
            applied = {"driver": drv, "segment": seg, "scale_from": sc.get(drv, 1.0), "scale_to": new_scale}
            sc[drv] = new_scale
            so = self.adjustments["segment_offsets"]
            new_off = max(-OFFSET_LIMIT, min(OFFSET_LIMIT, so.get(seg, 0.0) + sign * OFFSET_STEP))
            applied.update(offset_from=so.get(seg, 0.0), offset_to=new_off)
            so[seg] = new_off
            added = []
            if label == "confirmed":
                for a in c["accounts"]:
                    if not a.startswith("EXT_") and a not in self.adjustments["watchlist"]:
                        self.adjustments["watchlist"][a] = case_id
                        added.append(a)
            applied["watchlist_added"] = added
            c["status"] = label
            c["feedback"].append({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "label": label, "analyst": analyst, "note": note, "applied": applied})
            if note:
                c["notes"].append({"ts": c["feedback"][-1]["ts"], "analyst": analyst, "note": note})
            self.adjustments["history"].append({"case_id": case_id, "label": label, **{k: applied[k] for k in ("driver", "segment", "scale_from", "scale_to", "offset_from", "offset_to")}})
            self._touch(force=True)
            return {"case_id": case_id, "status": label, "before": before, "after": self._snapshot(c), "changed": True, "applied": applied}

    def _revert(self, c: dict) -> None:
        a = c["feedback"][-1]["applied"]
        self.adjustments["component_scale"][a["driver"]] = a["scale_from"]
        self.adjustments["segment_offsets"][a["segment"]] = a["offset_from"]
        for acc in a.get("watchlist_added", []):
            self.adjustments["watchlist"].pop(acc, None)

    def _snapshot(self, c: dict) -> dict:
        drv, seg = c["driver"], c["segment"]
        return {"driver": drv, "component_scale": round(self.adjustments["component_scale"].get(drv, 1.0), 4),
                "segment": seg, "segment_offset": round(self.adjustments["segment_offsets"].get(seg, 0.0), 4),
                "watchlist_size": len(self.adjustments["watchlist"]), "status": c["status"]}
