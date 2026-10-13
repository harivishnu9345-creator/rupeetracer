"""Recorded-backup mode + a common read interface for the dashboard.

python -m engine.recording            -> writes data/state/recorded/{decisions.jsonl,bundle.json}

LiveSource      : wraps a RiskEngine (full functionality, including analyst feedback).
RecordedSource  : replays a saved decision log with PRECOMPUTED lineages, graphs, metrics and story,
                  so the demo still works if the real models / ledger fail. Feedback is disabled there.
Both expose the same methods so dashboard/app.py never branches on the mode.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.engine import ROOT, RiskEngine, pseudonym
from engine.metrics import compute_metrics, pick_story_txn, story_outcomes

REC_DIR = ROOT / "data" / "state" / "recorded"
LINEAGE_PER_CASE = 40


def _slim(d: dict) -> dict:
    keep = ("txn_id", "ts", "src_account", "dst_account", "amount", "channel", "action", "risk_score", "cluster_risk", "reasons", "components",
            "recommended_intervention", "sla_minutes", "case_id", "latency_ms", "label_gt", "segment", "escalated", "features")
    return {k: d.get(k) for k in keep}


def record(out_dir: Path = REC_DIR, n: int | None = None, use_mock: bool | None = None) -> Path:
    """Run the full test split on a private engine instance (live state untouched) and save everything the UI needs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    e = RiskEngine(use_mock=use_mock, state_dir=out_dir / "_engine", persist=False)
    decs = e.next_batch(n or len(e.test_df))
    with open(out_dir / "decisions.jsonl", "w") as f:
        for d in decs:
            f.write(json.dumps(_slim(d), default=str) + "\n")
    cases, lineages, graphs, details = {}, {}, {}, {}
    for s in e.cases.list():
        c = e.cases.get(s["case_id"])
        cases[c["case_id"]] = c
        graphs[c["case_id"]] = e.case_graph(c["case_id"])
        ids = [t["txn_id"] for t in c["txns"]]
        step = max(1, len(ids) // LINEAGE_PER_CASE)
        for tid in set(ids[::step] + ids[-1:] + [c["peak"]["txn_id"]]):
            lineages[tid] = e.lineage(tid)
        for a in c["accounts"]:
            details[a] = e.account_details(a)
    story = None
    st = pick_story_txn(e)
    if st:
        lineages[st["txn_id"]] = e.lineage(st["txn_id"])
        story = {"txn": _slim(st), "outcomes": story_outcomes(e, st["txn_id"]), "lineage": lineages[st["txn_id"]], "case_id": st["case_id"]}
    bundle = {"meta": {"backend": e.backend, "data_source": e.data_source, "n_decisions": len(decs), "fusion": e.fusion_info()},
              "cases": cases, "lineages": lineages, "graphs": graphs, "account_details": details, "metrics": compute_metrics(e), "story": story}
    (out_dir / "bundle.json").write_text(json.dumps(bundle, default=str))
    return out_dir


class LiveSource:
    mode = "live"
    can_feedback = True

    def __init__(self, engine: RiskEngine):
        self.e = engine

    @property
    def meta(self):
        return {"backend": self.e.backend, "data_source": self.e.data_source, "fusion": self.e.fusion_info()}

    def next_batch(self, n):
        return self.e.next_batch(n)

    def reset(self):
        self.e.reset_stream()

    def total(self):
        return len(self.e.test_df)

    def position(self):
        return self.e.cursor

    def list_cases(self, status=None):
        return self.e.cases.list(status)

    def get_case(self, cid):
        return self.e.cases.get(cid)

    def lineage(self, tid):
        return self.e.lineage(tid)

    def lineage_txns(self, cid):
        return [t["txn_id"] for t in (self.e.cases.get(cid) or {}).get("txns", [])]

    def case_graph(self, cid, max_nodes=60):
        return self.e.case_graph(cid, max_nodes)

    def account_details(self, a):
        return self.e.account_details(a)

    def feedback(self, cid, label, note=""):
        return self.e.feedback(cid, label, note)

    def add_note(self, cid, note):
        self.e.cases.add_note(cid, note)

    def metrics(self):
        return compute_metrics(self.e)

    def adjustments(self):
        return self.e.fusion_info()

    def story(self):
        if self.e.cursor == 0:  # the story needs the test split to have been scored once
            self.e.next_batch(len(self.e.test_df))
        st = pick_story_txn(self.e)
        if not st:
            return None
        return {"txn": _slim(st), "outcomes": story_outcomes(self.e, st["txn_id"]), "lineage": self.e.lineage(st["txn_id"]), "case_id": st["case_id"]}


class RecordedSource:
    mode = "recorded"
    can_feedback = False

    def __init__(self, rec_dir: Path = REC_DIR):
        self.dir = Path(rec_dir)
        self.b = json.loads((self.dir / "bundle.json").read_text())
        self.decs = [json.loads(line) for line in (self.dir / "decisions.jsonl").read_text().splitlines()]
        self.cursor = 0

    @staticmethod
    def available(rec_dir: Path = REC_DIR) -> bool:
        return (Path(rec_dir) / "bundle.json").exists() and (Path(rec_dir) / "decisions.jsonl").exists()

    meta = property(lambda s: s.b["meta"])

    def next_batch(self, n):
        out = self.decs[self.cursor:self.cursor + n]
        self.cursor += n
        return out

    def reset(self):
        self.cursor = 0

    def total(self):
        return len(self.decs)

    def position(self):
        return self.cursor

    def list_cases(self, status=None):
        from engine.cases import CaseManager
        out = [CaseManager.summary(c) for c in self.b["cases"].values() if status in (None, c["status"])]
        return sorted(out, key=lambda c: (c["status"] != "open", -c["max_risk"], -c["txn_count"]))

    def get_case(self, cid):
        return self.b["cases"].get(cid)

    def lineage(self, tid):
        return self.b["lineages"].get(tid, {})

    def lineage_txns(self, cid):
        return [t["txn_id"] for t in self.b["cases"][cid]["txns"] if t["txn_id"] in self.b["lineages"]]

    def case_graph(self, cid, max_nodes=60):
        return self.b["graphs"].get(cid, {"nodes": [], "edges": []})

    def account_details(self, a):
        return self.b["account_details"].get(a, {"account_id": a, "type": "external pseudo-account"})

    def feedback(self, *a, **k):
        raise RuntimeError("Analyst feedback is disabled in recorded-backup mode")

    def add_note(self, *a, **k):
        raise RuntimeError("Notes are disabled in recorded-backup mode")

    def metrics(self):
        return self.b["metrics"]

    def adjustments(self):
        return self.b["meta"]["fusion"]

    def story(self):
        return self.b.get("story")


if __name__ == "__main__":
    p = record()
    print("recorded ->", p, "| decisions:", sum(1 for _ in open(p / "decisions.jsonl")))
