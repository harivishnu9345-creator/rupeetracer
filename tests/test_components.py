"""Tests for dashboard.components (charts + ui). Plain asserts, no fixtures: runs under pytest or as a script."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.components import charts, ui  # noqa: E402

ACCS = ["ACC_00101", "ACC_00202", "ACC_00303", "ACC_00404", "ACC_00505"]


def _pseudo(a):
    return ui.local_pseudonym(a)


def _valid(fig):
    d = json.loads(fig.to_json())
    assert "data" in d and "layout" in d
    return d


def _lineage(n_nodes=100, path=ACCS):
    return {"txn_id": "T9", "root_txn_id": "T0", "origin_type": "CASH", "root_aggregate_amount": 500000.0, "fragment_count": 100,
            "hops": len(path) - 1, "account_path": list(path), "dwell_minutes": [12.0, 3.5, 190.0], "hash_chain_ok": True,
            "nodes": [{"lot_id": f"L{i}", "account": path[i % len(path)], "amount": 4800.0, "ts": "2026-01-25T02:30:00Z", "parent_lot_ids": []}
                      for i in range(n_nodes)]}


def _graph(n=150, shared=True):
    nodes = [{"id": f"ACC_{i:05d}", "pseudonym": _pseudo(f"ACC_{i:05d}"), "degree": n - i, "device": f"DEV_{i // 2}", "on_watchlist": i == 0} for i in range(n)]
    edges = [{"src": f"ACC_{i:05d}", "dst": f"ACC_{i + 1:05d}", "amount": 4800.0, "count": 2, "kind": "payment"} for i in range(n - 1)]
    if shared:
        edges.append({"src": "ACC_00000", "dst": "ACC_00001", "amount": 0.0, "count": 0, "kind": "shared_device", "device": "DEV_0"})
    return {"nodes": nodes, "edges": edges, "truncated": True, "total_accounts": n}


# ------------------------------------------------------------------------------- ui
def test_badge_colors_all_actions():
    for a, c in {"ALLOW": "#2e9e5b", "STEP_UP": "#e0a100", "HOLD": "#e8590c", "BLOCK": "#d63031", "ESCALATE": "#7b3fe4"}.items():
        h = ui.badge(a)
        assert c in h and "badge" in h and ui.ACTION_LABEL[a] in h
    assert "Custom" in ui.badge("HOLD", "Custom")


def test_badge_unknown_and_escaping():
    assert ui.UNKNOWN_COLOR in ui.badge("WAT")
    assert ui.UNKNOWN_COLOR in ui.badge(None)
    assert "<script>" not in ui.badge("HOLD", "<script>x</script>")


def test_inr_formatting():
    assert ui.inr(500000) == "₹5,00,000"
    assert ui.inr(4800) == "₹4,800"
    assert ui.inr(12345678) == "₹1,23,45,678"
    assert ui.inr(4800.5) == "₹4,800.50"
    assert ui.inr(-1500) == "-₹1,500"
    assert ui.inr(None) == "–" and ui.inr("x") == "–" and ui.inr(float("nan")) == "–"


def test_css_has_classes_app_uses():
    assert all(k in ui.CSS for k in (".badge", ".kpi", ".story", "<style>"))
    assert set(ui.ACTION_LABEL) == {"ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"}
    assert ui.COMPONENT_LABEL["provenance"]


def test_feed_table_rows_pseudonymised_and_gt_toggle():
    feed = [{"ts": "2026-01-22T02:30:00+00:00", "src_account": "ACC_00101", "dst_account": "ACC_00202", "amount": 4800, "channel": "UPI",
             "action": "HOLD", "risk_score": 0.61, "reasons": ["Fast relay <b>x</b>"], "label_gt": 1}]
    h = ui.feed_table(feed, _pseudo, False)
    assert "ACC_00101" not in h and _pseudo("ACC_00101") in h and "₹4,800" in h and "Hold" in h
    assert "<b>x</b>" not in h and "laundering" not in h
    assert "laundering" in ui.feed_table(feed, _pseudo, True)


def test_feed_table_empty_partial_and_bad_fn():
    assert "No transactions" in ui.feed_table([])
    assert "No transactions" in ui.feed_table(None)
    h = ui.feed_table([{}, {"action": "BLOCK"}, {"src_account": "ACC_1", "risk_score": "bad", "reasons": None}], lambda a: 1 / 0, True)
    assert h.count("<tr>") == 4  # header + 3 rows
    assert "ACC_1" not in h  # raising pseudonym fn falls back to local pseudonym, never raw


# ----------------------------------------------------------------------------- gauge
def test_gauge_value_color_and_clamp():
    d = _valid(charts.risk_gauge(0.9, "ESCALATE"))
    assert d["data"][0]["value"] == 0.9 and d["data"][0]["gauge"]["bar"]["color"] == "#7b3fe4"
    assert _valid(charts.risk_gauge(5, "BLOCK"))["data"][0]["value"] == 1.0
    assert _valid(charts.risk_gauge(-2))["data"][0]["value"] == 0.0


def test_gauge_bad_inputs_do_not_raise():
    for s, a in [("x", "HOLD"), (None, None), (float("nan"), "WAT")]:
        assert _valid(charts.risk_gauge(s, a))["data"][0]["value"] == 0.0


# ------------------------------------------------------------------- component bars
def test_component_breakdown_and_decision_bars():
    peak = {"behavior": 0.2, "anomaly": 0.1, "graph": 0.5, "provenance": 0.9}
    d = _valid(charts.component_breakdown(peak, {"behavior": 0.05, "provenance": 0.3}))
    assert len(d["data"]) == 2 and d["data"][0]["x"] == [0.2, 0.1, 0.5, 0.9]
    comps = {c: {"score": 0.4, "contribution": 0.1} for c in charts.COMPONENTS}
    d = _valid(charts.decision_bars(comps))
    assert d["data"][0]["y"] == [0.4] * 4 and d["data"][1]["y"] == [0.1] * 4
    assert _valid(charts.decision_bars({"behavior": 0.7}))["data"][0]["y"][0] == 0.7  # plain floats accepted


def test_component_charts_empty_and_partial():
    for f in (charts.component_breakdown, charts.decision_bars):
        assert _valid(f())["layout"]["annotations"]
        assert _valid(f({}))["layout"]["annotations"]
    assert _valid(charts.component_breakdown({"graph": None}, {"x": "bad"}))["data"]


# ---------------------------------------------------------------------------- sankey
def test_sankey_100_fragment_lineage():
    lin = _lineage(100)
    d = _valid(charts.lineage_sankey({"case_id": "CASE-1"}, lin, _pseudo))
    sk = d["data"][0]
    assert sk["node"]["label"][1] == "100 fragments"
    assert len(sk["node"]["label"]) == 2 + len(ACCS) and len(sk["link"]["source"]) == 1 + len(ACCS)
    assert max(sk["link"]["target"]) < len(sk["node"]["label"])
    t = d["layout"]["title"]["text"]
    assert "4 hops" in t and "100 fragments" in t and "intact" in t
    assert any("held 190 min" in x or "held 3.2 h" in x for x in sk["node"]["label"])
    assert "₹5,00,000" in sk["node"]["label"][0] and "CASH" in sk["node"]["label"][0]


def test_sankey_hash_chain_broken_and_unknown():
    lin = _lineage(3)
    lin["hash_chain_ok"] = False
    assert "BROKEN" in _valid(charts.lineage_sankey({}, lin, _pseudo))["layout"]["title"]["text"]
    del lin["hash_chain_ok"]
    assert "unknown" in _valid(charts.lineage_sankey({}, lin, _pseudo))["layout"]["title"]["text"]


def test_sankey_empty_and_partial_lineage():
    for lin in (None, {}, {"nodes": []}, {"account_path": []}):
        d = _valid(charts.lineage_sankey(None, lin))
        assert d["layout"]["annotations"] and not any(t["type"] == "sankey" for t in d["data"])
    d = _valid(charts.lineage_sankey(None, {"account_path": ["A", "B", "C"]}, _pseudo))  # missing everything else
    assert d["data"][0]["node"]["label"][1] == "Fragments"
    d = _valid(charts.lineage_sankey({"fragments": 7}, {"nodes": [{"account": "A"}, {"account": "B"}]}, _pseudo))  # path derived from nodes
    assert d["data"][0]["node"]["label"][1] == "7 fragments"


def test_sankey_repeated_account_and_long_path_bounded():
    d = _valid(charts.lineage_sankey({}, _lineage(5, path=["A", "B", "A", "B", "A"]), _pseudo))
    assert len(d["data"][0]["node"]["label"]) == 7  # positional nodes: no cycles
    long = _lineage(5, path=[f"ACC_{i}" for i in range(40)])
    d = _valid(charts.lineage_sankey({}, long, _pseudo))
    assert len(d["data"][0]["node"]["label"]) == 2 + charts.MAX_TRAIL and "shown" in d["layout"]["title"]["text"]


def test_sankey_analyst_toggle_hides_and_reveals():
    lin = _lineage(10)
    hidden = charts.lineage_sankey({}, lin, lambda a: a, analyst=False).to_json()
    assert not any(a in hidden for a in ACCS) and _pseudo(ACCS[0]) in hidden
    default_private = charts.lineage_sankey({}, lin).to_json()  # no lab -> pseudonymous
    assert not any(a in default_private for a in ACCS)
    shown = charts.lineage_sankey({}, lin, lambda a: a, analyst=True).to_json()
    assert all(a in shown for a in ACCS)


# ---------------------------------------------------------------------------- network
def _acct_trace(d):
    return next(t for t in d["data"] if t.get("name") == "Account")


def test_network_node_cap_enforced_with_note():
    d = _valid(charts.network_graph(_graph(150), _pseudo, False))
    assert len(_acct_trace(d)["x"]) == charts.MAX_NODES == 60
    assert any("Showing top 60 of 150" in a["text"] for a in d["layout"]["annotations"])


def test_network_cap_applies_even_if_caller_flags_nothing():
    g = _graph(90)
    g["truncated"] = False
    g.pop("total_accounts")
    d = _valid(charts.network_graph(g, _pseudo, False))
    assert len(_acct_trace(d)["x"]) == 60
    assert any("top 60 of 90" in a["text"] for a in d["layout"]["annotations"])


def test_network_small_graph_no_note_and_shared_device_highlighted():
    g = _graph(10)
    g["truncated"] = False
    g["total_accounts"] = 10
    d = _valid(charts.network_graph(g, _pseudo, False, highlight={"ACC_00003"}))
    assert not d["layout"]["annotations"]
    sd = next(t for t in d["data"] if t.get("name") == "Shared device link")
    assert sd["line"]["color"] == "#d63031" and sd["line"]["width"] > 2
    pay = next(t for t in d["data"] if t.get("name") == "Payment")
    assert pay["line"]["width"] < sd["line"]["width"]
    assert "#e8590c" in _acct_trace(d)["marker"]["line"]["color"]  # highlighted path account ringed


def test_network_analyst_toggle_hides_and_reveals():
    g = _graph(12)
    off = charts.network_graph(g, lambda a: a, False).to_json()
    assert "ACC_0" not in off and "DEV_" not in off and "ACC-" in off
    on = charts.network_graph(g, lambda a: a, True).to_json()
    assert "ACC_00001" in on and "DEV_0" in on


def test_network_off_ignores_leaky_lab_and_missing_pseudonym():
    g = _graph(5)
    for n in g["nodes"]:
        n.pop("pseudonym")
    off = charts.network_graph(g, lambda a: "REAL-" + a, False).to_json()
    assert "REAL-" not in off and "ACC_0" not in off and "DEV_" not in off


def test_network_empty_and_partial():
    for g in (None, {}, {"nodes": [], "edges": []}):
        d = _valid(charts.network_graph(g))
        assert d["layout"]["annotations"]
    g = {"nodes": [{"id": "A"}, {"id": "B"}, {"id": "A"}], "edges": [{"src": "A", "dst": "ZZZ"}, {"src": "A", "dst": "A"}, {"src": "A", "dst": "B"}]}
    d = _valid(charts.network_graph(g, None, False))
    assert len(_acct_trace(d)["x"]) == 2  # dup node dropped, dangling/self edges ignored


# ------------------------------------------------------------------------------ metrics
def test_recall_by_bucket():
    v = {"A_rules": {"label": "A. Rules", "recall_by_bucket": [{"bucket": "< Rs 5,000", "n": 40, "recall": 0.0}, {"bucket": ">= Rs 50,000", "n": 0, "recall": None}]},
         "D": {"label": "D. Full", "recall_by_bucket": [{"bucket": "< Rs 5,000", "n": 40, "recall": 0.8}, {"bucket": ">= Rs 50,000", "n": 0, "recall": None}]}}
    d = _valid(charts.recall_by_bucket(v))
    assert len(d["data"]) == 2 and d["data"][1]["y"] == [0.8, None] and d["data"][1]["name"] == "D. Full"


def test_recall_by_bucket_empty_and_partial():
    for v in (None, {}, {"x": {}}, {"x": {"recall_by_bucket": []}}):
        assert _valid(charts.recall_by_bucket(v))["layout"]["annotations"]
    assert _valid(charts.recall_by_bucket({"x": {"recall_by_bucket": [{"recall": "bad"}]}}))["data"]


# ------------------------------------------------------------------------------ purity
def test_inputs_not_mutated_and_output_deterministic():
    lin, g, v = _lineage(20), _graph(80), {"x": {"recall_by_bucket": [{"bucket": "b", "n": 1, "recall": 0.5}]}}
    snap = copy.deepcopy((lin, g, v))
    a = (charts.lineage_sankey({}, lin, _pseudo).to_json(), charts.network_graph(g, _pseudo, False).to_json(), charts.recall_by_bucket(v).to_json())
    b = (charts.lineage_sankey({}, lin, _pseudo).to_json(), charts.network_graph(g, _pseudo, False).to_json(), charts.recall_by_bucket(v).to_json())
    assert (lin, g, v) == snap and a == b
