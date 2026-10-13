"""RupeeTrace dashboard.   streamlit run dashboard/app.py

Pages: Live Monitor | Case Queue | Case Detail | Metrics | Demo Walkthrough
Sidebar: engine source (live engine or recorded-backup decision log).
Privacy: pseudonymous IDs everywhere; the analyst toggle on Case Detail reveals real IDs and account details.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from dashboard.components import charts, ui  # noqa: E402
from engine.engine import pseudonym  # noqa: E402
from engine.orchestrator import ACTION_ORDER, INTERVENTION, SLA_MINUTES  # noqa: E402
from engine.recording import REC_DIR, LiveSource, RecordedSource, record  # noqa: E402

st.set_page_config(page_title="RupeeTrace", page_icon="₹", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
PAGES = ["Live Monitor", "Case Queue", "Case Detail", "Metrics", "Demo Walkthrough"]
TICK = 0.25  # seconds per UI refresh while replaying


# --------------------------------------------------------------------------- helpers
def plot(fig, key: str):
    try:
        st.plotly_chart(fig, key=key, width="stretch")
    except TypeError:  # older streamlit
        st.plotly_chart(fig, key=key, use_container_width=True)


def goto(page: str, case_id: str | None = None):
    st.session_state.page = page
    if case_id:
        st.session_state.selected_case = case_id


@st.cache_resource(show_spinner="Starting the risk engine (training the fusion stacker)...")
def get_live() -> LiveSource:
    from engine.engine import RiskEngine
    return LiveSource(RiskEngine())


@st.cache_resource(show_spinner="Loading recorded decision log...")
def get_recorded(stamp: float) -> RecordedSource:
    return RecordedSource()


def get_source():
    mode = st.sidebar.radio("Engine source", ["Live engine", "Recorded backup"], key="mode_choice",
                            help="Recorded backup replays a saved decision log with precomputed lineages - use it if live models fail.")
    if mode == "Recorded backup":
        if not RecordedSource.available():
            st.sidebar.warning("No recording yet.")
            if st.sidebar.button("Record backup log now"):
                with st.spinner("Recording the test split..."):
                    record()
                st.rerun()
            st.stop()
        return get_recorded((REC_DIR / "bundle.json").stat().st_mtime)
    try:
        return get_live()
    except Exception as e:  # real models / ledger failed: fall back automatically
        st.sidebar.error(f"Live engine failed ({type(e).__name__}: {e}).")
        if RecordedSource.available():
            st.sidebar.info("Using recorded backup instead.")
            return get_recorded((REC_DIR / "bundle.json").stat().st_mtime)
        st.stop()


def init_state(src):
    ss = st.session_state
    ss.setdefault("page", PAGES[0])
    ss.setdefault("selected_case", None)
    ss.setdefault("running", False)
    ss.setdefault("step", 0)
    if ss.get("_src_mode") != src.mode:
        ss._src_mode = src.mode
        reset_monitor(src, hard=False)


def reset_monitor(src, hard: bool = True):
    if hard:
        src.reset()
    st.session_state.running = False
    st.session_state.k = {"counts": Counter(), "legit": 0, "legit_flag": 0, "legit_hard": 0, "fraud": 0, "fraud_flag": 0, "lat": [], "last": None, "feed": [], "tick": 0}


def case_options(src):
    return [c["case_id"] for c in src.list_cases()]


def label_fn(analyst: bool):
    return (lambda a: a) if analyst else pseudonym


def worst_action(case: dict) -> str:
    return max(case["actions"], key=ACTION_ORDER.index) if case["actions"] else "HOLD"


# ----------------------------------------------------------------------------- pages
def page_live(src):
    st.subheader("Live Monitor")
    st.caption(f"Replaying the held-out test split through the engine ({src.mode} mode, backend: {src.meta['backend']}). Account IDs are pseudonymous here.")
    ss, k = st.session_state, st.session_state.k
    finished = src.position() >= src.total()
    c1, c2, c3, c4, c5 = st.columns([3, 1, 1, 1, 2])
    c1.slider("Replay speed (transactions / second)", 1, 300, 40, key="speed")
    c2.button("▶ Start", key="start", on_click=lambda: ss.update(running=True), disabled=ss.running or finished, width="stretch")
    c3.button("⏸ Pause", key="pause", on_click=lambda: ss.update(running=False), disabled=not ss.running, width="stretch")
    c4.button("↺ Reset", key="reset", on_click=lambda: reset_monitor(src), width="stretch")
    show_gt = c5.toggle("Show ground-truth column (demo only)", value=False, key="show_gt")
    st.progress(min(1.0, src.position() / max(1, src.total())), text=f"{src.position():,} / {src.total():,} transactions")
    kpi_ph, mid_ph, feed_ph = st.empty(), st.empty(), st.empty()

    def draw():
        counts = k["counts"]
        with kpi_ph.container():
            cols = st.columns(5)
            for col, a in zip(cols, ["ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"]):
                col.markdown(f'<div class="kpi">{ui.badge(a, ui.ACTION_LABEL[a])}<h3 style="margin:4px 0 0 0">{counts[a]:,}</h3></div>', unsafe_allow_html=True)
        with mid_ph.container():
            g, f1, f2 = st.columns([2, 2, 2])
            last = k["last"]
            with g:
                plot(charts.risk_gauge(last["risk_score"] if last else 0.0, last["action"] if last else "ALLOW"), key=f"gauge_{k['tick']}")
            with f1:
                n = max(1, k["legit"])
                st.metric("Friction on legitimate customers", f"{k['legit_flag'] / n:.1%}",
                          help="Share of legitimate transactions that got ANY intervention (OTP, hold, block, escalation). Needs ground-truth labels (demo only).")
                st.caption(f"{k['legit_flag']:,} of {k['legit']:,} legitimate payments touched; hard friction (hold/block/escalate): {k['legit_hard'] / n:.2%}")
            with f2:
                nf = max(1, k["fraud"])
                st.metric("Laundering payments intervened on", f"{k['fraud_flag'] / nf:.1%}", help="Any intervention on a payment labelled laundering (demo only).")
                med = float(pd.Series(k["lat"]).median()) if k["lat"] else 0.0
                st.caption(f"{k['fraud_flag']:,} of {k['fraud']:,} | median decision latency {med:.2f} ms")
        feed_ph.markdown(ui.feed_table(k["feed"][:14], pseudonym, show_gt), unsafe_allow_html=True)

    def ingest(batch):
        for d in batch:
            k["counts"][d["action"]] += 1
            k["lat"] = (k["lat"] + [d["latency_ms"]])[-500:]
            if d.get("label_gt") is not None:
                if d["label_gt"] == 0:
                    k["legit"] += 1
                    k["legit_flag"] += d["action"] != "ALLOW"
                    k["legit_hard"] += d["action"] in ("HOLD", "BLOCK", "ESCALATE")
                else:
                    k["fraud"] += 1
                    k["fraud_flag"] += d["action"] != "ALLOW"
        if batch:
            k["last"] = batch[-1]
            k["feed"] = (list(reversed(batch)) + k["feed"])[:60]
        k["tick"] += 1

    draw()
    if ss.running:
        while ss.running and src.position() < src.total():
            t0 = time.time()
            ingest(src.next_batch(max(1, int(ss.speed * TICK))))
            draw()
            time.sleep(max(0.0, TICK - (time.time() - t0)))
        if src.position() >= src.total():
            ss.running = False
            st.toast("Replay finished - open the Case Queue.")
            st.rerun()


def page_queue(src):
    st.subheader("Case Queue")
    st.caption("One case per money trail (shared root) or connected cluster, not one per transaction. Ranked by risk, open cases first.")
    status = st.radio("Status", ["all", "open", "confirmed", "false_positive"], horizontal=True)
    cases = src.list_cases(None if status == "all" else status)
    if not cases:
        st.info("No cases yet. Start the Live Monitor replay to generate them.")
        return
    df = pd.DataFrame([{"Case": c["case_id"], "Status": c["status"], "Risk": c["max_risk"], "Cluster risk": c["max_cluster_risk"], "Payments": c["txn_count"],
                        "Accounts": c["account_count"], "Root amount": ui.inr(c["root_amount"]), "Total moved": ui.inr(c["total_amount"]),
                        "Fragments": c["fragments"], "Hops": c["hops"], "Main driver": ui.COMPONENT_LABEL.get(c["driver"], c["driver"]),
                        "Top reason": c["top_reason"]} for c in cases])
    ev = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key="queue_tbl",
                      column_config={"Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                     "Cluster risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    rows = ev.selection.rows if ev and ev.selection else []
    cid = df.iloc[rows[0]]["Case"] if rows else df.iloc[0]["Case"]
    st.button(f"Open {cid} →", on_click=goto, args=("Case Detail", cid), type="primary")


def page_detail(src):
    st.subheader("Case Detail")
    ids = case_options(src)
    if not ids:
        st.info("No cases yet. Run the Live Monitor first (or use Recorded backup).")
        return
    if st.session_state.selected_case not in ids:
        st.session_state.selected_case = ids[0]
    cid = st.selectbox("Case", ids, key="selected_case")
    case = src.get_case(cid)
    top = st.columns([1, 3])
    analyst = top[0].toggle("Analyst view", key="analyst", help="Reveal real account IDs and account details. Off = pseudonymous IDs.")
    top[1].caption("Privacy: pseudonymous IDs by default; account details are only shown here, in the analyst view.")
    lab = label_fn(analyst)
    n = max(1, case["txn_count"])
    status_color = {"open": "#e0a100", "confirmed": "#d63031", "false_positive": "#2e9e5b"}[case["status"]]
    st.markdown(f"### {case['case_id']} &nbsp; <span class='badge' style='background:{status_color}'>{case['status']}</span> &nbsp; "
                f"{ui.badge(worst_action(case))}", unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric("Peak risk", f"{case['max_risk']:.2f}")
    m[1].metric("Cluster risk", f"{case['max_cluster_risk']:.2f}")
    m[2].metric("Payments in case", f"{case['txn_count']:,}")
    m[3].metric("Root amount", ui.inr(case.get("root_amount")))
    m[4].metric("Accounts involved", f"{len(case['accounts'])}")
    frag, hops = case.get("fragments") or 0, case.get("hops") or 0
    st.markdown(f"<div class='story'>{case['txn_count']} payments moving {ui.inr(case['total_amount'])} trace back to one root of "
                f"<b>{ui.inr(case.get('root_amount'))}</b> ({case.get('origin_type') or 'unknown'} origin), split into <b>{frag}</b> fragments over up to "
                f"<b>{hops}</b> hops.</div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why this was flagged**")
        for r, cnt in sorted(case["reasons"].items(), key=lambda x: -x[1])[:8]:
            st.markdown(f"- {r} <span style='opacity:.6'>(×{cnt})</span>", unsafe_allow_html=True)
    with right:
        w = worst_action(case)
        st.markdown("**Recommended next step**")
        st.info(f"{INTERVENTION[w]}  \nSLA: {SLA_MINUTES[w]} min")
    avg = {c: v / n for c, v in case["driver_totals"].items()}
    plot(charts.component_breakdown((case.get("peak") or {}).get("components", {}), avg), key=f"bd_{cid}")

    # ----- lineage view
    st.markdown("#### Lineage view")
    tx_ids = src.lineage_txns(cid)
    step = max(1, len(tx_ids) // 80)
    opts = tx_ids[::step]
    peak = (case.get("peak") or {}).get("txn_id")
    if peak and peak not in opts:
        opts.append(peak)
    amt = {t["txn_id"]: t for t in case["txns"]}
    sel = st.selectbox("Transaction to trace", opts, index=opts.index(peak) if peak in opts else 0,
                       format_func=lambda t: f"{t} · {ui.inr(amt[t]['amount'])} · {amt[t]['action']}" if t in amt else t, key=f"lin_{cid}")
    lin = src.lineage(sel)
    if lin and lin.get("nodes"):
        k = st.columns(5)
        k[0].metric("Root", ui.inr(lin["root_aggregate_amount"]))
        k[1].metric("Origin", lin["origin_type"])
        k[2].metric("Fragments so far", lin["fragment_count"])
        k[3].metric("Hops", lin["hops"])
        k[4].metric("Hash chain", "✅ intact" if lin["hash_chain_ok"] else "⚠️ broken")
        plot(charts.lineage_sankey(case, lin, lab), key=f"sk_{cid}_{sel}")
        trail = []
        for i, nd in enumerate(lin["nodes"]):
            trail.append({"Hop": i + 1, "From": lab(lin["account_path"][i]), "To": lab(lin["account_path"][i + 1]), "Amount": ui.inr(nd["amount"]),
                          "Time": str(nd["ts"])[:19].replace("T", " "), "Dwell before hop (min)": lin["dwell_minutes"][i - 1] if i > 0 else None})
        st.markdown("**Account trail** (root origin → this payment)")
        st.dataframe(pd.DataFrame(trail), hide_index=True, width="stretch")
        if analyst:
            det = [src.account_details(a) for a in dict.fromkeys(lin["account_path"])]
            st.markdown("**Account details** (analyst view)")
            st.dataframe(pd.DataFrame(det), hide_index=True, width="stretch")
    else:
        st.info("No lineage available for this transaction.")

    # ----- network
    st.markdown("#### Related accounts")
    graph = src.case_graph(cid, 60)
    plot(charts.network_graph(graph, lab, analyst, highlight=set(lin.get("account_path", [])) if lin else None), key=f"net_{cid}_{analyst}_{sel}")
    if graph.get("truncated"):
        st.caption(f"Graph bounded to the 60 most active accounts out of {graph.get('total_accounts')}.")

    # ----- analyst actions
    st.markdown("#### Analyst decision")
    note = st.text_input("Note (optional)", key=f"note_{cid}")
    b1, b2, _ = st.columns([1, 1, 3])
    disabled = not src.can_feedback
    if b1.button("✅ Confirm laundering", disabled=disabled, type="primary", key=f"conf_{cid}"):
        st.session_state[f"fb_{cid}"] = src.feedback(cid, "confirmed", note)
        st.rerun()
    if b2.button("❌ Mark false positive", disabled=disabled, key=f"fp_{cid}"):
        st.session_state[f"fb_{cid}"] = src.feedback(cid, "false_positive", note)
        st.rerun()
    if disabled:
        st.caption("Feedback is disabled in recorded-backup mode (read-only replay).")
    fb = st.session_state.get(f"fb_{cid}")
    if fb and fb.get("changed"):
        b, a, rs = fb["before"], fb["after"], fb.get("rescore_peak_txn")
        st.success(f"Feedback applied: case is now **{a['status']}**. The engine adjusted itself:")
        rows = [("Weight scale of driving signal (" + ui.COMPONENT_LABEL.get(b["driver"], b["driver"]) + ")", b["component_scale"], a["component_scale"]),
                (f"Segment risk offset ({b['segment']})", b["segment_offset"], a["segment_offset"]), ("Watchlist accounts", b["watchlist_size"], a["watchlist_size"])]
        if rs:
            rows += [("Peak payment risk score", rs["before"]["risk_score"], rs["after"]["risk_score"]),
                     ("Peak payment action", rs["before"]["action"], rs["after"]["action"])]
        st.dataframe(pd.DataFrame(rows, columns=["Adjustment", "Before", "After"]).astype(str), hide_index=True, width="stretch")
    if case["notes"]:
        with st.expander(f"Analyst notes ({len(case['notes'])})"):
            for nt in case["notes"]:
                st.write(f"{nt['ts']} · {nt['analyst']}: {nt['note']}")
    with st.expander("Engine adjustments learned from feedback"):
        adj = src.adjustments()
        st.json({"fusion_mode": adj.get("mode"), "weights": adj.get("weights"), "component_scale": adj.get("component_scale"),
                 "segment_offsets": adj.get("segment_offsets"), "watchlist_size": adj.get("watchlist_size")})


def page_metrics(src):
    st.subheader("Metrics")
    if src.mode == "live" and src.position() == 0:
        st.info("Run the Live Monitor replay (or use Recorded backup) so the engine has decisions to evaluate.")
    m = src.metrics()
    if not m.get("variants"):
        st.warning("; ".join(m.get("notes") or ["No metrics available yet."]))
        return
    st.caption(f"Source: {m['source']}")
    rows = []
    for v in m["variants"].values():
        rows.append({"Variant": v["label"], "Recall (any action)": v["recall"], "Recall (hold+)": v["recall_hold_plus"], "Precision": v["precision"],
                     "Friction on legit": v["friction"], "Hard friction": v["friction_hard"],
                     "Median latency (ms)": (v.get("latency") or {}).get("median_ms")})
    df = pd.DataFrame(rows)
    fmt = {c: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")
           for c in ["Recall (any action)", "Recall (hold+)", "Precision", "Friction on legit", "Hard friction"]}
    st.markdown("**Ablation table**")
    st.dataframe(df, hide_index=True, width="stretch", column_config=fmt)
    plot(charts.recall_by_bucket(m["variants"]), key="recall_bucket")
    l1, l2 = st.columns(2)
    lat = (m.get("latency") or {}).get("engine_end_to_end", {})
    with l1:
        st.markdown("**Decision latency (end to end)**")
        if lat:
            a, b, c = st.columns(3)
            a.metric("Median", f"{lat['median_ms']:.2f} ms")
            b.metric("p95", f"{lat['p95_ms']:.2f} ms")
            c.metric("Max", f"{lat['max_ms']:.2f} ms")
            st.caption(f"Backend: {m['latency'].get('backend')}. Target: well under 100 ms.")
    with l2:
        st.markdown("**Friction on legitimate customers**")
        lf = m.get("live_friction")
        if lf:
            st.metric("Legitimate payments touched", f"{(lf['stepped_up'] + lf['held_or_worse']) / max(1, lf['legit_txns']):.1%}")
            st.caption(f"{lf['stepped_up']} OTP step-ups, {lf['held_or_worse']} hold/block/escalate, of {lf['legit_txns']:,} legitimate payments.")
    if m.get("ablation_csv"):
        with st.expander("Ablation table from eval/out (acc_1)"):
            st.dataframe(pd.DataFrame(m["ablation_csv"]), hide_index=True, width="stretch")
    for p in m.get("charts", [])[:6]:
        if p.lower().endswith((".png", ".jpg")):
            st.image(p, caption=Path(p).name)


def page_walkthrough(src):
    st.subheader("Demo Walkthrough")
    if src.mode == "live" and src.position() < src.total():
        with st.spinner("Scoring the remaining test transactions to prepare the story..."):
            story = src.story()
    else:
        story = src.story()
    if not story:
        st.warning("Could not find the demo transaction in this dataset.")
        return
    t, lin, outc = story["txn"], story["lineage"], {o["variant"][:1]: o for o in story["outcomes"]}
    ss = st.session_state
    titles = ["The payment", "Rules say", "ML-only says", "Combined engine says", "Why: money lineage"]
    st.progress((ss.step + 1) / len(titles), text=f"Step {ss.step + 1} of {len(titles)}: {titles[ss.step]}")
    nav = st.columns([1, 1, 6])
    nav[0].button("◀ Back", disabled=ss.step == 0, on_click=lambda: ss.update(step=ss.step - 1))
    nav[1].button("Next ▶", disabled=ss.step == len(titles) - 1, on_click=lambda: ss.update(step=ss.step + 1), type="primary")
    case = src.get_case(story["case_id"]) if story.get("case_id") else None
    frags = max(lin.get("fragment_count", 0), (case or {}).get("fragments") or 0)
    root = lin.get("root_aggregate_amount")
    if ss.step == 0:
        st.markdown(f"<div class='story'><h4 style='margin:0'>{ui.inr(t['amount'])} {t.get('channel') or 'UPI'} transfer</h4>"
                    f"{pseudonym(t['src_account'])} → {pseudonym(t['dst_account'])} · {str(t['ts'])[:16].replace('T', ' ')}<br>"
                    f"It is far below the {ui.inr(config.ALERT_THRESHOLD_INR)} alert threshold. Nothing about the sender looks unusual on its own.</div>", unsafe_allow_html=True)
        st.write("Four systems will look at this one payment. Press **Next**.")
    elif ss.step == 1:
        st.markdown(f"### Threshold rules: {ui.badge(outc['A']['action'])}", unsafe_allow_html=True)
        st.write(f"{ui.inr(t['amount'])} < {ui.inr(config.ALERT_THRESHOLD_INR)}, so the per-transaction rule never fires. Splitting money into small pieces is exactly how this is evaded.")
    elif ss.step == 2:
        st.markdown(f"### ML-only (behaviour + anomaly): {ui.badge(outc['B']['action'])} &nbsp; score {outc['B']['risk_score']:.2f}", unsafe_allow_html=True)
        st.markdown(f"Graph added: {ui.badge(outc['C']['action'])} (score {outc['C']['risk_score']:.2f}). The sending account has a normal history and no obvious device sharing, "
                    "so per-account and graph-only views see nothing alarming.", unsafe_allow_html=True)
    elif ss.step == 3:
        d = t
        st.markdown(f"### Combined engine: {ui.badge(d['action'])} &nbsp; risk {d['risk_score']:.2f} &nbsp; (SLA {d['sla_minutes']} min)", unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            plot(charts.decision_bars(d["components"]), key="story_bars")
        with c2:
            st.markdown("**Reasons**")
            for r in d["reasons"]:
                st.markdown(f"- {r}")
            st.info(d["recommended_intervention"])
        st.caption("Provenance is one of four inputs: on its own it can never reach HOLD. It is the combination that lifts the score.")
    else:
        st.markdown(f"<div class='story'>This {ui.inr(t['amount'])} payment is <b>one of ~{frags} fragments</b> of a <b>{ui.inr(root)}</b> root "
                    f"that moved through <b>{lin['hops']}</b> hops, with money relayed onward quickly. Hash chain: "
                    f"{'✅ intact' if lin['hash_chain_ok'] else '⚠️ broken'}.</div>", unsafe_allow_html=True)
        if case:
            plot(charts.lineage_sankey(case, lin, pseudonym), key="story_sankey")
            st.success(f"All {case['txn_count']} related payments are grouped into ONE case: {case['case_id']}.")
            st.button("Open this case in Case Detail →", on_click=goto, args=("Case Detail", case["case_id"]), type="primary")
    st.markdown("**Outcome per approach**")
    o = story["outcomes"]
    st.markdown(" &nbsp; ".join(f"{x['variant']}: {ui.badge(x['action'])}" for x in o), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- main
def main():
    st.sidebar.markdown("## ₹ RupeeTrace")
    st.sidebar.caption("Real-time fraud & AML risk orchestration")
    src = get_source()
    init_state(src)
    st.sidebar.radio("Page", PAGES, key="page")
    st.sidebar.caption(f"Backend: **{src.meta['backend']}** · data: {src.meta['data_source']}")
    if "mock" in src.meta["backend"]:
        st.sidebar.warning("Using MOCK scorer / ledger (USE_MOCK=True)")
    {"Live Monitor": page_live, "Case Queue": page_queue, "Case Detail": page_detail, "Metrics": page_metrics,
     "Demo Walkthrough": page_walkthrough}[st.session_state.page](src)


main()
