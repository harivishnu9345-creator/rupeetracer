"""
RupeeTrace main.py
------------------
python main.py          -> builds eval/out/dashboard.html from existing results and opens it
python main.py --run    -> first runs train + predictions + metrics, then builds the dashboard

Needs: pandas numpy plotly networkx. Reads data/out/*.csv and eval/out/*.csv.
All charts show pseudonymous account IDs. The Transactions tab has an analyst toggle.
"""
import sys, subprocess, hashlib, json, webbrowser, html as H
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval" / "out"
DATA = ROOT / "data" / "out"
PRED = {"A": "predictions_A_rules.csv", "B": "predictions_B_behavior_anomaly.csv",
        "C": "predictions_C_plus_graph.csv", "D": "predictions_D_plus_provenance.csv"}
PRED_COLS = ["txn_id", "risk_score", "action", "latency_ms", "reasons"]
COL = {"ALLOW": "#38D9A9", "STEP_UP": "#FFBE55", "HOLD": "#FF8F52", "BLOCK": "#FF667D", "ESCALATE": "#B69CFF"}
CYAN, TEXT, MUTED = "#35D6FF", "#EAF4FF", "#91A9C2"


# ---------------------------------------------------------------- pipeline
def run_pipeline():
    for mod in ["models.train", "eval.rules_baseline", "eval.make_predictions",
                "eval.metrics", "eval.ablation", "eval.buckets"]:
        print(f"\n=== running {mod} ===", flush=True)
        r = subprocess.run([sys.executable, "-m", mod], cwd=ROOT)
        if r.returncode != 0:
            print(f"[warn] {mod} exited with code {r.returncode}")


def fix_predictions():
    """Make B/C/D prediction files schema-valid (column order + JSON reasons)."""
    for k in "BCD":
        p = OUT / PRED[k]
        if not p.exists():
            continue
        d = pd.read_csv(p, keep_default_na=False)
        if "reasons" not in d.columns:
            d["reasons"] = "[]"
        d["reasons"] = d["reasons"].apply(lambda x: x if str(x).startswith("[") else "[]")
        d[PRED_COLS].to_csv(p, index=False)


# ---------------------------------------------------------------- helpers
def pid(x):
    x = str(x)
    return ("EXT-" if x.startswith("EXT_") else "AC-") + hashlib.md5(x.encode()).hexdigest()[:5].upper()


def nonempty(s):
    return s.astype(str).str.strip().replace({"nan": "", "None": ""}) != ""


def style(fig, h=420):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT), height=h, margin=dict(l=20, r=20, t=50, b=30),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


_first = [True]


def to_div(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})


def num(df, row_prefix, col, default=float("nan")):
    try:
        lab = df.columns[0] if "label" not in df.columns else "label"
        r = df[df[lab].astype(str).str.startswith(row_prefix)].iloc[0]
        return float(r[col])
    except Exception:
        return default


def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


# ---------------------------------------------------------------- load
def load():
    d = {}
    d["tx"] = pd.read_csv(DATA / "transactions.csv", dtype=str, keep_default_na=False)
    d["tx"]["amount"] = pd.to_numeric(d["tx"]["amount"], errors="coerce").fillna(0.0)
    d["tx"]["label"] = pd.to_numeric(d["tx"].get("label", 0), errors="coerce").fillna(0).astype(int)
    for k, f in PRED.items():
        p = OUT / f
        d[k] = pd.read_csv(p, keep_default_na=False, dtype={"txn_id": str}) if p.exists() else None
    for n in ["ablation", "buckets"]:
        p = OUT / f"{n}.csv"
        d[n] = pd.read_csv(p) if p.exists() else None
    return d


# ---------------------------------------------------------------- sections
def kpis(d):
    a = d["ablation"]
    cards = []
    if a is not None:
        cards = [
            ("Precision (D)", fmt(num(a, "D", "precision")), "of flagged payments were truly bad"),
            ("Recall (D)", fmt(num(a, "D", "recall")), f"rules caught {fmt(num(a, 'A', 'recall'))}"),
            ("False-positive rate (D)", fmt(num(a, "D", "fpr"), True), f"rules: {fmt(num(a, 'A', 'fpr'))}"),
            ("Customer friction (D)", fmt(num(a, "D", "friction_rate")), "legit users interrupted"),
        ]
    if d["D"] is not None:
        lat = pd.to_numeric(d["D"]["latency_ms"], errors="coerce").median()
        cards.append(("Median scoring latency", f"{lat:.1f} ms", "per transaction, full pipeline"))
        cards.append(("Test transactions", f"{len(d['D']):,}", "scored chronologically"))
    return "".join(f"<div class='kpi'><div class='kl'>{t}</div><div class='kv'>{v}</div><div class='ks'>{s}</div></div>"
                   for t, v, s in cards)


def fig_actions(d):
    fig = go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                        subplot_titles=("A: Rules only", "D: Full RupeeTrace"))
    for i, k in enumerate(["A", "D"], 1):
        if d[k] is None:
            continue
        c = d[k]["action"].value_counts()
        fig.add_trace(go.Pie(labels=c.index, values=c.values, hole=0.6, name=k,
                             marker=dict(colors=[COL.get(a, "#888") for a in c.index])), 1, i)
    fig.update_layout(title="Decision distribution")
    return style(fig, 380)


def fig_ablation(d):
    a = d["ablation"]
    if a is None:
        return ""
    lab = "label" if "label" in a.columns else a.columns[0]
    cols = [c for c in ["precision", "recall", "f1", "scenario_recall"] if c in a.columns]
    m = a.melt(id_vars=lab, value_vars=cols, var_name="metric", value_name="value")
    f1 = px.bar(m, x=lab, y="value", color="metric", barmode="group", text_auto=".2f",
                title="Ablation: A rules → B behavior/anomaly → C graph → D provenance")
    cols2 = [c for c in ["fpr", "friction_rate"] if c in a.columns]
    m2 = a.melt(id_vars=lab, value_vars=cols2, var_name="metric", value_name="value")
    f2 = px.bar(m2, x=lab, y="value", color="metric", barmode="group", text_auto=".3f",
                title="Cost to legitimate customers (lower is better)")
    return to_div(style(f1, 440)) + to_div(style(f2, 340))


def fig_buckets(d):
    b = d["buckets"]
    if b is None:
        return ""
    out = ""
    for col, ttl in [("recall", "Recall by amount bucket (the low-value laundering claim)"),
                     ("precision", "Precision by amount bucket")]:
        f = px.bar(b, x="bucket", y=col, color="model", barmode="group", text_auto=".2f", title=ttl)
        out += to_div(style(f, 420))
    return out


def scenario_edges(d):
    tx = d["tx"]
    sc = pd.DataFrame()
    for col in ["scenario_id", "root_txn_id_gt"]:
        if col in tx.columns and nonempty(tx[col]).any():
            sc = tx[nonempty(tx[col])].copy()
            key = col
            break
    if sc.empty:
        return None, None
    pick = None
    if "typology" in sc.columns:
        fr = sc[sc["typology"].astype(str).str.contains("frag|struct|smurf|layer", case=False, regex=True)]
        if not fr.empty:
            pick = fr[key].value_counts().index[0]
    if pick is None:
        pick = sc[key].value_counts().index[0]
    return pick, sc[sc[key] == pick].copy()


def fig_lineage(d):
    pick, s = scenario_edges(d)
    if s is None or s.empty:
        return "<p class='muted'>No scenario rows found in transactions.csv.</p>", None
    e = s.groupby(["src_account", "dst_account"], as_index=False)["amount"].sum()
    e = e[e.src_account != e.dst_account]
    srcs = set(e.src_account) - set(e.dst_account)
    if not srcs:
        srcs = {e.src_account.iloc[0]}
    depth = {n: 0 for n in srcs}
    frontier = list(srcs)
    for _ in range(8):  # hop cap
        nxt = []
        for u in frontier:
            for v in e[e.src_account == u].dst_account:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    nxt.append(v)
        frontier = nxt
    e = e[e.apply(lambda r: r.src_account in depth and r.dst_account in depth
                  and depth[r.dst_account] > depth[r.src_account], axis=1)]
    e = e.sort_values("amount", ascending=False).head(45)
    names = ["Root origin"] + sorted(set(e.src_account) | set(e.dst_account), key=lambda n: depth.get(n, 0))
    idx = {n: i for i, n in enumerate(names)}
    labels = ["Root origin"] + [pid(n) for n in names[1:]]
    S, T, V = [], [], []
    for sa in srcs:
        if sa in idx:
            S.append(0); T.append(idx[sa]); V.append(float(s[s.src_account == sa].amount.sum()))
    for _, r in e.iterrows():
        S.append(idx[r.src_account]); T.append(idx[r.dst_account]); V.append(float(r.amount))
    fig = go.Figure(go.Sankey(arrangement="snap",
                              node=dict(label=labels, pad=14, thickness=14, color=CYAN),
                              link=dict(source=S, target=T, value=V, color="rgba(53,214,255,.30)")))
    fig.update_layout(title=f"Money lineage – scenario {pick}: one root split into fragments across relay accounts")
    n_acc = len(set(s.src_account) | set(s.dst_account))
    info = (f"<div class='note'>Scenario <b>{H.escape(str(pick))}</b>: {len(s):,} transactions, "
            f"{n_acc} accounts, total moved ₹{s.amount.sum():,.0f}, "
            f"median fragment ₹{s.amount.median():,.0f}. Fragments sit below the ₹50,000 alert threshold, "
            f"so per-transaction rules ignore them while provenance links them to one origin.</div>")
    return info + to_div(style(fig, 560)), s


def fig_network(s):
    if s is None or s.empty:
        return "<p class='muted'>No scenario data.</p>"
    G = nx.DiGraph()
    for r in s.itertuples():
        if r.src_account != r.dst_account:
            G.add_edge(r.src_account, r.dst_account)
    if G.number_of_nodes() > 60:
        keep = [n for n, _ in sorted(G.degree, key=lambda x: -x[1])[:60]]
        G = G.subgraph(keep).copy()
    pos = nx.spring_layout(G, seed=42, k=0.6)
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    # shared-device links
    dx, dy = [], []
    if "device_id" in s.columns:
        dev = s[s.src_account.isin(G.nodes)].groupby("device_id")["src_account"].unique()
        for accs in dev:
            accs = list(accs)[:6]
            for i in range(len(accs) - 1):
                a, b = accs[i], accs[i + 1]
                dx += [pos[a][0], pos[b][0], None]; dy += [pos[a][1], pos[b][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#31536D"), hoverinfo="none", name="payments"))
    if dx:
        fig.add_trace(go.Scatter(x=dx, y=dy, mode="lines", line=dict(width=2, color="#FFBE55", dash="dot"),
                                 hoverinfo="none", name="shared device"))
    nodes = list(G.nodes)
    deg = dict(G.degree)
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers",
                             marker=dict(size=[10 + 2 * min(deg[n], 8) for n in nodes], color=[deg[n] for n in nodes],
                                         colorscale="Tealgrn", line=dict(width=1, color="#B7F2FF")),
                             text=[f"{pid(n)}<br>links: {deg[n]}" for n in nodes], hoverinfo="text", name="accounts"))
    fig.update_layout(title=f"Account network – showing {len(nodes)} accounts (pseudonymous)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return to_div(style(fig, 600))


def story(d):
    if d["D"] is None or d["A"] is None:
        return ""
    pick, s = scenario_edges(d)
    if s is None or not set(s.txn_id) & set(d["D"].txn_id):
        s = d["tx"][d["tx"].label == 1]
    ids = set(s.txn_id)
    dd = d["D"][d["D"].txn_id.isin(ids)].merge(s[["txn_id", "amount"]], on="txn_id")
    aa = d["A"][d["A"].txn_id.isin(ids)][["txn_id", "action"]].rename(columns={"action": "action_A"})
    dd = dd.merge(aa, on="txn_id")
    if dd.empty:
        return ""
    dd["gap"] = (dd.amount - 4800).abs()
    r = dd.sort_values("gap").iloc[0]
    return (f"<div class='story'><div class='kl'>THE LOW-VALUE STORY</div>"
            f"<div class='big'>₹{r.amount:,.0f} transfer</div>"
            f"<div>Rules-only decision: <span class='pill {r.action_A}'>{r.action_A}</span> &nbsp; "
            f"RupeeTrace decision: <span class='pill {r.action}'>{r.action}</span> "
            f"(risk {float(r.risk_score):.2f})</div>"
            f"<div class='muted'>Same payment, below the ₹50,000 threshold. Only the lineage signal "
            f"connects it to the shared origin.</div></div>")


def csv_section(d):
    out = "<h3>Result tables (from the CSV files)</h3>"
    for k, ttl in [("ablation", "Ablation: A rules, B behavior/anomaly, C graph, D provenance"),
                   ("buckets", "Recall and precision by amount bucket")]:
        if d[k] is not None:
            out += f"<h4>{ttl}</h4><div class='tw'>{d[k].round(3).to_html(index=False, border=0)}</div>"
    files = ["ablation.csv", "buckets.csv", "metrics_summary.csv"] + list(PRED.values())
    out += "<h4>Open or download the files</h4><p>" + " &nbsp; ".join(
        f"<a style='color:#35D6FF' href='{f}'>{f}</a>" for f in files if (OUT / f).exists()) + "</p>"
    return out


def table(d):
    if d["D"] is None:
        return "<p class='muted'>Run with --run first.</p>"
    tx = d["tx"]
    m = tx.merge(d["D"][["txn_id", "risk_score", "action"]], on="txn_id")
    if d["A"] is not None:
        m = m.merge(d["A"][["txn_id", "action"]].rename(columns={"action": "action_A"}), on="txn_id", how="left")
    else:
        m["action_A"] = ""
    m["risk_score"] = pd.to_numeric(m["risk_score"], errors="coerce").fillna(0)
    m = m.sort_values("risk_score", ascending=False).head(400)
    rows = []
    for r in m.itertuples():
        def acct(x):
            return f"<span class='real'>{H.escape(str(x))}</span><span class='ps'>{pid(x)}</span>"
        rows.append(
            f"<tr data-a='{r.action}'><td>{H.escape(str(r.txn_id))}</td><td>{H.escape(str(r.ts))[:16]}</td>"
            f"<td>{acct(r.src_account)}</td><td>{acct(r.dst_account)}</td><td>₹{r.amount:,.0f}</td>"
            f"<td>{H.escape(str(r.channel))}</td><td><span class='pill {r.action_A}'>{r.action_A}</span></td>"
            f"<td><span class='pill {r.action}'>{r.action}</span></td><td>{r.risk_score:.2f}</td>"
            f"<td>{'yes' if r.label == 1 else 'no'}</td></tr>")
    head = ("<tr><th>Txn</th><th>Time</th><th>From</th><th>To</th><th>Amount</th><th>Channel</th>"
            "<th>Rules</th><th>RupeeTrace</th><th>Risk</th><th>Actually bad</th></tr>")
    sel = "".join(f"<option>{a}</option>" for a in ["ALL", "ALLOW", "STEP_UP", "HOLD", "BLOCK", "ESCALATE"])
    return (f"<div class='bar'><input id='q' placeholder='Search txn / account / channel'>"
            f"<select id='sa'>{sel}</select>"
            f"<label><input type='checkbox' id='an'> Analyst view (show real account IDs)</label></div>"
            f"<div class='tw'><table id='tt'>{head}{''.join(rows)}</table></div>"
            f"<div class='muted'>Top 400 test-split transactions by risk. 'Actually bad' is the synthetic ground truth, "
            f"used for evaluation only.</div>")


# ---------------------------------------------------------------- page
CSS = """
body{margin:0;background:linear-gradient(135deg,#07111F,#0B1C30);color:#EAF4FF;font-family:Segoe UI,Inter,Arial,sans-serif}
header{padding:22px 34px;border-bottom:1px solid #1D3A55}
h1{margin:0;font-size:1.7rem}h1 span{color:#35D6FF}.muted{color:#91A9C2;font-size:.88rem;margin-top:6px}
nav{display:flex;gap:6px;padding:10px 34px;background:#081624;border-bottom:1px solid #1D3A55;flex-wrap:wrap}
nav button{background:#122A43;color:#EAF4FF;border:1px solid #2A526F;border-radius:9px;padding:9px 16px;cursor:pointer;font-size:.92rem}
nav button.on,nav button:hover{border-color:#35D6FF;color:#35D6FF}
main{padding:22px 34px;max-width:1300px;margin:auto}.tab{display:none}.tab.on{display:block}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin-bottom:18px}
.kpi{background:#0E2034;border:1px solid #24425F;border-radius:14px;padding:15px 18px}
.kl{color:#91A9C2;font-size:.78rem;letter-spacing:.06em;text-transform:uppercase}.kv{font-size:1.9rem;font-weight:800;color:#35D6FF}.ks{color:#91A9C2;font-size:.8rem}
.card{background:#0E2034;border:1px solid #24425F;border-radius:14px;padding:12px 16px;margin-bottom:18px}
.story{background:linear-gradient(135deg,#122A43,#1B2F55);border:1px solid #35D6FF;border-radius:14px;padding:18px 22px;margin-bottom:18px}
.big{font-size:1.8rem;font-weight:800;margin:4px 0 8px}.note{background:#0E2034;border-left:4px solid #35D6FF;padding:10px 14px;margin-bottom:12px;border-radius:6px}
.pill{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.74rem;font-weight:700;color:#07111F;background:#888}
.ALLOW{background:#38D9A9}.STEP_UP{background:#FFBE55}.HOLD{background:#FF8F52}.BLOCK{background:#FF667D}.ESCALATE{background:#B69CFF}
.bar{display:flex;gap:12px;margin-bottom:12px;align-items:center;flex-wrap:wrap}
.bar input[type=text],#q,#sa{background:#0E2034;color:#EAF4FF;border:1px solid #2A526F;border-radius:8px;padding:8px 10px}#q{width:300px}
.tw{max-height:560px;overflow:auto;border:1px solid #24425F;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:.85rem}th{position:sticky;top:0;background:#122A43;text-align:left}
td,th{padding:7px 10px;border-bottom:1px solid #1D3A55}
.chip{background:#122A43;border:1px solid #2A526F;border-radius:8px;padding:5px 10px;font-size:.82rem}.chip.root{border-color:#B69CFF;color:#B69CFF}
.arr{color:#91A9C2;font-size:.78rem;margin:0 6px}.path{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 14px;line-height:2.2}
button.b{background:#122A43;color:#EAF4FF;border:1px solid #2A526F;border-radius:8px;padding:8px 14px;cursor:pointer}button.b:hover{border-color:#35D6FF}
#rtSel,#lvSp{background:#0E2034;color:#EAF4FF;border:1px solid #2A526F;border-radius:8px;padding:8px 10px}
.prog{height:6px;background:#122A43;border-radius:4px;margin-bottom:14px}.prog div{height:6px;background:#35D6FF;border-radius:4px;width:0}
.feed{max-height:420px;overflow:auto}.fr{display:flex;gap:14px;align-items:center;padding:6px 10px;border-bottom:1px solid #1D3A55;font-size:.85rem}
.fr.gain{background:rgba(56,217,169,.14);border-left:3px solid #38D9A9}.fr b{color:#FF667D}
.real{display:none}body.an .real{display:inline}body.an .ps{display:none}
"""

JS = """
function show(id,b){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
document.querySelectorAll('nav button').forEach(t=>t.classList.remove('on'));
document.getElementById(id).classList.add('on');b.classList.add('on');
setTimeout(()=>window.dispatchEvent(new Event('resize')),60);}
function filt(){const q=document.getElementById('q').value.toLowerCase(),a=document.getElementById('sa').value;
document.querySelectorAll('#tt tr[data-a]').forEach(r=>{r.style.display=((a==='ALL'||r.dataset.a===a)&&r.innerText.toLowerCase().includes(q))?'':'none'});}
const q=document.getElementById('q');if(q){q.oninput=filt;document.getElementById('sa').onchange=filt;
document.getElementById('an').onchange=e=>document.body.classList.toggle('an',e.target.checked);}
"""



HOP_CAP = 8
ALERT = 50000


def trace_rupees(s):
    """Proportional money-lot replay: every payment is split across the lots the sender holds."""
    s = s.copy()
    s["tt"] = pd.to_datetime(s["ts"], errors="coerce", utc=True)
    s = s.dropna(subset=["tt"]).sort_values("tt")
    hold, lots, roots, res = {}, {}, {}, {}
    k = 0
    for r in s.itertuples():
        A = float(r.amount)
        if A <= 0:
            continue
        src, dst, t = r.src_account, r.dst_account, r.tt
        bal = sum(l["amt"] for l in hold.get(src, []))
        used = min(A, bal)
        kids, moved = [], []
        if used > 0.005:
            take = [(l, used * l["amt"] / bal) for l in hold[src]]
            for l, p in take:
                l["amt"] -= p
            hold[src] = [l for l in hold[src] if l["amt"] > 0.005]
            for l, p in take:
                if p <= 0.005:
                    continue
                k += 1
                kids.append(dict(id=f"L{k:05d}", root=l["root"], origin=l["origin"], amt=p, amt0=p, acct=dst, t=t,
                                 hops=min(l["hops"] + 1, HOP_CAP), parents=[(l["id"], p)],
                                 dwell=(t - l["t"]).total_seconds() / 60, frm=src))
                moved.append(dict(lot=l["id"], root=l["root"], origin=l["origin"], amt=p, hops=l["hops"]))
        fresh = A - used
        if fresh > 0.005:
            k += 1
            org = str(r.origin_type).strip() or "UNKNOWN"
            kids.append(dict(id=f"L{k:05d}", root=r.txn_id, origin=org, amt=fresh, amt0=fresh, acct=dst, t=t,
                             hops=0, parents=[], dwell=0.0, frm=src))
            roots[r.txn_id] = {"amt": fresh, "txns": set(), "subtx": set(), "accts": set()}
            moved.append(dict(lot="NEW", root=r.txn_id, origin=org, amt=fresh, hops=0))
        for c in kids:
            lots[c["id"]] = c
            hold.setdefault(dst, []).append(c)
            ro = roots[c["root"]]
            ro["txns"].add(r.txn_id)
            if A < ALERT:
                ro["subtx"].add(r.txn_id)
            ro["accts"].add(dst)
        if kids:
            res[r.txn_id] = (kids, moved, A, src, dst, t)
    out = {}
    for tid, (kids, moved, A, src, dst, t) in res.items():
        dom = max(kids, key=lambda c: c["amt0"])
        chain = [dom]
        while chain[-1]["parents"]:
            chain.append(lots[max(chain[-1]["parents"], key=lambda x: x[1])[0]])
        chain.reverse()
        ro = roots[dom["root"]]
        path = [pid(chain[0]["frm"])] + [pid(n["acct"]) for n in chain]
        dw = [0] + [round(n["dwell"]) for n in chain[1:]]
        parts = sorted(moved, key=lambda x: -x["amt"])[:6]
        for p in parts:
            p["share"] = round(p["amt"] / A, 4)
            p["amt"] = round(p["amt"], 2)
        out[tid] = dict(
            root=dom["root"], origin=dom["origin"], root_agg=round(ro["amt"]), frags=len(ro["txns"]),
            sub_share=round(len(ro["subtx"]) / max(1, len(ro["txns"])), 3), accts=len(ro["accts"]), hops=dom["hops"],
            path=path, dw=dw, dwell_med=float(np.median(dw[1:])) if len(dw) > 1 else 0.0, parts=parts,
            nodes=[dict(id=n["id"], acct=pid(n["acct"]), amt=round(n["amt0"], 2),
                        t=n["t"].strftime("%d %b %H:%M"), hops=n["hops"]) for n in chain])
    return out


def jdump(o):
    return json.dumps(o, ensure_ascii=False).replace("</", "<\\/")


JS_RT = r"""
const fm=n=>Number(n).toLocaleString('en-IN',{maximumFractionDigits:0});
const pill=a=>`<span class="pill ${a}">${a}</span>`;
function rtShow(i){
 const t=RT[i],L=t.lin;
 const k=(a,b,c)=>`<div class="kpi"><div class="kl">${a}</div><div class="kv" style="font-size:1.45rem">${b}</div><div class="ks">${c||''}</div></div>`;
 let h='<div class="kpis">'+k('This payment','₹'+fm(t.amount),t.src+' → '+t.dst)
  +k('Root origin','₹'+fm(L.root_agg),L.origin+' · root '+L.root)
  +k('Fragments of that root',L.frags,Math.round(L.sub_share*100)+'% are below ₹50,000')
  +k('Hops from origin',L.hops,L.accts+' accounts touched by this root')
  +k('Median dwell',Math.round(L.dwell_med)+' min','time money sat in each account')
  +k('Share of root',(t.amount/Math.max(1,L.root_agg)*100).toFixed(2)+'%','of the original deposit')+'</div>';
 h+='<p>Rules-only: '+pill(t.A)+' &nbsp; RupeeTrace: '+pill(t.D)+' &nbsp; risk '+t.risk.toFixed(2)+(t.y?' &nbsp; <b>(actually fraud)</b>':'')+'</p>';
 h+='<h4>1. Where the rupees in this payment came from</h4><div class="tw"><table><tr><th>Rupees</th><th>Share</th><th>From lot</th><th>Root txn</th><th>Origin</th><th>Hops so far</th></tr>'
  +L.parts.map(p=>`<tr><td>₹${fm(p.amt)}</td><td>${(p.share*100).toFixed(1)}%</td><td>${p.lot==='NEW'?'new money entering':p.lot}</td><td>${p.root}</td><td>${p.origin}</td><td>${p.hops}</td></tr>`).join('')+'</table></div>';
 h+='<h4>2. Path of the main rupees, root to this payment</h4><div class="path">'
  +L.path.map((a,i)=>`<span class="chip${i===0?' root':''}">${i===0?'ORIGIN '+a:a}</span>`+(i<L.dw.length?`<span class="arr">${L.dw[i]} min →</span>`:'')).join('')+'</div>';
 h+='<h4>3. Lot-by-lot chain</h4><div class="tw"><table><tr><th>Lot</th><th>Account</th><th>Amount</th><th>Time</th><th>Hop</th></tr>'
  +L.nodes.map(n=>`<tr><td>${n.id}</td><td>${n.acct}</td><td>₹${fm(n.amt)}</td><td>${n.t}</td><td>${n.hops}</td></tr>`).join('')+'</table></div>';
 h+='<div class="note">Why this matters: this payment alone is ₹'+fm(t.amount)+', far below the ₹50,000 alert line. But it is 1 of '+L.frags+' fragments of one ₹'+fm(L.root_agg)+' origin spread over '+L.accts+' accounts, and that is what the provenance signal measures.</div>';
 document.getElementById('rtOut').innerHTML=h;
}
if(RT.length)rtShow(0);
"""

JS_LIVE = r"""
let li=0,timer=null,C={n:0,a:0,d:0,g:0,fp:0};
function liveRender(){for(const [id,v] of [['lvN',C.n],['lvA',C.a],['lvD',C.d],['lvG',C.g],['lvF',C.fp]])document.getElementById(id).textContent=v;
 document.getElementById('lvBar').style.width=(100*li/Math.max(1,LIVE.length))+'%';}
function liveStep(){
 if(li>=LIVE.length){livePause();return;}
 const e=LIVE[li++];C.n++;
 if(e.A!=='ALLOW')C.a++;
 if(e.D!=='ALLOW')C.d++;
 const gain=(e.D!=='ALLOW'&&e.A==='ALLOW'&&e.y==1);
 if(gain)C.g++;
 if(e.D!=='ALLOW'&&e.y==0)C.fp++;
 const f=document.getElementById('lvFeed'),row=document.createElement('div');
 row.className='fr'+(gain?' gain':'');
 row.innerHTML=`<span>${e.ts}</span><span>₹${fm(e.amt)}</span><span>${e.src} → ${e.dst}</span>${pill(e.A)} ${pill(e.D)}<span>risk ${e.risk.toFixed(2)}</span>${e.y?'<b>FRAUD</b>':''}`;
 f.prepend(row);while(f.children.length>40)f.removeChild(f.lastChild);
 liveRender();
}
function liveStart(){if(timer)return;const sp=+document.getElementById('lvSp').value;timer=setInterval(liveStep,Math.max(12,500/sp));}
function livePause(){clearInterval(timer);timer=null;}
function liveReset(){livePause();li=0;C={n:0,a:0,d:0,g:0,fp:0};document.getElementById('lvFeed').innerHTML='';liveRender();}
function liveSpeed(){if(timer){livePause();liveStart();}}
liveRender();
"""


def rupee_tab(d, s):
    if s is None or s.empty:
        return "<p class='muted'>No scenario data to trace.</p>", ""
    lin = trace_rupees(s)
    if not lin:
        return "<p class='muted'>Could not trace rupees for this scenario.</p>", ""
    amt = s.drop_duplicates("txn_id").set_index("txn_id")
    ids = list(lin.keys())
    first = min(ids, key=lambda i: abs(float(amt.loc[i, "amount"]) - 4800))
    pick = [first] + [ids[j] for j in np.linspace(0, len(ids) - 1, min(59, len(ids))).astype(int)]
    pick = list(dict.fromkeys(pick))
    pa = d["A"].set_index("txn_id") if d["A"] is not None else None
    pd_ = d["D"].set_index("txn_id") if d["D"] is not None else None
    RT = []
    for i in pick:
        r = amt.loc[i]
        RT.append(dict(id=i, amount=float(r["amount"]), src=pid(r["src_account"]), dst=pid(r["dst_account"]),
                       y=int(r["label"]),
                       A=str(pa.loc[i, "action"]) if pa is not None and i in pa.index else "n/a",
                       D=str(pd_.loc[i, "action"]) if pd_ is not None and i in pd_.index else "n/a",
                       risk=float(pd.to_numeric(pd_.loc[i, "risk_score"], errors="coerce")) if pd_ is not None and i in pd_.index else 0.0,
                       lin=lin[i]))
    opts = "".join(f"<option value='{n}'>{x['id']} · ₹{x['amount']:,.0f} · {x['D']}</option>" for n, x in enumerate(RT))
    html = ("<h3>Rupee-to-rupee trace</h3><div class='muted'>Every payment is split across the money lots the sender holds "
            "(proportional split, hop cap 8). Pick a payment to see exactly which earlier rupees it is made of. "
            "This view replays the scenario inside this page; it is not the engine's own LotLedger output.</div>"
            f"<div class='bar'><select id='rtSel' onchange='rtShow(this.value)'>{opts}</select></div><div id='rtOut'></div>")
    return html, "const RT=" + jdump(RT) + ";\n" + JS_RT


def live_tab(d):
    if d["D"] is None:
        return "<p class='muted'>Needs predictions.</p>", ""
    m = d["tx"].merge(d["D"][["txn_id", "action", "risk_score"]], on="txn_id")
    if d["A"] is not None:
        m = m.merge(d["A"][["txn_id", "action"]].rename(columns={"action": "A"}), on="txn_id", how="left")
    else:
        m["A"] = "ALLOW"
    m["A"] = m["A"].fillna("ALLOW")
    m["risk_score"] = pd.to_numeric(m["risk_score"], errors="coerce").fillna(0)
    fl = (m.action != "ALLOW") | (m.A != "ALLOW")
    a, b = m[fl], m[~fl]
    m = pd.concat([a.sample(min(300, len(a)), random_state=42), b.sample(min(250, len(b)), random_state=42)]).sort_values("ts")
    LIVE = [dict(ts=str(r.ts)[5:16], amt=float(r.amount), src=pid(r.src_account), dst=pid(r.dst_account),
                 A=r.A, D=r.action, risk=float(r.risk_score), y=int(r.label)) for r in m.itertuples()]
    card = lambda i, t: f"<div class='kpi'><div class='kl'>{t}</div><div class='kv' id='{i}'>0</div></div>"
    html = ("<h3>Live replay</h3><div class='muted'>Replays real scored test transactions in time order, in the browser. "
            "It shows saved results from the engine run, not a live engine. Sample: all kinds of flagged payments plus some normal ones.</div>"
            "<div class='bar'><button class='b' onclick='liveStart()'>▶ Start</button><button class='b' onclick='livePause()'>❚❚ Pause</button>"
            "<button class='b' onclick='liveReset()'>↺ Reset</button><select id='lvSp' onchange='liveSpeed()'>"
            "<option value='1'>1x</option><option value='5' selected>5x</option><option value='20'>20x</option></select></div>"
            "<div class='prog'><div id='lvBar'></div></div>"
            "<div class='kpis'>" + card("lvN", "Processed") + card("lvA", "Flagged by rules") + card("lvD", "Flagged by RupeeTrace")
            + card("lvG", "Fraud caught by RupeeTrace, missed by rules") + card("lvF", "Legit flagged by RupeeTrace") + "</div>"
            "<div id='lvFeed' class='feed'></div>")
    return html, "const LIVE=" + jdump(LIVE) + ";\n" + JS_LIVE


def improve_tab(d):
    if d["D"] is None or d["A"] is None:
        return "<p class='muted'>Needs predictions.</p>"
    tx = d["tx"]
    keep = ["txn_id", "amount", "label"] + [c for c in ["typology"] if c in tx.columns]
    m = tx[keep].merge(d["D"][["txn_id", "action", "risk_score"]].rename(columns={"action": "D"}), on="txn_id")
    for k in "ABC":
        if d[k] is not None:
            m = m.merge(d[k][["txn_id", "action"]].rename(columns={"action": k}), on="txn_id", how="left")
    fl = lambda c: (m[c].fillna("ALLOW") != "ALLOW") if c in m.columns else pd.Series(False, index=m.index)
    y = m.label == 1
    big = m.amount >= ALERT
    V = {"A: Rules": fl("A"), "B: Behavior+Anomaly": fl("B"), "C: + Graph": fl("C"), "D: RupeeTrace": fl("D"),
         "D + keep rule for amounts ≥ ₹50k": fl("D") | (big & fl("A")),
         "B or D (either flags)": fl("B") | fl("D"),
         "D or A (either flags)": fl("D") | fl("A")}
    rows = []
    for n, f in V.items():
        tp, fp = int((f & y).sum()), int((f & ~y).sum())
        rows.append((n, tp / max(1, tp + fp), tp / max(1, int(y.sum())), fp / max(1, int((~y).sum())), f.mean()))
    best = max(r[2] for r in rows)
    tb = "".join(f"<tr><td>{n}</td><td>{p:.3f}</td><td{' style=\"color:#38D9A9;font-weight:700\"' if r == best else ''}>{r:.3f}</td>"
                 f"<td>{fp:.4f}</td><td>{fr:.3f}</td></tr>" for n, p, r, fp, fr in rows)
    out = ("<h3>Where accuracy can still improve</h3><div class='muted'>Computed here with one definition: a payment is 'flagged' "
           "when its action is anything other than ALLOW. This can differ slightly from eval/metrics.py. These are what-if combinations "
           "of the saved predictions, not retrained models.</div>"
           f"<div class='tw'><table><tr><th>Variant</th><th>Precision</th><th>Recall</th><th>False-positive rate</th><th>Flagged share</th></tr>{tb}</table></div>")
    if "typology" in m.columns:
        g = m[y].assign(missed=lambda x: ~fl("D")[x.index]).groupby("typology").agg(frauds=("label", "size"), missed=("missed", "sum")).reset_index()
        g["recall"] = 1 - g.missed / g.frauds
        out += "<h4>RupeeTrace (D) recall by fraud type</h4><div class='tw'>" + g.sort_values("recall").round(3).to_html(index=False, border=0) + "</div>"
    miss = m[y & ~fl("D")].sort_values("amount", ascending=False).head(25)
    cols = [c for c in ["txn_id", "amount", "typology", "A", "B", "C", "D", "risk_score"] if c in miss.columns]
    out += "<h4>Fraud that RupeeTrace missed (largest 25)</h4><div class='tw'>" + miss[cols].round(2).to_html(index=False, border=0) + "</div>"
    out += ("<div class='note'>To really raise accuracy you must change the models, which this page cannot do: train on more fraud rows, "
            "retune the fusion weights and thresholds on a validation slice, and feed the provenance features from the lot trace into the stacker. "
            "Use the tables above to see which fraud types and amount ranges to target first.</div>")
    return out


def build():
    fix_predictions()
    d = load()
    lineage_html, scen = fig_lineage(d)
    rt_html, rt_js = rupee_tab(d, scen)
    lv_html, lv_js = live_tab(d)
    tabs = [
        ("ov", "Overview", f"<div class='kpis'>{kpis(d)}</div>{story(d)}<div class='card'>{to_div(fig_actions(d))}</div>"),
        ("rs", "Results", f"<div class='card'>{csv_section(d)}</div><div class='card'>{fig_ablation(d)}</div><div class='card'>{fig_buckets(d)}</div>"),
        ("ln", "Money lineage", f"<div class='card'>{lineage_html}</div>"),
        ("rt", "Rupee trace", f"<div class='card'>{rt_html}</div>"),
        ("lv", "Live replay", f"<div class='card'>{lv_html}</div>"),
        ("nw", "Account network", f"<div class='card'>{fig_network(scen)}</div>"),
        ("tx", "Transactions", f"<div class='card'>{table(d)}</div>"),
        ("im", "Improve accuracy", f"<div class='card'>{improve_tab(d)}</div>"),
    ]
    nav = "".join(f"<button class='{'on' if i == 0 else ''}' onclick=\"show('{t[0]}',this)\">{t[1]}</button>"
                  for i, t in enumerate(tabs))
    body = "".join(f"<section id='{t[0]}' class='tab {'on' if i == 0 else ''}'>{t[2]}</section>" for i, t in enumerate(tabs))
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>RupeeTrace</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style><script>{get_plotlyjs()}</script></head><body>"
            f"<header><h1>₹ Rupee<span>Trace</span></h1>"
            f"<div class='muted'>Real-time fraud and AML risk orchestration · rupee-level provenance · synthetic data, reduced test split</div></header>"
            f"<nav>{nav}</nav><main>{body}</main><script>{JS}{rt_js}{lv_js}</script></body></html>")
    out = OUT / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    print("Dashboard written:", out)
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_pipeline()
    build()
