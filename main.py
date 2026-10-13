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


def build():
    fix_predictions()
    d = load()
    lineage_html, scen = fig_lineage(d)
    tabs = [
        ("ov", "Overview", f"<div class='kpis'>{kpis(d)}</div>{story(d)}<div class='card'>{to_div(fig_actions(d))}</div>"),
        ("rs", "Results", f"<div class='card'>{csv_section(d)}</div><div class='card'>{fig_ablation(d)}</div><div class='card'>{fig_buckets(d)}</div>"),
        ("ln", "Money lineage", f"<div class='card'>{lineage_html}</div>"),
        ("nw", "Account network", f"<div class='card'>{fig_network(scen)}</div>"),
        ("tx", "Transactions", f"<div class='card'>{table(d)}</div>"),
    ]
    nav = "".join(f"<button class='{'on' if i == 0 else ''}' onclick=\"show('{t[0]}',this)\">{t[1]}</button>"
                  for i, t in enumerate(tabs))
    body = "".join(f"<section id='{t[0]}' class='tab {'on' if i == 0 else ''}'>{t[2]}</section>" for i, t in enumerate(tabs))
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>RupeeTrace</title>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style><script>{get_plotlyjs()}</script></head><body>"
            f"<header><h1>₹ Rupee<span>Trace</span></h1>"
            f"<div class='muted'>Real-time fraud and AML risk orchestration · rupee-level provenance · synthetic data, reduced test split</div></header>"
            f"<nav>{nav}</nav><main>{body}</main><script>{JS}</script></body></html>")
    out = OUT / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    print("Dashboard written:", out)
    webbrowser.open(out.as_uri())


if __name__ == "__main__":
    if "--run" in sys.argv:
        run_pipeline()
    build()