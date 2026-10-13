"""Plotly figure builders for the RupeeTrace dashboard.

Pure functions: dicts/DataFrame-like in, plotly Figure out. No network, no global state, inputs never mutated.
Privacy: account identifiers are pseudonymous unless the analyst view is on. When it is off, raw account IDs and
device IDs are never written into the figure (labels, hover text, ids), whatever the `lab` callable returns.
"""
from __future__ import annotations

import math

import plotly.graph_objects as go

from dashboard.components.ui import COMPONENT_LABEL, action_color, inr, local_pseudonym, safe_float

COMPONENTS = ["behavior", "anomaly", "graph", "provenance"]
MAX_NODES = 60          # network graph cap
MAX_TRAIL = 12          # Sankey account-trail cap (HOP_CAP is 8, so this only guards bad input)
BANDS = [(0.0, 0.30, "#d6f0de"), (0.30, 0.55, "#fbeec2"), (0.55, 0.80, "#fbd9c4"), (0.80, 1.0, "#f6c6c6")]
BUCKET_PALETTE = ["#868e96", "#4c6ef5", "#15aabf", "#7b3fe4"]
_TRANSPARENT = "rgba(0,0,0,0)"


def _base(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=44 if title else 14, b=10), paper_bgcolor=_TRANSPARENT,
                      plot_bgcolor=_TRANSPARENT, title=dict(text=title, x=0.01, font=dict(size=14)) if title else None)
    return fig


def empty_figure(message: str, height: int = 240) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#868e96"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base(fig, height)


def _label(lab, account, analyst) -> str:
    """analyst False/None-with-no-lab -> pseudonym; otherwise whatever lab returns (identity in analyst view)."""
    if analyst is False or lab is None:
        return local_pseudonym(account)
    try:
        return str(lab(account))
    except Exception:
        return local_pseudonym(account)


def _score(v) -> float:
    if isinstance(v, dict):
        v = v.get("score", v.get("contribution"))
    return max(0.0, min(1.0, safe_float(v)))


# ------------------------------------------------------------------------------ gauge
def risk_gauge(score, action="ALLOW") -> go.Figure:
    """Semicircular 0-1 risk gauge, bar coloured by action; bands at 0.30 / 0.55 / 0.80."""
    s = max(0.0, min(1.0, safe_float(score)))
    a = str(action).upper() if action else "ALLOW"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=s, number=dict(valueformat=".2f"),
        title=dict(text=f"Risk score · {a}", font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, 1], tickvals=[0, 0.3, 0.55, 0.8, 1]), bar=dict(color=action_color(a)),
                   steps=[dict(range=[lo, hi], color=c) for lo, hi, c in BANDS])))
    return _base(fig, 230).update_layout(margin=dict(l=20, r=20, t=50, b=10))


# ------------------------------------------------------------------- component charts
def component_breakdown(peak_components=None, avg=None) -> go.Figure:
    """Grouped horizontal bars per signal: peak payment's raw score vs the case-average contribution."""
    peak_components, avg = peak_components or {}, avg or {}
    if not peak_components and not avg:
        return empty_figure("No component scores available")
    labels = [COMPONENT_LABEL[c] for c in COMPONENTS]
    fig = go.Figure()
    fig.add_bar(y=labels, x=[_score(peak_components.get(c)) for c in COMPONENTS], orientation="h", name="Peak payment: signal score",
                marker_color="#7b3fe4", hovertemplate="%{y}: %{x:.2f}<extra>peak score</extra>")
    fig.add_bar(y=labels, x=[_score(avg.get(c)) for c in COMPONENTS], orientation="h", name="Case average: contribution to risk",
                marker_color="#15aabf", hovertemplate="%{y}: %{x:.2f}<extra>avg contribution</extra>")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(range=[0, 1], title="0 - 1")
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 280, "Score breakdown by signal")


def decision_bars(components=None) -> go.Figure:
    """Per-signal score and contribution for one decision. Accepts {c: {score, contribution}} or {c: float}."""
    components = components or {}
    if not components:
        return empty_figure("No component scores available")
    labels = [COMPONENT_LABEL[c] for c in COMPONENTS]
    sc = [_score(components.get(c)) for c in COMPONENTS]
    ct = [max(0.0, min(1.0, safe_float((components.get(c) or {}).get("contribution") if isinstance(components.get(c), dict) else None)))
          for c in COMPONENTS]
    fig = go.Figure()
    fig.add_bar(x=labels, y=sc, name="Signal score", marker_color="#7b3fe4", hovertemplate="%{x}: %{y:.2f}<extra>score</extra>")
    fig.add_bar(x=labels, y=ct, name="Contribution to risk", marker_color="#15aabf", hovertemplate="%{x}: %{y:.2f}<extra>contribution</extra>")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(range=[0, 1])
    return _base(fig, 300, "Why this score")


# ---------------------------------------------------------------------------- lineage
def _fmt_dwell(m) -> str | None:
    if m is None:
        return None
    m = safe_float(m, -1)
    if m < 0:
        return None
    return f"{m:.0f} min" if m < 120 else f"{m / 60:.1f} h"


def lineage_sankey(case=None, lin=None, lab=None, analyst=None) -> go.Figure:
    """Root origin -> fragments hub -> account trail (one node per hop position).

    Link widths are uniform by design: a root of Rs 5,00,000 split into Rs 4,800 fragments would make the trail
    invisible at true scale, so true amounts and dwell times are in node labels and hover text instead.
    `lab` maps account id -> display label; pass analyst=False to force pseudonyms regardless of `lab`.
    """
    case, lin = case or {}, lin or {}
    path = list(lin.get("account_path") or [])
    nodes = list(lin.get("nodes") or [])
    if not path and nodes:
        path = [n.get("account") for n in nodes if n.get("account") is not None]
    if not path:
        return empty_figure("No lineage available for this transaction")
    note = ""
    if len(path) > MAX_TRAIL:
        note = f" (first {MAX_TRAIL} of {len(path)} accounts shown)"
        path = path[:MAX_TRAIL]
    dwell = list(lin.get("dwell_minutes") or [])
    root_amt = lin.get("root_aggregate_amount") or case.get("root_amount")
    frags = int(max(safe_float(lin.get("fragment_count")), safe_float(case.get("fragments"))))
    hops = lin.get("hops")
    hops = int(safe_float(hops)) if hops is not None else max(0, len(path) - 1)
    origin = str(lin.get("origin_type") or case.get("origin_type") or "UNKNOWN")
    ok = lin.get("hash_chain_ok")
    chain = "hash chain intact ✅" if ok is True else "hash chain BROKEN ⚠️" if ok is False else "hash chain unknown"

    labels = [f"ROOT ORIGIN\n{origin}\n{inr(root_amt)}", f"{frags} fragments" if frags else "Fragments"]
    colors = ["#7b3fe4", "#868e96"]
    for i, acct in enumerate(path):
        txt = _label(lab, acct, analyst)
        d = _fmt_dwell(dwell[i - 1]) if 0 < i <= len(dwell) else None
        if d:
            txt += f"\nheld {d}"
        labels.append(txt)
        colors.append("#d63031" if i == len(path) - 1 and i > 0 else "#4c6ef5")
    src, tgt, hover = [0, 1], [1, 2], [f"Root {inr(root_amt)} split into {frags} fragments", "Fragment enters first account"]
    for i in range(len(path) - 1):
        n = nodes[i] if i < len(nodes) and isinstance(nodes[i], dict) else {}
        d = _fmt_dwell(dwell[i - 1]) if 0 < i <= len(dwell) else None
        src.append(2 + i)
        tgt.append(3 + i)
        hover.append(f"Hop {i + 1}: {inr(n.get('amount'))}" + (f" · dwell before hop {d}" if d else ""))
    vals = [3.0] + [1.0] * (len(src) - 1)
    fig = go.Figure(go.Sankey(
        arrangement="snap", node=dict(label=labels, color=colors, pad=22, thickness=18),
        link=dict(source=src, target=tgt, value=vals, customdata=hover, hovertemplate="%{customdata}<extra></extra>",
                  color="rgba(123,63,228,0.25)")))
    return _base(fig, 360, f"Money lineage · {hops} hops · {frags} fragments · {chain}{note}")


# ----------------------------------------------------------------------------- network
def _layout(ids: list, edges: list[tuple]) -> dict:
    n = len(ids)
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(ids)
        g.add_edges_from(edges)
        pos = nx.spring_layout(g, seed=42, k=1.2 / math.sqrt(max(n, 1)), iterations=60)
        return {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
    except Exception:  # deterministic circle fallback
        return {a: (math.cos(2 * math.pi * i / max(n, 1)), math.sin(2 * math.pi * i / max(n, 1))) for i, a in enumerate(ids)}


def network_graph(graph=None, lab=None, analyst: bool = False, highlight=None) -> go.Figure:
    """Related-accounts graph, capped at MAX_NODES (top by degree). Shared-device links drawn thick red.

    graph: {nodes:[{id,pseudonym,degree,device,on_watchlist}], edges:[{src,dst,amount,count,kind,device?}], truncated, total_accounts}
    analyst False -> only pseudonyms, no device IDs anywhere in the figure; True -> labels from `lab`, devices in hover.
    highlight: set of raw account ids (e.g. the traced lineage path) drawn with an orange ring.
    """
    graph = graph or {}
    analyst = bool(analyst)
    raw_nodes = [n for n in (graph.get("nodes") or []) if isinstance(n, dict) and n.get("id") is not None]
    if not raw_nodes:
        return empty_figure("No related accounts to display")
    total = int(graph.get("total_accounts") or len(raw_nodes))
    seen, nodes = set(), []
    for n in sorted(raw_nodes, key=lambda n: -safe_float(n.get("degree"))):
        if n["id"] not in seen:
            seen.add(n["id"])
            nodes.append(n)
    nodes = nodes[:MAX_NODES]
    keep = {n["id"] for n in nodes}
    truncated = bool(graph.get("truncated")) or len(seen) > len(nodes) or total > len(nodes)
    highlight = set(highlight or [])
    edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict) and e.get("src") in keep and e.get("dst") in keep and e["src"] != e["dst"]]
    pos = _layout([n["id"] for n in nodes], [(e["src"], e["dst"]) for e in edges])

    def disp(n):
        return (_label(lab, n["id"], True) if analyst else str(n.get("pseudonym") or local_pseudonym(n["id"])))

    names = {n["id"]: disp(n) for n in nodes}
    fig = go.Figure()
    for kind, name, color, width, dash in (("payment", "Payment", "rgba(134,142,150,0.55)", 1.2, "solid"),
                                           ("shared_device", "Shared device link", "#d63031", 3.5, "dash")):
        es = [e for e in edges if (e.get("kind") == "shared_device") == (kind == "shared_device")]
        if not es:
            continue
        xs, ys, mx, my, tx = [], [], [], [], []
        for e in es:
            (x0, y0), (x1, y1) = pos[e["src"]], pos[e["dst"]]
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            mx.append((x0 + x1) / 2)
            my.append((y0 + y1) / 2)
            if kind == "payment":
                tx.append(f"{names[e['src']]} → {names[e['dst']]}<br>{inr(e.get('amount'))} in {int(safe_float(e.get('count')))} payment(s)")
            else:
                dv = f"<br>device {e.get('device')}" if analyst and e.get("device") else ""
                tx.append(f"Shared device: {names[e['src']]} ↔ {names[e['dst']]}{dv}")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", name=name))
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=8, opacity=0), text=tx, hoverinfo="text", showlegend=False))
    hx = [pos[n["id"]][0] for n in nodes]
    hy = [pos[n["id"]][1] for n in nodes]
    colors, lines, sizes, hov = [], [], [], []
    for n in nodes:
        wl, hl = bool(n.get("on_watchlist")), n["id"] in highlight
        colors.append("#d63031" if wl else "#4c6ef5")
        lines.append("#e8590c" if hl else "#ffffff")
        sizes.append(10 + 5 * math.log1p(max(0.0, safe_float(n.get("degree")))) + (4 if hl else 0))
        t = f"{names[n['id']]}<br>activity: {int(safe_float(n.get('degree')))} payments"
        if analyst and n.get("device"):
            t += f"<br>device: {n['device']}"
        if wl:
            t += "<br>on watchlist"
        if hl:
            t += "<br>on traced money path"
        hov.append(t)
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers+text" if len(nodes) <= 25 else "markers", text=[names[n["id"]] for n in nodes],
                             textposition="top center", textfont=dict(size=10), hovertext=hov, hoverinfo="text", name="Account",
                             marker=dict(size=sizes, color=colors, line=dict(color=lines, width=2.5))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(legend=dict(orientation="h", y=-0.05), hovermode="closest")
    if truncated:
        fig.add_annotation(text=f"Showing top {len(nodes)} of {max(total, len(nodes))} accounts", xref="paper", yref="paper", x=0, y=1.0,
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color="#868e96"))
    return _base(fig, 460)


# ----------------------------------------------------------------------------- metrics
def recall_by_bucket(variants=None) -> go.Figure:
    """Grouped bars: recall per amount bucket for each ablation variant (from metrics['variants'])."""
    variants = variants or {}
    fig = go.Figure()
    for i, (key, v) in enumerate(variants.items()):
        rows = (v or {}).get("recall_by_bucket") or []
        if not rows:
            continue
        fig.add_bar(x=[str(r.get("bucket", "?")) for r in rows], y=[None if r.get("recall") is None else safe_float(r.get("recall")) for r in rows],
                    name=str((v or {}).get("label") or key), marker_color=BUCKET_PALETTE[i % len(BUCKET_PALETTE)],
                    customdata=[int(safe_float(r.get("n"))) for r in rows],
                    hovertemplate="%{x}<br>recall %{y:.0%} (n=%{customdata} laundering txns)<extra>%{fullData.name}</extra>")
    if not fig.data:
        return empty_figure("No recall-by-amount data yet")
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Recall (any intervention)")
    return _base(fig, 340, "Recall by transaction amount bucket")
