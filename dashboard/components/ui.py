"""HTML helpers for the dashboard: CSS, action badges, INR formatting, decision-feed table.

Pure functions returning strings. Every dynamic value is HTML-escaped. Never raises on missing/partial input.
"""
from __future__ import annotations

import hashlib
import html
import math

ACTION_COLOR = {"ALLOW": "#2e9e5b", "STEP_UP": "#e0a100", "HOLD": "#e8590c", "BLOCK": "#d63031", "ESCALATE": "#7b3fe4"}
ACTION_TEXT = {"STEP_UP": "#1d1d1d"}  # dark text on yellow for contrast; white elsewhere
ACTION_LABEL = {"ALLOW": "Allow", "STEP_UP": "Step-up (OTP)", "HOLD": "Hold", "BLOCK": "Block", "ESCALATE": "Escalate"}
COMPONENT_LABEL = {"behavior": "Behaviour", "anomaly": "Anomaly", "graph": "Graph", "provenance": "Provenance"}
UNKNOWN_COLOR = "#868e96"

CSS = """<style>
.badge{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-weight:600;font-size:.82rem;letter-spacing:.2px;white-space:nowrap}
.kpi{border:1px solid rgba(128,128,128,.25);border-radius:10px;padding:10px 12px;background:rgba(128,128,128,.06)}
.story{border-left:4px solid #7b3fe4;background:rgba(123,63,228,.07);padding:10px 14px;border-radius:6px;margin:8px 0}
table.feed{width:100%;border-collapse:collapse;font-size:.85rem}
table.feed th{text-align:left;padding:4px 8px;border-bottom:2px solid rgba(128,128,128,.35);opacity:.75;font-weight:600}
table.feed td{padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.18);vertical-align:middle}
table.feed .num{text-align:right;font-variant-numeric:tabular-nums}
.riskbar{display:inline-block;height:6px;border-radius:3px;vertical-align:middle;margin-right:6px}
.feed-empty{opacity:.6;padding:12px;font-style:italic}
</style>"""


def esc(x) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def local_pseudonym(account_id) -> str:
    """Same scheme as engine.engine.pseudonym (kept local so this module has no engine dependency)."""
    s = str(account_id)
    h = hashlib.sha1(("rt-salt:" + s).encode()).hexdigest()[:6].upper()
    return ("EXT-" if s.startswith("EXT_") else "ACC-") + h


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) or math.isinf(v) else v


def action_color(action) -> str:
    return ACTION_COLOR.get(str(action).upper() if action is not None else "", UNKNOWN_COLOR)


def badge(action, label: str | None = None) -> str:
    """Coloured pill for an action. ALLOW green, STEP_UP yellow, HOLD orange, BLOCK red, ESCALATE purple."""
    a = str(action).upper() if action is not None else ""
    text = label if label is not None else ACTION_LABEL.get(a, a or "UNKNOWN")
    return (f'<span class="badge" style="background:{action_color(a)};color:{ACTION_TEXT.get(a, "#fff")}">'
            f'{esc(text)}</span>')


def inr(x) -> str:
    """Indian-grouped rupee string, e.g. 500000 -> '₹5,00,000'. None/NaN/garbage -> '–'."""
    if x is None:
        return "–"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    if math.isnan(v) or math.isinf(v):
        return "–"
    sign = "-" if v < 0 else ""
    v = abs(v)
    whole, frac = divmod(round(v, 2), 1)
    s = str(int(whole))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    cents = round(frac * 100)
    return f"{sign}₹{s}" + (f".{int(cents):02d}" if cents else "")


def _pseudo(fn, account) -> str:
    if fn is not None:
        try:
            return str(fn(account))
        except Exception:
            pass
    return local_pseudonym(account)


def feed_row(d: dict, pseudonym_fn=None, show_gt: bool = False) -> str:
    """One <tr> for a decision dict (live or recorded). Account IDs go through pseudonym_fn."""
    d = d or {}
    risk = max(0.0, min(1.0, safe_float(d.get("risk_score"))))
    action = d.get("action")
    reasons = d.get("reasons") or []
    reason = str(reasons[0]) if isinstance(reasons, (list, tuple)) and reasons else ""
    if len(reason) > 90:
        reason = reason[:87] + "..."
    ts = str(d.get("ts") or "")[:19].replace("T", " ")
    src, dst = d.get("src_account"), d.get("dst_account")
    frm = _pseudo(pseudonym_fn, src) if src is not None else "–"
    to = _pseudo(pseudonym_fn, dst) if dst is not None else "–"
    cells = [f"<td>{esc(ts) or '–'}</td>", f"<td>{esc(frm)} → {esc(to)}</td>", f'<td class="num">{esc(inr(d.get("amount")))}</td>',
             f"<td>{esc(d.get('channel') or '–')}</td>",
             f'<td><span class="riskbar" style="width:{int(risk * 60)}px;background:{action_color(action)}"></span>{risk:.2f}</td>',
             f"<td>{badge(action)}</td>", f"<td>{esc(reason)}</td>"]
    if show_gt:
        gt = d.get("label_gt")
        cells.append(f"<td>{'laundering' if gt == 1 else 'legit' if gt == 0 else '–'}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def feed_table(feed, pseudonym_fn=None, show_gt: bool = False) -> str:
    """HTML table of recent decisions (newest first as given). Empty feed -> a placeholder div."""
    if not feed:
        return '<div class="feed-empty">No transactions scored yet. Press Start.</div>'
    head = ["Time (UTC)", "From → To", "Amount", "Channel", "Risk", "Action", "Top reason"] + (["Ground truth (demo)"] if show_gt else [])
    rows = "".join(feed_row(d, pseudonym_fn, show_gt) for d in feed)
    return '<table class="feed"><thead><tr>' + "".join(f"<th>{h}</th>" for h in head) + f"</tr></thead><tbody>{rows}</tbody></table>"
