"""Karpathy-style autoresearch dashboard for one or more lineage.jsonl files.

Reads `lineage.jsonl` files and renders a self-contained HTML page with:
  - Per-slot summary card (slot, totals, best champion)
  - Champion progression: solve_ms vs proposal index, with KEPT points
    highlighted as stars and discarded points faded
  - Lineage chain ASCII (each kept proposal is a child of the previous)
  - Discard-reason bar chart
  - Auto-refresh meta so live runs update in place

Pure HTML + inline SVG. No JS dependencies, no server needed — open the
output file in a browser, or `python -m http.server` from the directory.

Usage:
  python scripts/lineage_dashboard.py PATH [PATH ...] [--out FILE] [--refresh SEC]

Each PATH is a directory containing `lineage.jsonl` or the file itself.
Default output is `lineage_dashboard.html` in CWD; `--refresh` adds a
meta-refresh tag (default: 0 = no refresh).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from html import escape
from pathlib import Path


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _slot_label(slot: dict) -> str:
    return f"{slot.get('problem_family')}/{slot.get('algorithm')} ({slot.get('hardware')}/{slot.get('dtype')})"


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    slot = rows[0].get("slot", {})
    kept_rows = [r for r in rows if (r.get("decision") or {}).get("accepted")]
    crash_rows = [r for r in rows if "crash" in (r.get("decision") or {}).get("reason", "")]
    discard_rows = [
        r for r in rows
        if not (r.get("decision") or {}).get("accepted")
        and "crash" not in (r.get("decision") or {}).get("reason", "")
    ]

    best = None
    best_ms = math.inf
    for r in kept_rows:
        s = r.get("score") or {}
        ms = s.get("solve_ms_median")
        if isinstance(ms, (int, float)) and ms < best_ms:
            best, best_ms = r, ms

    first_kept_ms = None
    if kept_rows:
        s = kept_rows[0].get("score") or {}
        first_kept_ms = s.get("solve_ms_median")

    reasons = Counter((r.get("decision") or {}).get("reason", "?") for r in rows)
    return {
        "slot": slot,
        "total": len(rows),
        "kept": len(kept_rows),
        "discarded": len(discard_rows),
        "crashed": len(crash_rows),
        "kept_rows": kept_rows,
        "best": best,
        "best_ms": best_ms if math.isfinite(best_ms) else None,
        "first_kept_ms": first_kept_ms,
        "speedup": (
            first_kept_ms / best_ms
            if isinstance(first_kept_ms, (int, float)) and math.isfinite(best_ms) and best_ms > 0
            else None
        ),
        "reasons": reasons,
        "rows": rows,
    }


def _svg_progression(summary: dict, *, width: int = 900, height: int = 280) -> str:
    """Plot solve_ms (y-log) vs proposal index (x). Discarded faded, kept as stars,
    champion track as a line connecting accepted points.
    """
    rows = summary["rows"]
    kept_rows = summary["kept_rows"]
    if not rows:
        return f'<svg width="{width}" height="{height}"><text x="20" y="40">no data</text></svg>'

    # Collect (idx, ms, kept_or_crash). Skip rows with no score (crash rows).
    pts: list[tuple[int, float, str, str]] = []
    for i, r in enumerate(rows):
        s = r.get("score") or {}
        ms = s.get("solve_ms_median")
        decision = (r.get("decision") or {}).get("reason", "")
        if "crash" in decision:
            kind = "crash"
        elif (r.get("decision") or {}).get("accepted"):
            kind = "kept"
        else:
            kind = "discard"
        if isinstance(ms, (int, float)) and ms > 0:
            pts.append((i + 1, float(ms), kind, decision))
    if not pts:
        return f'<svg width="{width}" height="{height}"><text x="20" y="40">no scored points</text></svg>'

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) * 0.8, max(ys) * 1.2
    log_y_min = math.log10(max(y_min, 1e-3))
    log_y_max = math.log10(y_max)

    margin_l, margin_r, margin_t, margin_b = 60, 20, 30, 40

    def x_of(x: float) -> float:
        return margin_l + (x - x_min) / (x_max - x_min) * (width - margin_l - margin_r)

    def y_of(y: float) -> float:
        ly = math.log10(max(y, 1e-3))
        return margin_t + (1 - (ly - log_y_min) / max(log_y_max - log_y_min, 1e-9)) * (height - margin_t - margin_b)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'style="background:#fafafa; font-family:monospace; font-size:11px">',
        '<style>'
        '.discard{fill:#bbb}.kept{fill:#0070f3;stroke:#003e8a;stroke-width:1}'
        '.crash{fill:#e00}.champion-line{stroke:#0070f3;stroke-width:2;fill:none}'
        '.axis{stroke:#666;stroke-width:1}.tick{stroke:#999;stroke-width:0.5}'
        '.tick-label{fill:#444}.title{fill:#222;font-size:12px;font-weight:bold}'
        '</style>',
    ]

    # axes
    parts.append(
        f'<line class="axis" x1="{margin_l}" y1="{margin_t}" '
        f'x2="{margin_l}" y2="{height - margin_b}" />'
    )
    parts.append(
        f'<line class="axis" x1="{margin_l}" y1="{height - margin_b}" '
        f'x2="{width - margin_r}" y2="{height - margin_b}" />'
    )

    # y axis ticks (log)
    decade_lo = math.floor(log_y_min)
    decade_hi = math.ceil(log_y_max)
    for decade in range(int(decade_lo), int(decade_hi) + 1):
        ms_val = 10 ** decade
        if ms_val < y_min * 0.5 or ms_val > y_max * 2:
            continue
        ty = y_of(ms_val)
        if margin_t <= ty <= height - margin_b:
            parts.append(f'<line class="tick" x1="{margin_l - 4}" y1="{ty}" x2="{width - margin_r}" y2="{ty}" />')
            label = f"{ms_val:g} ms"
            parts.append(f'<text class="tick-label" x="{margin_l - 8}" y="{ty + 4}" text-anchor="end">{label}</text>')

    # x axis ticks
    n_ticks = min(10, max(2, len(rows)))
    step = max(1, len(rows) // n_ticks)
    for i in range(0, len(rows) + 1, step):
        tx = x_of(i + 1)
        parts.append(f'<line class="tick" x1="{tx}" y1="{height - margin_b}" x2="{tx}" y2="{height - margin_b + 4}" />')
        parts.append(f'<text class="tick-label" x="{tx}" y="{height - margin_b + 16}" text-anchor="middle">{i + 1}</text>')

    # discarded + crashed points
    for x, y, kind, _ in pts:
        if kind == "discard":
            parts.append(f'<circle class="discard" cx="{x_of(x)}" cy="{y_of(y)}" r="3" />')
        elif kind == "crash":
            cx, cy = x_of(x), y_of(y) - 5  # crash plotted at top
            parts.append(f'<text class="crash" x="{cx}" y="{cy}" text-anchor="middle">x</text>')

    # champion progression line (kept points only, in order)
    kept_pts = [(x_of(p[0]), y_of(p[1])) for p in pts if p[2] == "kept"]
    if len(kept_pts) >= 2:
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in kept_pts)
        parts.append(f'<path class="champion-line" d="{path}" />')

    # kept points as stars
    for x, y, kind, dec in pts:
        if kind == "kept":
            cx, cy = x_of(x), y_of(y)
            parts.append(f'<circle class="kept" cx="{cx}" cy="{cy}" r="5" />')

    # title
    parts.append(
        f'<text class="title" x="{width // 2}" y="20" text-anchor="middle">'
        f'solve_ms vs proposal index — kept (blue stars) traces the champion line</text>'
    )

    parts.append('</svg>')
    return "\n".join(parts)


def _ascii_chain(summary: dict, *, max_chars: int = 120) -> str:
    parts: list[str] = []
    for r in summary["kept_rows"]:
        s = r.get("score") or {}
        ms = s.get("solve_ms_median")
        ms_str = f"{ms:.2f}ms" if isinstance(ms, (int, float)) else "??"
        iters = s.get("iters", "?")
        rid = (r.get("id") or "")[:8]
        gen = r.get("generation", "?")
        rat = (r.get("edit") or {}).get("rationale", "")[:max_chars].replace("\n", " ")
        parts.append(f"  gen {gen:>3}  {ms_str:>10}  iters={iters:<5}  {rid}  {escape(rat)}")
    return "\n".join(parts) if parts else "  (no kept proposals yet)"


def _reason_bars(summary: dict, *, width: int = 600) -> str:
    counts = summary["reasons"].most_common()
    if not counts:
        return ""
    max_count = max(c for _, c in counts)
    lines = []
    for reason, count in counts:
        bar_w = int((count / max_count) * 200)
        color = (
            "#0070f3" if "keep" in reason
            else "#e00" if "crash" in reason
            else "#999"
        )
        bar = (
            f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0">'
            f'<div style="width:280px;font-size:11px">{escape(reason)}</div>'
            f'<div style="background:{color};height:14px;width:{bar_w}px"></div>'
            f'<div style="font-size:11px">{count}</div>'
            f'</div>'
        )
        lines.append(bar)
    return "\n".join(lines)


def _slot_card(summary: dict) -> str:
    if not summary:
        return ""
    slot_label = _slot_label(summary["slot"])
    best = summary.get("best")
    best_ms = summary.get("best_ms")
    speedup = summary.get("speedup")
    best_section = ""
    if best is not None:
        s = best.get("score") or {}
        rat = (best.get("edit") or {}).get("rationale", "")[:400].replace("\n", " ")
        best_section = (
            f'<div style="margin-top:8px;padding:8px;background:#eef6ff;border-left:3px solid #0070f3">'
            f'<b>champion</b>: gen {best.get("generation")} '
            f'<code>{best.get("id", "")[:8]}</code>'
            f' &mdash; {best_ms:.2f} ms / iters={s.get("iters")} / fitness={s.get("fitness_final"):.2e}'
            f'<br/><span style="color:#444;font-size:11px">{escape(rat)}</span>'
            f'</div>'
        )
    speedup_str = f' &mdash; <b>{speedup:.2f}x</b> over first accept' if speedup else ''
    return (
        f'<div style="border:1px solid #ddd;border-radius:6px;padding:14px;margin-bottom:18px;background:white">'
        f'<h2 style="margin:0 0 6px 0;font-size:14px;font-family:monospace">{escape(slot_label)}</h2>'
        f'<div style="font-size:12px;color:#444">'
        f'total <b>{summary["total"]}</b>'
        f' &mdash; kept <b style="color:#0070f3">{summary["kept"]}</b>'
        f' / discarded <b style="color:#666">{summary["discarded"]}</b>'
        f' / crashed <b style="color:#e00">{summary["crashed"]}</b>'
        f'{speedup_str}'
        f'</div>'
        f'{best_section}'
        f'<div style="margin-top:10px">'
        f'{_svg_progression(summary)}'
        f'</div>'
        f'<details style="margin-top:8px"><summary style="cursor:pointer;font-size:11px;color:#444">'
        f'champion lineage chain ({len(summary["kept_rows"])} kept)</summary>'
        f'<pre style="background:#f6f8fa;padding:10px;border-radius:4px;overflow-x:auto;font-size:11px">'
        f'{escape(_ascii_chain(summary))}'
        f'</pre></details>'
        f'<details style="margin-top:8px"><summary style="cursor:pointer;font-size:11px;color:#444">'
        f'discard reason histogram</summary>'
        f'<div style="margin-top:6px">{_reason_bars(summary)}</div>'
        f'</details>'
        f'</div>'
    )


def _render_html(summaries: list[dict], *, refresh_sec: int) -> str:
    refresh_meta = (
        f'<meta http-equiv="refresh" content="{refresh_sec}">' if refresh_sec > 0 else ""
    )
    cards = "\n".join(_slot_card(s) for s in summaries if s)
    refresh_note = f'<small>auto-refresh: {refresh_sec}s</small>' if refresh_sec > 0 else ""

    total_props = sum(s.get("total", 0) for s in summaries)
    total_kept = sum(s.get("kept", 0) for s in summaries)
    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>convexkernels autoresearch dashboard</title>
{refresh_meta}
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; padding: 24px; max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 18px; margin: 0 0 4px 0; }}
.summary {{ font-size: 12px; color: #555; margin-bottom: 20px; }}
</style>
</head><body>
<h1>convexkernels autoresearch dashboard</h1>
<div class="summary">
{len(summaries)} slot(s) &middot; {total_props} proposals total &middot;
{total_kept} kept ({total_kept / max(total_props, 1) * 100:.1f}%)
&middot; {refresh_note}
</div>
{cards}
</body></html>"""


def _generate_once(paths: list[str], out: Path, refresh_sec: int) -> int:
    summaries = []
    for path_arg in paths:
        path = Path(path_arg)
        if path.is_dir():
            path = path / "lineage.jsonl"
        rows = _load(path)
        s = _summarize(rows)
        if s:
            s["path"] = str(path)
        summaries.append(s)

    html = _render_html(summaries, refresh_sec=refresh_sec)
    out.write_text(html)
    return sum(s.get("total", 0) for s in summaries)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="lineage.jsonl files or dirs")
    p.add_argument("--out", default="lineage_dashboard.html")
    p.add_argument("--refresh", type=int, default=0,
                   help="auto-refresh meta tag interval in seconds (0 = no refresh)")
    p.add_argument("--watch", type=int, default=0,
                   help="re-render every N seconds (0 = render once and exit)")
    args = p.parse_args(argv[1:])

    out = Path(args.out)
    if args.watch <= 0:
        n = _generate_once(args.paths, out, args.refresh)
        print(f"wrote {out} ({out.stat().st_size} bytes, {n} proposals)")
        return 0

    import time
    print(f"watching {len(args.paths)} lineage path(s); regenerating {out} every {args.watch}s")
    print("ctrl-c to stop")
    try:
        while True:
            n = _generate_once(args.paths, out, args.refresh)
            print(f"  [{time.strftime('%H:%M:%S')}] {n} proposals")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
