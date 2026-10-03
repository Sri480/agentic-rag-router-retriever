import json
import html
from pathlib import Path

# ---------------------------------------------------------------------------
# This module is to convert the outputs to HTML for easier viewing
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


ROUTE_COLORS = {"RAG": "#2563eb", "WEB": "#16a34a", "BOTH": "#9333ea"}


def _route_badge(route: str) -> str:
    color = ROUTE_COLORS.get(route, "#6b7280")
    return f'<span class="badge" style="background:{color}">{html.escape(route)}</span>'


def _esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def build_html(results_path: str, trace_path: str, out_path: str):
    results = load_jsonl(results_path) if Path(results_path).exists() else []
    trace = load_jsonl(trace_path) if Path(trace_path).exists() else []

    trace_by_run = {}
    for row in trace:
        trace_by_run.setdefault(row["run_id"], []).append(row)

    cards = []
    for r in results:
        run_trace = trace_by_run.get(r["run_id"], [])
        trace_rows = "".join(
            f"<tr><td class='mono'>{html.escape(t['timestamp'])}</td>"
            f"<td>{html.escape(t['agent'])}</td>"
            f"<td>{html.escape(t['action'])}</td>"
            f"<td>{_esc(t['detail'])}</td></tr>"
            for t in run_trace
        )
        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <h2>{html.escape(r['query'])}</h2>
                {_route_badge(r['route'])}
            </div>
            <p class="reasoning"><strong>Why:</strong> {html.escape(r['reasoning'])}</p>
            <div class="answer">{_esc(r['answer'])}</div>
            <details>
                <summary>Trace ({len(run_trace)} steps)</summary>
                <table>
                    <thead><tr><th>Time</th><th>Agent</th><th>Action</th><th>Detail</th></tr></thead>
                    <tbody>{trace_rows}</tbody>
                </table>
            </details>
            <p class="timestamp">Saved: {html.escape(r.get('saved_at', ''))}</p>
        </div>
        """)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agentic RAG Results</title>
<style>
    :root {{
        --bg: #f8fafc; --card-bg: #ffffff; --text: #0f172a;
        --muted: #64748b; --border: #e2e8f0;
    }}
    @media (prefers-color-scheme: dark) {{
        :root {{ --bg: #0f172a; --card-bg: #1e293b; --text: #f1f5f9;
                 --muted: #94a3b8; --border: #334155; }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--bg); color: var(--text); margin: 0;
        padding: 32px 16px;
    }}
    .container {{ max-width: 800px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 0.9rem; }}
    .card {{
        background: var(--card-bg); border: 1px solid var(--border);
        border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;
    }}
    .card-header {{
        display: flex; justify-content: space-between; align-items: flex-start;
        gap: 12px; margin-bottom: 8px;
    }}
    .card-header h2 {{ font-size: 1.05rem; margin: 0; font-weight: 600; flex: 1; }}
    .badge {{
        color: white; font-size: 0.75rem; font-weight: 600;
        padding: 3px 10px; border-radius: 999px; white-space: nowrap;
    }}
    .reasoning {{ color: var(--muted); font-size: 0.88rem; margin: 8px 0; }}
    .answer {{
        background: var(--bg); border-radius: 8px; padding: 14px 16px;
        margin: 12px 0; line-height: 1.5; font-size: 0.95rem;
    }}
    details {{ margin-top: 8px; font-size: 0.85rem; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.82rem; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
    .mono {{ font-family: ui-monospace, monospace; white-space: nowrap; }}
    .timestamp {{ color: var(--muted); font-size: 0.78rem; margin-top: 12px; margin-bottom: 0; }}
</style>
</head>
<body>
<div class="container">
    <h1>Agentic RAG: Results</h1>
    <div class="subtitle">{len(results)} question(s) answered</div>
    {''.join(cards) if cards else '<p>No results found.</p>'}
</div>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(doc, encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    build_html(
        results_path="outputs/results.jsonl",
        trace_path="outputs/trace_log.jsonl",
        out_path="outputs/report.html",
    )

