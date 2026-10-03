# ---------------------------------------------------------------------------
# Import libraries
# ---------------------------------------------------------------------------
import json
import time
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging Agent Actions
# ---------------------------------------------------------------------------
class TraceLogger:
    """Records every agent action across a run for later inspection."""

    def __init__(self):
        self.entries = []

    def log(self, run_id: str, agent: str, action: str, detail: str):
        self.entries.append({
            "run_id": run_id,
            "timestamp": time.strftime("%H:%M:%S"),
            "agent": agent,
            "action": action,
            "detail": (detail[:160] + "...") if len(detail) > 160 else detail,
        })

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)

    def print_trace(self, run_id: str | None = None):
        rows = self.entries if run_id is None else [e for e in self.entries if e["run_id"] == run_id]
        for r in rows:
            print(f"[{r['timestamp']}] {r['agent']:<22} {r['action']:<16} {r['detail']}")

    # output to file - in JSONL format
    def save_to_file(self, path: str, run_id: str | None = None):
        """Appends trace entries to a JSONL file, one JSON object per line."""
        rows = self.entries if run_id is None else [e for e in self.entries if e["run_id"] == run_id]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------------------
# Single shared instance, imported wherever logging happens.
# ---------------------------------------------------------------------------
trace_logger = TraceLogger()

