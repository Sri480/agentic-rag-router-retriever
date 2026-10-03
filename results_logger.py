import json
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# helper function to save the results to a file on JSONL format
# ---------------------------------------------------------------------------
def save_result(result: dict, path: str = "outputs/results.jsonl"):
    """Appends one query result to a JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    entry = {**result, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
