"""Rebuild the India weight map from existing historical records."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.spatial.weight_map import build_india_weight_map, summarize_weight_map


if __name__ == "__main__":
    history_path = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"
    output_path = REPO_ROOT / "data" / "processed" / "india_weight_map_7d.json"
    records = json.loads(history_path.read_text(encoding="utf-8"))["records"]
    weight_map = build_india_weight_map(records)
    output_path.write_text(json.dumps(weight_map, indent=2), encoding="utf-8")
    print(json.dumps(summarize_weight_map(weight_map), sort_keys=True))
