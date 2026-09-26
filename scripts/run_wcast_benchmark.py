"""Generate the reproducible W-CAST benchmark artifact."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.evaluation.benchmark import benchmark_file


if __name__ == "__main__":
    result = benchmark_file(
        REPO_ROOT / "data" / "processed" / "history_7d_multilead.json",
        REPO_ROOT / "data" / "processed" / "wcast_benchmark.json",
    )
    print(
        f"paired_records={result['paired_record_count']} "
        f"groups={len(result['groups'])}"
    )
