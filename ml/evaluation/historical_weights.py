import json
from pathlib import Path

from ml.blending.adaptive_weights import adaptive_weights


def load_history(path="data/processed/history_2d.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def build_historical_weights(path="data/processed/history_2d.json"):
    records = load_history(path)

    groups = {}

    for r in records:
        if r["gfs_absolute_error"] is None or r["gefs_absolute_error"] is None:
            continue

        key = (
            r["city"],
            r["variable"],
            r["lead_hours"],
        )

        groups.setdefault(key, {"gfs": [], "gefs": []})

        groups[key]["gfs"].append(r["gfs_absolute_error"])
        groups[key]["gefs"].append(r["gefs_absolute_error"])

    result = {}

    for (city, variable, lead_hours), errors in groups.items():
        gfs_mae = sum(errors["gfs"]) / len(errors["gfs"])
        gefs_mae = sum(errors["gefs"]) / len(errors["gefs"])

        weights = adaptive_weights({
            "gfs": gfs_mae,
            "gefs": gefs_mae,
        })

        result.setdefault(city, {})
        result[city].setdefault(variable, {})
        result[city][variable][str(lead_hours)] = {
            "weights": weights,
            "gfs_mae": gfs_mae,
            "gefs_mae": gefs_mae,
            "sample_count": len(errors["gfs"]),
        }

    return result


if __name__ == "__main__":
    weights = build_historical_weights()

    print(json.dumps(weights, indent=2))