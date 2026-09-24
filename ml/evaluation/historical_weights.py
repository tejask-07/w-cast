import json

from ml.blending.adaptive_weights import adaptive_weights


DEFAULT_HISTORY = "data/processed/history_7d_multilead.json"
MIN_LOCAL_SAMPLES = 3


def load_history(path=DEFAULT_HISTORY):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["records"]


def _mae(values):
    return sum(values) / len(values) if values else None


def build_historical_weights(path=DEFAULT_HISTORY):
    records = load_history(path)

    local = {}
    broader = {}

    for r in records:
        gfs_error = r.get("gfs_absolute_error")
        gefs_error = r.get("gefs_absolute_error")

        key = (
            r["city"],
            r["variable"],
            r["lead_hours"],
        )

        broad_key = (
            r["variable"],
            r["lead_hours"],
        )

        local.setdefault(key, {"gfs": [], "gefs": []})
        broader.setdefault(broad_key, {"gfs": [], "gefs": []})

        if gfs_error is not None:
            local[key]["gfs"].append(gfs_error)
            broader[broad_key]["gfs"].append(gfs_error)

        if gefs_error is not None:
            local[key]["gefs"].append(gefs_error)
            broader[broad_key]["gefs"].append(gefs_error)

    result = {}

    for (city, variable, lead_hours), errors in local.items():
        gfs_errors = errors["gfs"]
        gefs_errors = errors["gefs"]

        if (
            len(gfs_errors) >= MIN_LOCAL_SAMPLES
            and len(gefs_errors) >= MIN_LOCAL_SAMPLES
        ):
            skill = errors
            source = "local"
        else:
            skill = broader[(variable, lead_hours)]
            source = "broader"

        gfs_mae = _mae(skill["gfs"])
        gefs_mae = _mae(skill["gefs"])

        if gfs_mae is None and gefs_mae is None:
            continue

        errors_for_weights = {
            "gfs": gfs_mae,
            "gefs": gefs_mae,
        }

        weights = adaptive_weights(errors_for_weights)

        result.setdefault(city, {})
        result[city].setdefault(variable, {})
        result[city][variable][str(lead_hours)] = {
            "weights": weights,
            "gfs_mae": gfs_mae,
            "gefs_mae": gefs_mae,
            "gfs_sample_count": len(skill["gfs"]),
            "gefs_sample_count": len(skill["gefs"]),
            "source": source,
        }

    return result


if __name__ == "__main__":
    weights = build_historical_weights()
    print(json.dumps(weights, indent=2))