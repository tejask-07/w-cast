"""Out-of-sample GFS/GEFS/W-CAST benchmark utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.metrics import bias, mae, rmse

MODELS = ("gfs", "gefs", "equal_blend", "wcast")


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    if not actual:
        return {"mae": None, "rmse": None, "bias": None, "sample_count": 0}
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "bias": bias(actual, predicted),
        "sample_count": len(actual),
    }


def benchmark_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Benchmark paired records with chronological leave-one-out weights."""
    paired = [
        record for record in records
        if record.get("gfs") is not None
        and record.get("gefs") is not None
        and record.get("observation") is not None
    ]
    paired.sort(key=lambda record: str(record.get("valid_time", "")))
    values: dict[tuple[str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, record in enumerate(paired):
        variable = str(record["variable"])
        lead = int(record["lead_hours"])
        gfs = float(record["gfs"])
        gefs = float(record["gefs"])
        observation = float(record["observation"])
        training = [
            candidate for candidate in paired[:index]
            if candidate["variable"] == variable
            and int(candidate["lead_hours"]) == lead
        ]
        errors = {
            "gfs": [abs(float(candidate["gfs"]) - float(candidate["observation"])) for candidate in training],
            "gefs": [abs(float(candidate["gefs"]) - float(candidate["observation"])) for candidate in training],
        }
        if len(errors["gfs"]) >= 3 and len(errors["gefs"]) >= 3:
            weights = adaptive_weights({
                "gfs": float(np.mean(errors["gfs"])),
                "gefs": float(np.mean(errors["gefs"])),
            })
        else:
            weights = {"gfs": 0.5, "gefs": 0.5}
        predictions = {
            "gfs": gfs,
            "gefs": gefs,
            "equal_blend": 0.5 * gfs + 0.5 * gefs,
            "wcast": weights["gfs"] * gfs + weights["gefs"] * gefs,
        }
        key = (variable, lead, "all")
        for model, prediction in predictions.items():
            values[key][model].append(prediction)
        values[key]["actual"].append(observation)

    result: dict[str, Any] = {
        "paired_record_count": len(paired),
        "method": "chronological leave-one-out inverse-MAE weights; minimum training samples=3",
        "groups": {},
    }
    for (variable, lead, scope), group in sorted(values.items()):
        metrics = {
            model: _metrics(group["actual"], group[model])
            for model in MODELS
        }
        best_individual = min(
            (metrics[model]["mae"], model) for model in ("gfs", "gefs")
        )
        wcast_mae = metrics["wcast"]["mae"]
        metrics["wcast"]["improvement_vs_gfs"] = (
            None if wcast_mae is None else metrics["gfs"]["mae"] - wcast_mae
        )
        metrics["wcast"]["improvement_vs_gefs"] = (
            None if wcast_mae is None else metrics["gefs"]["mae"] - wcast_mae
        )
        metrics["wcast"]["improvement_vs_best_individual"] = (
            None if wcast_mae is None else best_individual[0] - wcast_mae
        )
        result["groups"][f"{variable}:{lead}:{scope}"] = metrics
    return result


def benchmark_file(
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = benchmark_records(payload["records"])
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
