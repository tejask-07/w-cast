"""Offline skill, weight, baseline, and W-CAST evaluation for paired JSONL data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.historical_skill import (
    MIN_LOCAL_SAMPLES,
    aggregate_hierarchical_skill,
    select_hierarchical_skill,
)
from ml.evaluation.metrics import bias, mae, rmse
from ml.spatial.weight_map import build_india_weight_map, summarize_weight_map

DEFAULT_DATASET = Path("data/processed/historical_paired_validation.jsonl")
DEFAULT_WEIGHT_MAP = Path("data/processed/india_weight_map_validation.json")
DEFAULT_REPORT = Path("data/processed/historical_paired_evaluation.report.json")
EXTERNAL_VARIABLES = ("temperature", "rainfall", "wind_speed")
INTERNAL_VARIABLES = {
    "temperature": "temperature",
    "rainfall": "precipitation",
    "wind_speed": "wind_speed",
}
LEADS = (24, 48, 72)


def load_paired_jsonl(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            records.append(record)
    return records


def _finite(record: Mapping[str, Any], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Record has invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Record has non-finite {field}")
    return value


def _normalize_for_hierarchy(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        variable = str(record["variable"])
        if variable not in INTERNAL_VARIABLES:
            raise ValueError(f"Unsupported dataset variable: {variable}")
        normalized.append({
            "city": str(record["location_name"]),
            "region": str(record["region"]),
            "variable": INTERNAL_VARIABLES[variable],
            "lead_hours": int(record["lead_hours"]),
            "observation": _finite(record, "observation_value"),
            "gfs": _finite(record, "gfs_value"),
            "gefs": _finite(record, "gefs_value"),
            "precipitation_mm": _finite(record, "observation_value") if variable == "rainfall" else None,
        })
    return normalized


def _metric_summary(actual: list[float], forecast: list[float]) -> dict[str, Any]:
    if not actual:
        return {"mae": None, "rmse": None, "bias": None, "sample_count": 0}
    return {
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "bias": bias(actual, forecast),
        "sample_count": len(actual),
    }


def _improvement(baseline_mae: float | None, wcast_mae: float | None) -> float | None:
    if baseline_mae is None or wcast_mae is None or baseline_mae == 0:
        return None
    return 100.0 * (baseline_mae - wcast_mae) / baseline_mae


def _group_metrics(
    rows: list[dict[str, Any]],
    weight_lookup: Mapping[tuple[str, str, int], Mapping[str, float]],
) -> dict[str, Any]:
    actual = [row["observation_value"] for row in rows]
    gfs = [row["gfs_value"] for row in rows]
    gefs = [row["gefs_value"] for row in rows]
    equal = [(gfs_value + gefs_value) / 2.0 for gfs_value, gefs_value in zip(gfs, gefs)]
    wcast = []
    for row in rows:
        weights = weight_lookup[(row["location_name"], row["variable"], row["lead_hours"])]
        wcast.append(weights["gfs"] * row["gfs_value"] + weights["gefs"] * row["gefs_value"])
    metrics = {
        "gfs": _metric_summary(actual, gfs),
        "gefs": _metric_summary(actual, gefs),
        "equal_weight": _metric_summary(actual, equal),
        "wcast": _metric_summary(actual, wcast),
    }
    wcast_mae = metrics["wcast"]["mae"]
    metrics["improvement_vs_gfs_percent"] = _improvement(metrics["gfs"]["mae"], wcast_mae)
    metrics["improvement_vs_gefs_percent"] = _improvement(metrics["gefs"]["mae"], wcast_mae)
    metrics["improvement_vs_equal_percent"] = _improvement(metrics["equal_weight"]["mae"], wcast_mae)
    return metrics


def _weight_for_skill(skill: Mapping[str, Any]) -> dict[str, float]:
    weights = adaptive_weights({
        "gfs": skill["gfs"]["mae"],
        "gefs": skill["gefs"]["mae"],
    })
    gfs_weight = float(weights["gfs"])
    gefs_weight = float(weights["gefs"])
    if not (0 <= gfs_weight <= 1 and 0 <= gefs_weight <= 1):
        raise ValueError("Adaptive weights must be between zero and one")
    if not math.isclose(gfs_weight + gefs_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Adaptive weights must sum to one")
    return {"gfs": gfs_weight, "gefs": gefs_weight}


def _group_records(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    weight_lookup: Mapping[tuple[str, str, int], Mapping[str, float]],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = {}
    for group_key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        key = "|".join(str(value) for value in group_key) if len(group_key) > 1 else str(group_key[0])
        result[key] = _group_metrics(values, weight_lookup)
    return result


def evaluate_records(
    records: list[Mapping[str, Any]],
    minimum_samples: int = MIN_LOCAL_SAMPLES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an evaluation report and separately generated India validation map."""
    if not records:
        raise ValueError("Paired dataset contains no records")
    source_rows = [dict(record) for record in records]
    hierarchy_rows = _normalize_for_hierarchy(source_rows)
    aggregates = aggregate_hierarchical_skill(hierarchy_rows)

    locations = sorted({str(record["location_name"]) for record in source_rows})
    variables = sorted({str(record["variable"]) for record in source_rows})
    leads = sorted({int(record["lead_hours"]) for record in source_rows})
    weight_lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    weight_rows: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()

    for location in locations:
        for variable in variables:
            internal_variable = INTERNAL_VARIABLES[variable]
            for lead in leads:
                selected = select_hierarchical_skill(
                    aggregates,
                    location,
                    internal_variable,
                    lead,
                    minimum_samples=minimum_samples,
                )
                weights = _weight_for_skill(selected)
                gfs_count = int(selected["sample_count"]["gfs"])
                gefs_count = int(selected["sample_count"]["gefs"])
                sample_count = min(gfs_count, gefs_count)
                key = (location, variable, lead)
                weight_lookup[key] = weights
                resolution_counts[selected["source"]] += 1
                weight_rows.append({
                    "location_name": location,
                    "variable": variable,
                    "lead_hours": lead,
                    "skill_source": selected["source"],
                    "sample_count": sample_count,
                    "gfs_sample_count": gfs_count,
                    "gefs_sample_count": gefs_count,
                    "gfs_weight": weights["gfs"],
                    "gefs_weight": weights["gefs"],
                    "gfs_mae": selected["gfs"]["mae"],
                    "gefs_mae": selected["gefs"]["mae"],
                })

    grouped = {
        "overall": _group_metrics(source_rows, weight_lookup),
        "by_variable": _group_records(source_rows, ("variable",), weight_lookup),
        "by_lead_hours": _group_records(source_rows, ("lead_hours",), weight_lookup),
        "by_location": _group_records(source_rows, ("location_name",), weight_lookup),
        "by_location_variable_lead": _group_records(
            source_rows,
            ("location_name", "variable", "lead_hours"),
            weight_lookup,
        ),
    }

    validation_map = build_india_weight_map(
        hierarchy_rows,
        variables=tuple(INTERNAL_VARIABLES[variable] for variable in variables),
        leads=leads,
        minimum_samples=minimum_samples,
    )
    map_summary = summarize_weight_map(validation_map)
    rounded_pairs = {
        (round(row["gfs_weight"], 12), round(row["gefs_weight"], 12))
        for row in weight_rows
    }
    locations_with_different_weights = 0
    for location in locations:
        own = {
            (variable, lead): (round(weight_lookup[(location, variable, lead)]["gfs"], 12), round(weight_lookup[(location, variable, lead)]["gefs"], 12))
            for variable in variables
            for lead in leads
        }
        if any(
            own[(variable, lead)] != (
                round(weight_lookup[(other, variable, lead)]["gfs"], 12),
                round(weight_lookup[(other, variable, lead)]["gefs"], 12),
            )
            for other in locations if other != location
            for variable in variables
            for lead in leads
        ):
            locations_with_different_weights += 1

    report = {
        "dataset_records": len(source_rows),
        "date_range": {
            "start": min(str(row["forecast_date"]) for row in source_rows),
            "end": max(str(row["forecast_date"]) for row in source_rows),
        },
        "variables": variables,
        "lead_hours": leads,
        "weight_method": "inverse MAE: score=1/(MAE+1e-6), then normalize GFS/GEFS scores",
        "minimum_samples": minimum_samples,
        "skill_resolution_counts": dict(resolution_counts),
        "location_weight_resolutions": weight_rows,
        "number_unique_weight_combinations": len(rounded_pairs),
        "locations_with_differing_weights": locations_with_different_weights,
        "map_source_resolution_counts": map_summary,
        "metrics": grouped,
        "evaluation_note": "Weights are fit and scored on the same initial 10-date dataset (in-sample resubstitution), not a held-out benchmark.",
    }
    map_artifact = {
        "metadata": {
            "dataset": "data/processed/historical_paired_validation.jsonl",
            "dataset_records": len(source_rows),
            "date_range": report["date_range"],
            "variables": variables,
            "lead_hours": leads,
            "gefs_sources": sorted({str(row.get("gefs_source", "unknown")) for row in source_rows}),
            "gfs_sources": sorted({str(row.get("gfs_source", "unknown")) for row in source_rows}),
            "skill_source_counts": map_summary,
            "minimum_samples": minimum_samples,
            "weight_method": report["weight_method"],
            "map_grid_resolution_degrees": 0.5,
            "evaluation_only": True,
        },
        "weight_map": validation_map,
    }
    return report, map_artifact


def run_evaluation(
    dataset_path: str | Path = DEFAULT_DATASET,
    weight_map_path: str | Path = DEFAULT_WEIGHT_MAP,
    report_path: str | Path = DEFAULT_REPORT,
    minimum_samples: int = MIN_LOCAL_SAMPLES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = load_paired_jsonl(dataset_path)
    report, weight_map = evaluate_records(records, minimum_samples=minimum_samples)
    report["dataset_path"] = str(dataset_path)
    weight_map["metadata"]["dataset"] = str(dataset_path)
    Path(weight_map_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(weight_map_path).write_text(json.dumps(weight_map, indent=2, allow_nan=False), encoding="utf-8")
    Path(report_path).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report, weight_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired historical GFS/GEFS forecasts")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minimum-samples", type=int, default=MIN_LOCAL_SAMPLES)
    args = parser.parse_args()
    report, _ = run_evaluation(args.dataset, args.weight_map, args.report, args.minimum_samples)
    print(json.dumps({
        "overall": report["metrics"]["overall"],
        "skill_resolution_counts": report["skill_resolution_counts"],
        "number_unique_weight_combinations": report["number_unique_weight_combinations"],
        "locations_with_differing_weights": report["locations_with_differing_weights"],
        "weight_map": str(args.weight_map),
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()