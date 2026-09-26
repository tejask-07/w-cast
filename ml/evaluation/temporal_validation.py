"""Expanding-window out-of-sample evaluation for paired historical forecasts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.evaluate_paired_dataset import (
    DEFAULT_DATASET,
    INTERNAL_VARIABLES,
    _improvement,
    _metric_summary,
    load_paired_jsonl,
)
from ml.evaluation.historical_skill import (
    MIN_LOCAL_SAMPLES,
    aggregate_hierarchical_skill,
    select_hierarchical_skill,
)

DEFAULT_OUTPUT = Path("data/processed/historical_paired_oos_evaluation.jsonl")
DEFAULT_REPORT = Path("data/processed/historical_paired_oos_evaluation.report.json")
DEFAULT_EXISTING_MAP = Path("data/processed/india_weight_map_validation.json")
MODEL_NAMES = ("gfs", "gefs", "equal_weight", "wcast")
BASELINES = ("gfs", "gefs", "equal_weight")


def _date(value: str) -> str:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()


def _finite(record: Mapping[str, Any], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {key} in paired dataset")
    return value


def _metric_bundle(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [float(row["observation_value"]) for row in rows]
    result: dict[str, Any] = {}
    for name in MODEL_NAMES:
        forecasts = [float(row[f"{name}_value"]) for row in rows]
        result[name] = _metric_summary(observed, forecasts)
    result["win_rates"] = {}
    for baseline in BASELINES:
        wins = sum(
            abs(float(row["wcast_value"]) - float(row["observation_value"]))
            < abs(float(row[f"{baseline}_value"]) - float(row["observation_value"]))
            for row in rows
        )
        result["win_rates"][baseline] = {
            "wins": wins,
            "cases": len(rows),
            "percent": 100.0 * wins / len(rows) if rows else None,
        }
    result["improvement_percent"] = {
        baseline: _improvement(result[baseline]["mae"], result["wcast"]["mae"])
        for baseline in BASELINES
    }
    return result


def _group_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: _metric_bundle(value) for key, value in sorted(groups.items())}


def _unique_weights(rows: list[Mapping[str, Any]]) -> int:
    return len({
        (round(float(row["gfs_weight"]), 12), round(float(row["gefs_weight"]), 12))
        for row in rows
    })


def evaluate_expanding_window(
    records: list[Mapping[str, Any]],
    minimum_training_dates: int = MIN_LOCAL_SAMPLES,
    existing_map_path: str | Path | None = DEFAULT_EXISTING_MAP,
) -> dict[str, Any]:
    """Fit weights on dates before each test date and evaluate only that date."""
    if minimum_training_dates < MIN_LOCAL_SAMPLES:
        raise ValueError(
            f"minimum_training_dates cannot be below existing hierarchy threshold {MIN_LOCAL_SAMPLES}"
        )
    if not records:
        raise ValueError("Paired dataset contains no records")

    dates = sorted({_date(str(row["forecast_date"])) for row in records})
    by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_date[_date(str(record["forecast_date"]))].append(record)

    evaluated: list[dict[str, Any]] = []
    skipped_dates: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    resolution_counts: Counter[str] = Counter()

    for test_date in dates:
        train_dates = [day for day in dates if day < test_date]
        if len(train_dates) < minimum_training_dates:
            skipped_dates.append({
                "test_date": test_date,
                "training_dates": len(train_dates),
                "required_training_dates": minimum_training_dates,
                "reason": "insufficient_prior_dates",
            })
            continue

        training_records = [record for day in train_dates for record in by_date[day]]
        if any(_date(str(record["forecast_date"])) >= test_date for record in training_records):
            raise AssertionError("Temporal leakage: training set contains test/future date")
        aggregates = aggregate_hierarchical_skill([
            {
                "city": row["location_name"],
                "region": row["region"],
                "variable": INTERNAL_VARIABLES[row["variable"]],
                "lead_hours": int(row["lead_hours"]),
                "observation": _finite(row, "observation_value"),
                "gfs": _finite(row, "gfs_value"),
                "gefs": _finite(row, "gefs_value"),
            }
            for row in training_records
        ])
        train_date_min = min(train_dates)
        train_date_max = max(train_dates)

        for test_record in by_date[test_date]:
            location = str(test_record["location_name"])
            variable = str(test_record["variable"])
            lead = int(test_record["lead_hours"])
            skill = select_hierarchical_skill(
                aggregates,
                location,
                INTERNAL_VARIABLES[variable],
                lead,
                minimum_samples=MIN_LOCAL_SAMPLES,
            )
            gfs_samples = int(skill["sample_count"]["gfs"])
            gefs_samples = int(skill["sample_count"]["gefs"])
            if min(gfs_samples, gefs_samples) == 0:
                skipped_cases.append({
                    "test_date": test_date,
                    "location_name": location,
                    "variable": variable,
                    "lead_hours": lead,
                    "reason": "insufficient_training_skill_samples",
                    "training_sample_count": {"gfs": gfs_samples, "gefs": gefs_samples},
                })
                continue

            weights = adaptive_weights({
                "gfs": skill["gfs"]["mae"],
                "gefs": skill["gefs"]["mae"],
            })
            gfs_weight = float(weights["gfs"])
            gefs_weight = float(weights["gefs"])
            if not (0 <= gfs_weight <= 1 and 0 <= gefs_weight <= 1):
                raise ValueError("Historical weights must be in [0, 1]")
            if not math.isclose(gfs_weight + gefs_weight, 1.0, abs_tol=1e-12):
                raise ValueError("Historical weights must sum to one")

            gfs_value = _finite(test_record, "gfs_value")
            gefs_value = _finite(test_record, "gefs_value")
            observation = _finite(test_record, "observation_value")
            equal_value = (gfs_value + gefs_value) / 2.0
            wcast_value = gfs_weight * gfs_value + gefs_weight * gefs_value
            evaluated.append({
                "test_date": test_date,
                "training_date_start": train_date_min,
                "training_date_end": train_date_max,
                "training_date_count": len(train_dates),
                "location_name": location,
                "latitude": float(test_record["latitude"]),
                "longitude": float(test_record["longitude"]),
                "region": str(test_record["region"]),
                "variable": variable,
                "lead_hours": lead,
                "forecast_cycle": test_record["forecast_cycle"],
                "valid_time": test_record["valid_time"],
                "gefs_source": test_record.get("gefs_source"),
                "gfs_source": test_record.get("gfs_source"),
                "skill_source": skill["source"],
                "training_sample_count": {"gfs": gfs_samples, "gefs": gefs_samples},
                "training_gfs_mae": skill["gfs"]["mae"],
                "training_gefs_mae": skill["gefs"]["mae"],
                "gfs_weight": gfs_weight,
                "gefs_weight": gefs_weight,
                "gfs_value": gfs_value,
                "gefs_value": gefs_value,
                "equal_weight_value": equal_value,
                "wcast_value": wcast_value,
                "observation_value": observation,
            })
            resolution_counts[skill["source"]] += 1

    existing_map_summary = None
    if existing_map_path is not None and Path(existing_map_path).is_file():
        from ml.spatial.weight_map import summarize_weight_map

        weight_map_payload = json.loads(Path(existing_map_path).read_text(encoding="utf-8"))
        weight_map = (
            weight_map_payload.get("weight_map", [])
            if isinstance(weight_map_payload, dict)
            else weight_map_payload
        )
        existing_map_summary = summarize_weight_map(weight_map)

    grouped_keys = sorted({
        (row["location_name"], row["variable"], row["lead_hours"])
        for row in evaluated
    }, key=lambda value: tuple(map(str, value)))
    provenance = {
        "gfs_sources": dict(Counter(str(row.get("gfs_source")) for row in records)),
        "gefs_sources": dict(Counter(str(row.get("gefs_source")) for row in records)),
        "observation_source": "Open-Meteo Archive API",
        "gfs_resolution_degrees": sorted({row.get("gfs_resolution_degrees") for row in records}),
        "gefs_resolution_degrees": sorted({row.get("gefs_resolution_degrees") for row in records}),
        "record_date_start": min(dates),
        "record_date_end": max(dates),
        "training_period": {
            "start": min((row["training_date_start"] for row in evaluated), default=None),
            "end": max((row["training_date_end"] for row in evaluated), default=None),
        },
        "test_period": {
            "start": min((row["test_date"] for row in evaluated), default=None),
            "end": max((row["test_date"] for row in evaluated), default=None),
        },
        "test_dates_used": sorted({row["test_date"] for row in evaluated}),
        "record_count": len(records),
        "rejected_record_count": 0,
    }
    report = {
        "evaluation": "expanding_window_temporal_out_of_sample",
        "minimum_training_dates": minimum_training_dates,
        "hierarchy_sample_threshold": MIN_LOCAL_SAMPLES,
        "dataset_record_count": len(records),
        "forecast_dates": dates,
        "training_test_policy": "For each test date, training contains strictly earlier forecast_date values only.",
        "skipped_dates": skipped_dates,
        "skipped_cases": skipped_cases,
        "evaluated_cases": len(evaluated),
        "skill_resolution_counts": dict(resolution_counts),
        "unique_weight_combinations": _unique_weights(evaluated),
        "existing_1820_cell_validation_map": existing_map_summary,
        "metrics": {
            "overall": _metric_bundle(evaluated),
            "by_variable": _group_summaries(evaluated, "variable"),
            "by_lead_hours": _group_summaries(evaluated, "lead_hours"),
            "by_location": _group_summaries(evaluated, "location_name"),
            "by_location_variable_lead": {
                "|".join(map(str, key)): _metric_bundle([
                    row for row in evaluated
                    if (row["location_name"], row["variable"], row["lead_hours"]) == key
                ])
                for key in grouped_keys
            },
        },
        "provenance": provenance,
        "evaluation_note": "Expanding-window out-of-sample. Training dates are strictly prior to each test date; target-date outcomes never fit their own weights.",
    }
    return {"records": evaluated, "report": report}


def run_temporal_validation(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_path: str | Path = DEFAULT_OUTPUT,
    report_path: str | Path = DEFAULT_REPORT,
    existing_map_path: str | Path | None = DEFAULT_EXISTING_MAP,
    minimum_training_dates: int = MIN_LOCAL_SAMPLES,
) -> dict[str, Any]:
    paired = load_paired_jsonl(dataset_path)
    result = evaluate_expanding_window(
        paired,
        minimum_training_dates=minimum_training_dates,
        existing_map_path=existing_map_path,
    )
    output = Path(output_path)
    report_file = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in result["records"]:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    report_file.write_text(json.dumps(result["report"], indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run temporal out-of-sample W-CAST evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minimum-training-dates", type=int, default=MIN_LOCAL_SAMPLES)
    args = parser.parse_args()
    result = run_temporal_validation(
        dataset_path=args.dataset,
        output_path=args.output,
        report_path=args.report,
        minimum_training_dates=args.minimum_training_dates,
    )
    print(json.dumps({
        "evaluated_cases": result["report"]["evaluated_cases"],
        "skipped_dates": result["report"]["skipped_dates"],
        "skipped_cases": len(result["report"]["skipped_cases"]),
        "overall": result["report"]["metrics"]["overall"],
        "skill_resolution_counts": result["report"]["skill_resolution_counts"],
        "unique_weight_combinations": result["report"]["unique_weight_combinations"],
        "output": str(args.output),
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
