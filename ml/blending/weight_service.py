"""Canonical historical-weight lookup shared by forecast consumers."""

import json
from pathlib import Path
from typing import Any

from ml.blending.adaptive_weights import adaptive_weights
from ml.evaluation.historical_skill import (
    aggregate_contextual_hierarchical_skill,
    select_contextual_hierarchical_skill,
)
from ml.spatial.location import resolve_location
from ml.spatial.lookup import get_spatial_weights


def _candidate_history_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return [
        repo_root / "data" / "processed" / "history_7d_multilead.json",
        repo_root / "data" / "processed" / "history_2d.json",
        repo_root / "data" / "processed" / "history_2d_20locations_multilead.json",
        repo_root / "data" / "processed" / "history_30d_20locations_multilead.json",
    ]


def _load_historical_context_records(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    def _coerce_records(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return records
        raise ValueError("Historical context records must be a JSON list or a dict containing a 'records' list")

    if path is not None:
        source = Path(path)
        if source.is_file():
            with source.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return _coerce_records(data)
        raise FileNotFoundError(f"Historical context file not found: {source}")

    for candidate in _candidate_history_paths():
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as file:
                data = json.load(file)
            try:
                return _coerce_records(data)
            except ValueError:
                continue
    return []


def _contextual_weight_lookup(
    latitude: float,
    longitude: float,
    variable: str,
    lead_hours: int,
    season: str | None,
    regime: str | None,
    historical_path: str | Path | None,
) -> dict[str, Any] | None:
    if season is None and regime is None:
        return None

    try:
        city = resolve_location(latitude, longitude)["city"]
    except ValueError:
        return None

    if historical_path is None:
        records = _load_historical_context_records()
    else:
        records = _load_historical_context_records(historical_path)
    if not records:
        return None

    aggregates = aggregate_contextual_hierarchical_skill(records)
    try:
        selected = select_contextual_hierarchical_skill(
            aggregates,
            city,
            variable,
            lead_hours,
            season=season or "unknown",
            regime=regime or "UNKNOWN",
            minimum_samples=1,
        )
    except ValueError:
        try:
            selected = select_contextual_hierarchical_skill(
                aggregates,
                city,
                variable,
                lead_hours,
                season=season or None,
                regime=None,
                minimum_samples=1,
            )
        except ValueError:
            return None

    gfs_mae = selected["gfs"].get("mae")
    gefs_mae = selected["gefs"].get("mae")
    if gfs_mae is None or gefs_mae is None:
        return None

    weights = adaptive_weights({"gfs": float(gfs_mae), "gefs": float(gefs_mae)})
    return {
        "season": season,
        "regime": regime,
        "skill_source": selected["source"],
        "weights": weights,
        "gfs_mae": float(gfs_mae),
        "gefs_mae": float(gefs_mae),
        "gfs_sample_count": int(selected["gfs"].get("sample_count", 0)),
        "gefs_sample_count": int(selected["gefs"].get("sample_count", 0)),
    }


def get_forecast_weights(
	latitude: float,
	longitude: float,
	variable: str,
	lead_hours: int,
	season: str | None = None,
	regime: str | None = None,
	path: str | Path | None = None,
    historical_path: str | Path | None = None,
) -> dict[str, Any]:
	"""Return weights and provenance from the canonical spatial artifact.

	When a forecast-time season/regime is supplied, contextual historical skill is
	used first and only falls back to the spatial baseline if there is no safe
	contextual bucket to use.
	"""
	kwargs = {} if path is None else {"path": path}
	result = get_spatial_weights(
		latitude=latitude,
		longitude=longitude,
		variable=variable,
		lead_hours=lead_hours,
		**kwargs,
	)
	contextual = _contextual_weight_lookup(
	    latitude=latitude,
	    longitude=longitude,
	    variable=variable,
	    lead_hours=lead_hours,
	    season=season,
	    regime=regime,
	    historical_path=historical_path,
	)
	if contextual is not None:
		result.update(contextual)
	result["season"] = season
	result["regime"] = regime
	return result