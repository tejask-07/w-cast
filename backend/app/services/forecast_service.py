from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from app.services.gfs_forecast_service import (
    generate_real_gfs_forecast as _generate_real_gfs_forecast,
)

from ml.pipeline import generate_forecast as generate_forecast_ml
from ml.spatial.india_grid import is_in_india
from ml.spatial.lookup import get_spatial_weights
from ml.preprocessing.download import (
    GFSDownloadError,
    GFS_SUBSET_SEARCHES,
    download_gfs_subsets,
)
from ml.preprocessing.gefs import download_gefs_subsets
from ml.regimes.classifier import classify_regime
from ml.regimes.extremes import detect_extremes


SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHT_MAP_PATH = REPO_ROOT / "data" / "processed" / "india_weight_map_7d.json"


def resolve_location_name(lat: float, lon: float) -> str | None:
    if not is_in_india(lat, lon):
        return None

    locations = {
        "Mumbai": (19.0760, 72.8777),
        "Delhi": (28.6139, 77.2090),
        "Kolkata": (22.5726, 88.3639),
        "Chennai": (13.0827, 80.2707),
    }

    return min(
        locations,
        key=lambda name: (
            (locations[name][0] - lat) ** 2
            + (locations[name][1] - lon) ** 2
        ),
    )


def validate_variable(variable: str) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")


def generate_real_gfs_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
) -> Dict[str, float]:
    return _generate_real_gfs_forecast(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable=variable,
    )


def _download_forecast_sources(lead_hours: int):
    now = datetime.now(timezone.utc)

    cycles = [18, 12, 6, 0]
    attempted_cycles: list[str] = []
    cycle_failures: list[dict[str, object]] = []
    required_gfs_subsets = set(GFS_SUBSET_SEARCHES)

    for cycle in cycles:
        cycle_time = now.replace(
            hour=cycle,
            minute=0,
            second=0,
            microsecond=0,
        )

        if cycle_time > now:
            cycle_time = cycle_time.replace(
                day=cycle_time.day - 1
            )

        print(
            f"Trying forecast cycle: "
            f"{cycle_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        cycle_label = cycle_time.strftime('%Y-%m-%d %H:%M UTC')
        attempted_cycles.append(cycle_label)

        try:
            gfs_paths = download_gfs_subsets(
                date=cycle_time,
                forecast_hour=lead_hours,
                save_dir="data/raw/gfs",
            )

            missing_gfs_subsets = sorted(
                required_gfs_subsets - set(gfs_paths)
            )
            if missing_gfs_subsets:
                raise GFSDownloadError(
                    f"GFS cycle missing required subsets: "
                    f"{', '.join(missing_gfs_subsets)}",
                    getattr(gfs_paths, "source_failures", []),
                    missing_gfs_subsets,
                )

            gefs_paths = download_gefs_subsets(
                date=cycle_time,
                forecast_hour=lead_hours,
                save_dir="data/raw/gefs",
            )

            print(
                f"Using forecast cycle: "
                f"{cycle_time.strftime('%Y-%m-%d %H:%M UTC')}"
            )

            return gfs_paths, gefs_paths

        except Exception as exc:
            failures = getattr(exc, "source_failures", None)
            if failures:
                cycle_failures.extend(
                    {
                        "cycle": cycle_label,
                        **failure,
                    }
                    for failure in failures
                )
            else:
                cycle_failures.append(
                    {
                        "cycle": cycle_label,
                        "source": "cycle",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            print(
                f"Cycle unavailable: "
                f"{cycle_label} "
                f"({exc})"
            )

    failure_summary = "; ".join(
        f"{failure['cycle']} {failure.get('subset', 'cycle')}/"
        f"{failure.get('source', 'unknown')}: {failure['reason']}"
        for failure in cycle_failures
    )
    message = (
        f"GFS download failed for F{lead_hours:03d}. "
        f"Tried cycles: {', '.join(attempted_cycles)}. "
        f"Failures: {failure_summary}"
    )
    print(message)
    raise GFSDownloadError(
        message,
        cycle_failures,
        sorted(required_gfs_subsets),
    )


def _api_variable(variable: str) -> str:
    return "precipitation" if variable == "rainfall" else variable


def _risk_level(extremes: dict[str, bool]) -> str:
    count = sum(extremes.values())

    if count >= 2:
        return "high"

    if count == 1:
        return "moderate"

    return "low"


def generate_forecast(
    lat: float,
    lon: float,
    lead_hours: int,
    variable: str,
) -> Dict[str, object]:

    validate_variable(variable)

    if lead_hours not in (24, 48, 72):
        raise ValueError("lead_hours must be one of: 24, 48, 72")

    location_name = resolve_location_name(lat, lon)
    history_path = REPO_ROOT / "data" / "processed" / "history_7d_multilead.json"

    gfs_paths, gefs_paths = _download_forecast_sources(
        lead_hours
    )

    result = generate_forecast_ml(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable=_api_variable(variable),
        gfs_file_path=gfs_paths,
        gefs_file_paths=gefs_paths,
        history_path=history_path,
    )

    selected_weights = result["all_weights"][_api_variable(variable)]

    return {
        "location": {
            "name": location_name,
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "forecast": {
            "temperature": result["forecast"]["temperature"],
            "rainfall": result["forecast"]["precipitation"],
            "wind_speed": result["forecast"]["wind_speed"],
        },
        "weights": {
            "gfs": selected_weights["gfs"],
            "gefs": selected_weights["gefs"],
            "baseline": 0.0,
        },
        "regime": result["regime"],
        "model_source": (
            "W-CAST (Adaptive)"
            if is_in_india(lat, lon)
            else "Global GFS + GEFS"
        ),
        "extremes": {
            **result["extremes"],
            "risk_level": _risk_level(result["extremes"]),
        },
    }


def generate_weights(
    lat: float,
    lon: float,
    lead_hours: int,
) -> Dict[str, object]:

    if lead_hours not in (24, 48, 72):
        raise ValueError("lead_hours must be one of: 24, 48, 72")

    variables = {
        "temperature": "temperature",
        "rainfall": "precipitation",
        "wind_speed": "wind_speed",
    }

    result = {}

    for api_name, variable in variables.items():
        result[api_name] = get_spatial_weights(
            latitude=lat,
            longitude=lon,
            variable=variable,
            lead_hours=lead_hours,
            path=WEIGHT_MAP_PATH,
        )

    rainfall = result["rainfall"]

    return {
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "lead_hours": lead_hours,
        "weights": {
            "gfs": rainfall["weights"]["gfs"],
            "gefs": rainfall["weights"]["gefs"],
            "baseline": 0.0,
        },
        "regime": classify_regime(
            0.0
        ),
    }


def generate_extremes(
    lat: float,
    lon: float,
    lead_hours: int,
) -> Dict[str, object]:
    result = generate_forecast(
        lat=lat,
        lon=lon,
        lead_hours=lead_hours,
        variable="temperature",
    )

    return {
        **result["extremes"],
        "risk_level": result["extremes"]["risk_level"],
    }
