from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")
GFS_SAVE_DIR = Path("data/raw/gfs")

download_gfs_subsets = None
extract_gfs_point = None


def _load_ml_gfs_modules():
    global download_gfs_subsets, extract_gfs_point
    if download_gfs_subsets is None or extract_gfs_point is None:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from ml.preprocessing.download import download_gfs_subsets as _download
        from ml.preprocessing.gfs_extract import extract_gfs_point as _extract

        if download_gfs_subsets is None:
            download_gfs_subsets = _download
        if extract_gfs_point is None:
            extract_gfs_point = _extract
    return download_gfs_subsets, extract_gfs_point


def select_gfs_cycle(now: datetime | None = None) -> datetime:
    """Return the latest completed GFS cycle in UTC.

    Standard GFS cycles are available at 00, 06, 12, and 18 UTC. We choose the
    latest cycle whose analysis/forecast window has already completed relative to
    the supplied current time.
    """
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cycle_hours = [0, 6, 12, 18]
    candidate = None
    for hour in cycle_hours:
        cycle = current.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cycle <= current:
            candidate = cycle
    if candidate is None:
        previous = current - timedelta(days=1)
        candidate = previous.replace(hour=18, minute=0, second=0, microsecond=0)
    return candidate


def map_api_variable_to_gfs(variable: str) -> str:
    mapping = {
        "temperature": "temperature_C",
        "rainfall": "precipitation_mm",
        "wind_speed": "wind_speed_ms",
    }
    if variable not in mapping:
        raise ValueError(f"Unsupported variable: {variable}")
    return mapping[variable]


def generate_real_gfs_forecast(lat: float, lon: float, lead_hours: int, variable: str) -> dict[str, float]:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")

    download_gfs_subsets, extract_gfs_point = _load_ml_gfs_modules()
    cycle_time = select_gfs_cycle()
    try:
        subset_paths = download_gfs_subsets(
            date=cycle_time,
            forecast_hour=lead_hours,
            save_dir=GFS_SAVE_DIR,
        )
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise RuntimeError("GFS data unavailable from NOAA NOMADS") from exc

    try:
        extracted = extract_gfs_point(subset_paths, lat, lon, lead_hours=lead_hours)
    except Exception as exc:  # pragma: no cover - service boundary failsafe
        raise RuntimeError("GFS extraction failed for the requested point") from exc

    required_fields = {
        "temperature": "temperature_C",
        "rainfall": "precipitation_mm",
        "wind_speed": "wind_speed_ms",
    }
    missing = [name for name, field in required_fields.items() if extracted.get(field) is None]
    if missing:
        raise RuntimeError(f"GFS field unavailable for requested forecast variables: {', '.join(missing)}")

    mapped_field = map_api_variable_to_gfs(variable)
    if extracted.get(mapped_field) is None:
        raise RuntimeError(f"GFS field missing for variable '{variable}'")

    return {
        "temperature": float(extracted["temperature_C"]),
        "rainfall": float(extracted["precipitation_mm"]),
        "wind_speed": float(extracted["wind_speed_ms"]),
    }
