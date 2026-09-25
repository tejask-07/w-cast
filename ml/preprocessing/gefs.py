"""GEFS ensemble-mean download and point extraction."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.preprocessing.download import _as_naive_utc
from ml.preprocessing.gfs_extract import extract_gfs_point

GEFS_MODEL = "gefs"
GEFS_PRODUCT = "atmos.25"
GEFS_MEMBER = "avg"
GEFS_PRIORITY = ["nomads"]


def _is_archive_unavailable(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        "no index file was found" in message
        or "did not find" in message
        or "index file" in message and "none" in message
    )


def _inventory_searches(forecast: Any) -> dict[str, str]:
    """Find unique verified field searches in the actual GEFS index."""
    inventory = forecast.inventory()
    requirements = {
        "temperature": ("TMP", "2 m above ground"),
        "wind_u": ("UGRD", "10 m above ground"),
        "wind_v": ("VGRD", "10 m above ground"),
        "precipitation": ("APCP", "surface"),
    }
    searches: dict[str, str] = {}
    for name, (variable, level) in requirements.items():
        matches = inventory[
            (inventory["variable"] == variable)
            & (inventory["level"] == level)
            & inventory["search_this"].str.contains("ens mean", na=False)
        ]
        if len(matches) == 1:
            searches[name] = str(matches.iloc[0]["search_this"])
            continue

        if name == "precipitation" and len(matches) == 0:
            # APCP ensemble mean is not guaranteed to be present
            # in this GEFS inventory/forecast hour.
            continue

        raise RuntimeError(
            f"GEFS inventory did not provide exactly one ensemble-mean {name} field "
            f"for {variable} at {level}; found {len(matches)}"
        )
    return searches


def download_gefs_subsets(
    date: datetime,
    forecast_hour: int,
    save_dir: str | Path = "data/raw/gefs",
    product: str = GEFS_PRODUCT,
    member: str = GEFS_MEMBER,
) -> dict[str, Path] | None:
    """Download ensemble-mean GEFS subsets from NOAA NOMADS.

    The function queries the real Herbie index and downloads only the four
    matching ensemble-mean messages. No GEFS ensemble member is selected as a
    proxy: ``member='avg'`` is the official GEFS ensemble mean product.
    """
    if forecast_hour < 0:
        raise ValueError("forecast_hour must be non-negative")
    if member != GEFS_MEMBER:
        raise ValueError("Only member='avg' is supported for the GEFS MVP")
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from herbie import Herbie

        forecast = Herbie(
            date=_as_naive_utc(date),
            model=GEFS_MODEL,
            product=product,
            member=member,
            fxx=forecast_hour,
            priority=GEFS_PRIORITY,
        )
    except Exception as exc:
        if _is_archive_unavailable(exc):
            return None
        raise RuntimeError(
            f"Could not inspect GEFS ensemble-mean inventory on NOAA NOMADS: {exc}"
        ) from exc

    if getattr(forecast, "idx", True) in (None, False):
        return None

    try:
        searches = _inventory_searches(forecast)
    except Exception as exc:
        if _is_archive_unavailable(exc):
            return None
        raise RuntimeError(
            f"Could not inspect GEFS ensemble-mean inventory on NOAA NOMADS: {exc}"
        ) from exc

    paths: dict[str, Path] = {}
    for name, search in searches.items():
        try:
            downloaded = forecast.download(search=search, save_dir=destination)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download GEFS subset '{name}' from NOAA NOMADS "
                f"using inventory search '{search}': {exc}"
            ) from exc
        if downloaded is None:
            raise RuntimeError(f"NOAA NOMADS returned no GEFS file for '{name}'")
        path = Path(downloaded)
        if not path.is_file():
            raise FileNotFoundError(f"Herbie reported a missing GEFS subset file: {path}")
        paths[name] = path
    return paths


def extract_gefs_point(
    file_paths: Mapping[str, str | Path],
    lat: float,
    lon: float,
    lead_hours: int | None = None,
) -> dict[str, Any]:
    """Extract the GEFS ensemble mean using the normalized GFS schema."""
    result = extract_gfs_point(file_paths, lat, lon, lead_hours)
    result.update(
        {
            "temperature_source": "GEFS ensemble mean TMP 2m",
            "wind_source": "GEFS ensemble mean UGRD/VGRD 10m",
            "precipitation_source": "GEFS ensemble mean APCP",
            "ensemble_member": GEFS_MEMBER,
            "ensemble_representation": "official GEFS ensemble mean",
        }
    )
    return result
