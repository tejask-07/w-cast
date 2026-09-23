"""Download GFS GRIB2 files from NOAA through Herbie."""

from datetime import datetime, timezone
from pathlib import Path


GFS_SUBSET_SEARCHES = {
    "temperature": ":TMP:2 m above ground:",
    "wind_u": ":UGRD:10 m above ground:",
    "wind_v": ":VGRD:10 m above ground:",
    "precipitation": ":APCP:surface:",
}

GFS_SOURCES = [
    "nomads",
    "aws",
    "google",
    "azure",
]


def _as_naive_utc(date: datetime) -> datetime:
    """Return a timezone-naive UTC datetime for Herbie."""
    if date.tzinfo is not None:
        return date.astimezone(timezone.utc).replace(tzinfo=None)

    return date


def _create_herbie(
    date: datetime,
    forecast_hour: int,
    product: str,
    save_dir: Path,
):
    from herbie import Herbie

    return Herbie(
        date=_as_naive_utc(date),
        model="gfs",
        product=product,
        fxx=forecast_hour,
        priority=GFS_SOURCES,
        save_dir=save_dir,
        overwrite=False,
    )


def download_gfs(
    date: datetime,
    forecast_hour: int,
    product: str = "pgrb2.0p25",
    save_dir: str | Path = "data/raw/gfs",
) -> Path:
    """Download one complete GFS forecast file."""

    if forecast_hour < 0:
        raise ValueError("forecast_hour must be non-negative")

    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        forecast = _create_herbie(
            date=date,
            forecast_hour=forecast_hour,
            product=product,
            save_dir=destination,
        )

        downloaded = forecast.download(
            save_dir=destination,
            overwrite=False,
            errors="raise",
        )

    except Exception as exc:
        raise RuntimeError(
            f"Could not download GFS F{forecast_hour:03d}: {exc}"
        ) from exc

    if downloaded is None:
        raise RuntimeError(
            f"Herbie returned no GFS file for F{forecast_hour:03d}"
        )

    path = Path(downloaded)

    if not path.exists():
        raise FileNotFoundError(
            f"Herbie reported a missing GFS file: {path}"
        )

    return path


def download_gfs_subsets(
    date: datetime,
    forecast_hour: int,
    product: str = "pgrb2.0p25",
    save_dir: str | Path = "data/raw/gfs",
) -> dict[str, Path]:
    """Download the required GFS field subsets."""

    if forecast_hour < 0:
        raise ValueError("forecast_hour must be non-negative")

    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)

    try:
        forecast = _create_herbie(
            date=date,
            forecast_hour=forecast_hour,
            product=product,
            save_dir=destination,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not access GFS forecast F{forecast_hour:03d}: {exc}"
        ) from exc

    paths: dict[str, Path] = {}

    for name, search in GFS_SUBSET_SEARCHES.items():
        try:
            downloaded = forecast.download(
                search=search,
                save_dir=destination,
                overwrite=False,
                errors="raise",
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not download GFS subset '{name}' "
                f"for F{forecast_hour:03d}: {exc}"
            ) from exc

        if downloaded is None:
            raise RuntimeError(
                f"No GFS subset returned for '{name}' "
                f"at F{forecast_hour:03d}"
            )

        path = Path(downloaded)

        if not path.exists():
            raise FileNotFoundError(
                f"Herbie reported a missing GFS subset for "
                f"'{name}': {path}"
            )

        paths[name] = path

    return paths