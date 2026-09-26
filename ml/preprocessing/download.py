"""Download GFS GRIB2 files from NOAA through Herbie."""

import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
GFS_REQUEST_TIMEOUT = 30


class GFSDownloadError(RuntimeError):
    """Raised when no GFS subset could be downloaded from any source."""

    def __init__(self, message: str, source_failures: list[dict[str, Any]], missing_subsets: list[str]):
        super().__init__(message)
        self.source_failures = source_failures
        self.missing_subsets = missing_subsets


class GFSDownloadResult(dict[str, Path]):
    """Downloaded GFS subsets plus non-schema diagnostics for the builder."""

    def __init__(
        self,
        paths: dict[str, Path],
        source_failures: list[dict[str, Any]],
        missing_subsets: list[str],
        subset_sources: dict[str, str] | None = None,
    ):
        super().__init__(paths)
        self.source_failures = source_failures
        self.missing_subsets = missing_subsets
        self.subset_sources = subset_sources or {}


@contextmanager
def _bounded_network() -> Iterator[None]:
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(GFS_REQUEST_TIMEOUT)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous_timeout)


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
    source: str | None = None,
):
    from herbie import Herbie

    return Herbie(
        date=_as_naive_utc(date),
        model="gfs",
        product=product,
        fxx=forecast_hour,
        priority=GFS_SOURCES if source is None else [source],
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

    failures = []
    for source in GFS_SOURCES:
        try:
            with _bounded_network():
                forecast = _create_herbie(
                    date=date,
                    forecast_hour=forecast_hour,
                    product=product,
                    save_dir=destination,
                    source=source,
                )
                downloaded = forecast.download(
                    save_dir=destination,
                    overwrite=False,
                    errors="raise",
                    source=source,
                )
            if downloaded is None:
                raise RuntimeError("Herbie returned no GFS file")
            path = Path(downloaded)
            if not path.exists():
                raise FileNotFoundError(f"Herbie reported a missing GFS file: {path}")
            return path
        except Exception as exc:
            failures.append({"source": source, "reason": f"{type(exc).__name__}: {exc}"})

    raise GFSDownloadError(
        f"Could not download GFS F{forecast_hour:03d} from any configured source",
        failures,
        ["full_file"],
    )


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

    paths: dict[str, Path] = {}
    subset_sources: dict[str, str] = {}
    source_failures: list[dict[str, Any]] = []
    missing_subsets: list[str] = []
    forecasts: dict[str, Any] = {}
    unavailable_sources: set[str] = set()

    for name, search in GFS_SUBSET_SEARCHES.items():
        for source in GFS_SOURCES:
            if source in unavailable_sources:
                continue
            try:
                if source not in forecasts:
                    with _bounded_network():
                        forecasts[source] = _create_herbie(
                            date=date,
                            forecast_hour=forecast_hour,
                            product=product,
                            save_dir=destination,
                            source=source,
                        )
                with _bounded_network():
                    downloaded = forecasts[source].download(
                        search=search,
                        save_dir=destination,
                        overwrite=False,
                        errors="raise",
                        source=source,
                    )
                if downloaded is None:
                    raise RuntimeError("Herbie returned no GFS subset")
                path = Path(downloaded)
                if not path.exists():
                    raise FileNotFoundError(f"Herbie reported a missing GFS subset: {path}")
                paths[name] = path
                subset_sources[name] = source
                break
            except Exception as exc:
                source_failures.append(
                    {
                        "subset": name,
                        "source": source,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                if source not in forecasts:
                    unavailable_sources.add(source)
        else:
            missing_subsets.append(name)

    if not paths:
        raise GFSDownloadError(
            f"Could not download any GFS subset for F{forecast_hour:03d}",
            source_failures,
            missing_subsets,
        )

    return GFSDownloadResult(paths, source_failures, missing_subsets, subset_sources)