"""Download GFS GRIB2 files from NOAA NOMADS through Herbie."""

from datetime import datetime, timezone
from pathlib import Path


GFS_SUBSET_SEARCHES = {
	"temperature": ":TMP:2 m above ground:",
	"wind_u": ":UGRD:10 m above ground:",
	"wind_v": ":VGRD:10 m above ground:",
	"precipitation": ":APCP:surface:0-1 day acc fcst:",
}


def _as_naive_utc(date: datetime) -> datetime:
	"""Return a datetime suitable for Herbie's timezone-naive GFS template."""
	if date.tzinfo is not None:
		return date.astimezone(timezone.utc).replace(tzinfo=None)
	return date


def download_gfs(
	date: datetime,
	forecast_hour: int,
	product: str = "pgrb2.0p25",
	save_dir: str | Path = "data/raw/gfs",
) -> Path:
	"""Download one GFS forecast and return its local GRIB2 path.

	Herbie resolves the concrete NOMADS URL and filename. The source is
	deliberately restricted to NOMADS and SSL verification is left enabled.
	"""
	if forecast_hour < 0:
		raise ValueError("forecast_hour must be non-negative")
	destination = Path(save_dir)
	destination.mkdir(parents=True, exist_ok=True)
	from herbie import Herbie

	forecast = Herbie(
		date=_as_naive_utc(date),
		model="gfs",
		product=product,
		fxx=forecast_hour,
		priority=["nomads"],
	)
	downloaded = forecast.download(save_dir=destination)
	path = Path(downloaded)
	if not path.exists():
		raise FileNotFoundError(f"Herbie reported a missing downloaded file: {path}")
	return path


def download_gfs_subsets(
	date: datetime,
	forecast_hour: int,
	product: str = "pgrb2.0p25",
	save_dir: str | Path = "data/raw/gfs",
) -> dict[str, Path]:
	"""Download the four verified GFS field subsets from NOAA NOMADS.

	Raises:
	    RuntimeError: If Herbie cannot resolve or download a requested subset.
    FileNotFoundError: If Herbie returns a path that does not exist locally.
	"""
	if forecast_hour < 0:
		raise ValueError("forecast_hour must be non-negative")
	destination = Path(save_dir)
	destination.mkdir(parents=True, exist_ok=True)
	from herbie import Herbie

	try:
		forecast = Herbie(
			date=_as_naive_utc(date),
			model="gfs",
			product=product,
			fxx=forecast_hour,
			priority=["nomads"],
		)
	except Exception as exc:
		raise RuntimeError(f"Could not access GFS forecast on NOAA NOMADS: {exc}") from exc
	paths: dict[str, Path] = {}
	for name, search in GFS_SUBSET_SEARCHES.items():
		try:
			downloaded = forecast.download(search=search, save_dir=destination)
		except Exception as exc:
			raise RuntimeError(
				f"Could not download GFS subset '{name}' from NOAA NOMADS "
				f"using search '{search}': {exc}"
			) from exc
		if downloaded is None:
			raise RuntimeError(
				f"NOAA NOMADS returned no file for GFS subset '{name}' "
				f"using search '{search}'"
			)
		path = Path(downloaded)
		if not path.exists():
			raise FileNotFoundError(f"Herbie reported a missing subset file: {path}")
		paths[name] = path
	return paths
