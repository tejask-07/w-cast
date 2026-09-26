"""Historical operational GEFS adapter using NOAA's AWS Open Data archive."""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from ml.preprocessing.download import _as_naive_utc
from ml.preprocessing.gefs import extract_gefs_point

GEFS_HISTORICAL_SOURCE = "noaa-gefs-operational-aws"
GEFS_SOURCE_DESCRIPTION = "GEFS operational archive (NOAA AWS Open Data)"
GEFS_PRODUCT = "atmos.25"
GEFS_MEMBER = "avg"
GEFS_PRIORITY = ["aws"]
GEFS_RESOLUTION_DEGREES = 0.25
SUPPORTED_LEADS = (24, 48, 72)
VARIABLE_FIELDS = {
    "temperature": ("TMP", "2 m above ground", "temperature", "temperature_C"),
    "rainfall": ("APCP", "surface", "precipitation", "precipitation_mm"),
    "wind_speed": ("WIND", "10 m above ground", "wind", "wind_speed_ms"),
}
ACCUMULATION_INTERVAL_HOURS = 3


class HistoricalGEFSUnavailable(RuntimeError):
    """Raised when the exact operational GEFS cycle/lead/field is unavailable."""


def initialization_time(forecast_date: date | str, cycle_hour: int) -> datetime:
    if isinstance(forecast_date, datetime):
        day = forecast_date.astimezone(timezone.utc).date() if forecast_date.tzinfo else forecast_date.date()
    elif isinstance(forecast_date, date):
        day = forecast_date
    else:
        try:
            day = date.fromisoformat(forecast_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("forecast_date must be an ISO date") from exc
    if cycle_hour not in (0, 6, 12, 18):
        raise ValueError("forecast_cycle must be one of 0, 6, 12, or 18 UTC")
    return datetime.combine(day, time(cycle_hour), tzinfo=timezone.utc)


def validate_pair_timestamps(
    gfs_initialization: datetime,
    gefs_initialization: datetime,
    gfs_valid_time: datetime,
    gefs_valid_time: datetime,
    lead_hours: int,
) -> None:
    """Require both models to initialize together and share the same valid time."""
    normalize = lambda value: value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    gfs_init = normalize(gfs_initialization)
    gefs_init = normalize(gefs_initialization)
    gfs_valid = normalize(gfs_valid_time)
    gefs_valid = normalize(gefs_valid_time)
    if gfs_init != gefs_init:
        raise ValueError("GFS and GEFS initialization times do not match")
    expected = gfs_init + timedelta(hours=lead_hours)
    if gfs_valid != expected or gefs_valid != expected:
        raise ValueError("GFS and GEFS valid times do not match the requested lead")


def _lead_matches(forecast_time: Any, lead_hours: int, accumulation: bool) -> bool:
    value = str(forecast_time).strip()
    if accumulation:
        return re.fullmatch(rf"(?:\d+-)?{lead_hours} hour acc fcst", value) is not None
    return value == f"{lead_hours} hour fcst"


def _search_for(inventory, grib_variable: str, level: str, lead_hours: int) -> str:
    matches = inventory[
        (inventory["variable"] == grib_variable)
        & (inventory["level"] == level)
        & inventory["search_this"].str.contains("ens mean", na=False)
        & inventory["forecast_time"].map(
            lambda value: _lead_matches(
                value,
                lead_hours,
                accumulation=grib_variable == "APCP",
            )
        )
    ]
    if len(matches) != 1:
        raise HistoricalGEFSUnavailable(
            f"Expected one GEFS ensemble-mean {grib_variable} field at {level} "
            f"for F{lead_hours:03d}; found {len(matches)}"
        )
    return str(matches.iloc[0]["search_this"])


class HistoricalGEFSAdapter:
    """Download/extract exact-cycle operational GEFS ensemble-mean points."""

    def __init__(self, save_dir: str | Path = "data/raw/gefs/historical_aws") -> None:
        self.save_dir = Path(save_dir)
        self._forecasts: dict[tuple[datetime, int], Any] = {}
        self._inventories: dict[tuple[datetime, int], Any] = {}
        self._searches: dict[tuple[datetime, int, str], str] = {}
        self._paths: dict[tuple[datetime, int, str], Path] = {}

    def _forecast(self, init: datetime, lead_hours: int):
        key = (init, lead_hours)
        if key in self._forecasts:
            return self._forecasts[key]
        try:
            from herbie import Herbie

            self.save_dir.mkdir(parents=True, exist_ok=True)
            forecast = Herbie(
                date=_as_naive_utc(init),
                model="gefs",
                product=GEFS_PRODUCT,
                member=GEFS_MEMBER,
                fxx=lead_hours,
                priority=GEFS_PRIORITY,
                save_dir=self.save_dir,
            )
        except Exception as exc:
            raise HistoricalGEFSUnavailable(
                f"Could not create AWS GEFS request for {init.isoformat()} F{lead_hours:03d}: {exc}"
            ) from exc
        if getattr(forecast, "idx", None) in (None, False):
            raise HistoricalGEFSUnavailable(
                f"AWS operational GEFS index unavailable for {init.isoformat()} F{lead_hours:03d}"
            )
        self._forecasts[key] = forecast
        return forecast

    def _field_path(
        self,
        forecast: Any,
        init: datetime,
        lead_hours: int,
        variable: str,
        field_name: str,
        grib_variable: str,
        level: str,
    ) -> Path:
        field_key = (init, lead_hours, field_name)
        if field_key in self._paths:
            return self._paths[field_key]
        if (init, lead_hours) not in self._inventories:
            try:
                self._inventories[(init, lead_hours)] = forecast.inventory()
            except Exception as exc:
                raise HistoricalGEFSUnavailable(
                    f"Could not load AWS GEFS inventory for {init.isoformat()} F{lead_hours:03d}: {exc}"
                ) from exc
        search_key = (init, lead_hours, field_name)
        if search_key not in self._searches:
            try:
                self._searches[search_key] = _search_for(
                    self._inventories[(init, lead_hours)],
                    grib_variable,
                    level,
                    lead_hours,
                )
            except HistoricalGEFSUnavailable:
                raise
            except Exception as exc:
                raise HistoricalGEFSUnavailable(
                    f"Could not select AWS GEFS {variable} inventory field: {exc}"
                ) from exc
        try:
            downloaded = forecast.download(
                search=self._searches[search_key],
                save_dir=self.save_dir,
            )
        except Exception as exc:
            raise HistoricalGEFSUnavailable(
                f"Could not download AWS GEFS {field_name} for {init.isoformat()} F{lead_hours:03d}: {exc}"
            ) from exc
        if downloaded is None or not Path(downloaded).is_file():
            raise HistoricalGEFSUnavailable(
                f"AWS GEFS returned no file for {field_name} at {init.isoformat()} F{lead_hours:03d}"
            )
        self._paths[field_key] = Path(downloaded)
        return self._paths[field_key]

    def get_forecast(
        self,
        forecast_date: date | str,
        forecast_cycle: int,
        lead_hours: int,
        variable: str,
        latitude: float,
        longitude: float,
    ) -> dict[str, Any]:
        if variable not in VARIABLE_FIELDS:
            raise ValueError(f"Unsupported historical GEFS variable: {variable}")
        if lead_hours not in SUPPORTED_LEADS:
            raise ValueError(f"lead_hours must be one of {SUPPORTED_LEADS}")
        if not math.isfinite(float(latitude)) or not -90 <= float(latitude) <= 90:
            raise ValueError("latitude must be between -90 and 90")
        if not math.isfinite(float(longitude)) or not -180 <= float(longitude) <= 180:
            raise ValueError("longitude must be between -180 and 180")

        init = initialization_time(forecast_date, forecast_cycle)
        valid = init + timedelta(hours=lead_hours)
        grib_variable, level, output_field, point_field = VARIABLE_FIELDS[variable]
        file_fields: dict[str, Path] = {}

        if variable == "rainfall":
            accumulated = 0.0
            for interval_end in range(ACCUMULATION_INTERVAL_HOURS, lead_hours + 1, ACCUMULATION_INTERVAL_HOURS):
                forecast = self._forecast(init, interval_end)
                path = self._field_path(
                    forecast,
                    init,
                    interval_end,
                    variable,
                    output_field,
                    grib_variable,
                    level,
                )
                try:
                    interval = extract_gefs_point(
                        {output_field: path},
                        latitude,
                        longitude,
                        interval_end,
                    ).get(point_field)
                except Exception as exc:
                    raise HistoricalGEFSUnavailable(
                        f"Could not extract AWS GEFS rainfall interval ending F{interval_end:03d}: {exc}"
                    ) from exc
                if interval is None:
                    raise HistoricalGEFSUnavailable(
                        f"AWS GEFS rainfall interval ending F{interval_end:03d} is missing"
                    )
                numeric_interval = float(interval)
                if not math.isfinite(numeric_interval):
                    raise HistoricalGEFSUnavailable(
                        f"AWS GEFS rainfall interval ending F{interval_end:03d} is non-finite"
                    )
                accumulated += numeric_interval
            numeric = accumulated
            return {
                "forecast_value": numeric,
                "source": GEFS_HISTORICAL_SOURCE,
                "gefs_source": GEFS_SOURCE_DESCRIPTION,
                "initialization_time": init,
                "valid_time": valid,
                "spatial_resolution_degrees": GEFS_RESOLUTION_DEGREES,
                "accumulation_period_hours": lead_hours,
            }

        forecast = self._forecast(init, lead_hours)
        if variable == "wind_speed":
            for field_name, component in (("wind_u", "UGRD"), ("wind_v", "VGRD")):
                file_fields[field_name] = self._field_path(
                    forecast,
                    init,
                    lead_hours,
                    variable,
                    field_name,
                    component,
                    "10 m above ground",
                )
        else:
            file_fields[output_field] = self._field_path(
                forecast,
                init,
                lead_hours,
                variable,
                output_field,
                grib_variable,
                level,
            )

        try:
            extracted = extract_gefs_point(file_fields, latitude, longitude, lead_hours)
        except Exception as exc:
            raise HistoricalGEFSUnavailable(
                f"Could not extract AWS GEFS {variable} at {latitude},{longitude}: {exc}"
            ) from exc
        value = extracted.get(point_field)
        if value is None:
            raise HistoricalGEFSUnavailable(
                f"AWS GEFS {variable} is missing at {latitude},{longitude} for F{lead_hours:03d}"
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise HistoricalGEFSUnavailable(f"AWS GEFS {variable} value is invalid") from exc
        if not math.isfinite(numeric):
            raise HistoricalGEFSUnavailable(f"AWS GEFS {variable} value is non-finite")

        return {
            "forecast_value": numeric,
            "source": GEFS_HISTORICAL_SOURCE,
            "gefs_source": GEFS_SOURCE_DESCRIPTION,
            "initialization_time": init,
            "valid_time": valid,
            "spatial_resolution_degrees": GEFS_RESOLUTION_DEGREES,
        }