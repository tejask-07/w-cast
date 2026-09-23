"""Build matched historical GFS, GEFS, and observation records."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ml.evaluation.historical_skill import build_verification_record
from ml.preprocessing.align import CITIES
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets, extract_gefs_point
from ml.preprocessing.gfs_extract import extract_gfs_point
from ml.preprocessing.observations import get_historical_observations


DEFAULT_CITIES = ("Mumbai", "Delhi", "Kolkata", "Chennai")

DEFAULT_VARIABLES = (
    "temperature",
    "precipitation",
    "wind_speed",
)

VARIABLE_FIELDS = {
    "temperature": "temperature_C",
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}


def _utc(value: datetime | str, name: str) -> datetime:
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid datetime") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return (
            value.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            key: _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]

    return value


def _observation_at(
    observations: Mapping[str, Any],
    valid_time: datetime,
) -> dict[str, Any] | None:

    timestamps = observations.get("hourly", [])

    for index, timestamp in enumerate(timestamps):
        if _utc(timestamp, "observation timestamp") == valid_time:
            return {
                field: observations.get(
                    field,
                    [None] * len(timestamps),
                )[index]
                for field in VARIABLE_FIELDS.values()
            }

    return None


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def build_historical_records(
    dates: Iterable[datetime | str],
    cities: Iterable[str] | None = None,
    lead_hours: int | Iterable[int] = 24,
    variables: Iterable[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:

    city_list = list(
        DEFAULT_CITIES if cities is None else cities
    )

    variable_list = list(
        DEFAULT_VARIABLES if variables is None else variables
    )

    if isinstance(lead_hours, int):
        lead_list = [lead_hours]
    else:
        lead_list = list(lead_hours)

    if not lead_list:
        raise ValueError("At least one lead time is required")

    if any(
        not isinstance(lead, int) or lead < 0
        for lead in lead_list
    ):
        raise ValueError(
            "lead_hours must contain non-negative integers"
        )

    unknown_cities = [
        city for city in city_list
        if city not in CITIES
    ]

    if unknown_cities:
        raise ValueError(
            f"Unknown city '{unknown_cities[0]}'"
        )

    unknown_variables = [
        variable for variable in variable_list
        if variable not in VARIABLE_FIELDS
    ]

    if unknown_variables:
        raise ValueError(
            f"Unsupported variable '{unknown_variables[0]}'"
        )

    init_times = [
        _utc(value, "dates")
        for value in dates
    ]

    init_times = list(dict.fromkeys(init_times))

    now = datetime.now(timezone.utc)

    for init_time in init_times:
        if init_time > now:
            raise ValueError(
                "historical dates cannot be in the future"
            )

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    forecast_cache: dict[
        tuple[datetime, int],
        dict[str, tuple[
            Mapping[str, Any] | None,
            Mapping[str, Any] | None,
        ]],
    ] = {}

    observation_cache: dict[
        str,
        Mapping[str, Any] | None,
    ] = {}

    for init_time in init_times:
        for lead in lead_list:

            valid_time = init_time + timedelta(hours=lead)

            if valid_time > now:
                continue

            print(
                f"Processing "
                f"{init_time.strftime('%Y-%m-%d %H:%M')} UTC "
                f"F{lead:03d}"
            )

            gfs_files = None
            gefs_files = None

            try:
                gfs_files = download_gfs_subsets(
                    init_time,
                    lead,
                )
            except Exception as exc:
                gfs_reason = _reason(exc)
                print(f"  GFS unavailable: {gfs_reason}")
            else:
                gfs_reason = None

            try:
                gefs_files = download_gefs_subsets(
                    init_time,
                    lead,
                )
            except Exception as exc:
                gefs_reason = _reason(exc)
                print(f"  GEFS unavailable: {gefs_reason}")
            else:
                gefs_reason = None

            forecasts_by_city = {}

            for city in city_list:

                latitude, longitude = CITIES[city]

                gfs = None
                gefs = None

                if gfs_files is not None:
                    try:
                        gfs = extract_gfs_point(
                            gfs_files,
                            latitude,
                            longitude,
                            lead,
                        )
                    except Exception as exc:
                        gfs_reason = _reason(exc)

                if gefs_files is not None:
                    try:
                        gefs = extract_gefs_point(
                            gefs_files,
                            latitude,
                            longitude,
                            lead,
                        )
                    except Exception as exc:
                        gefs_reason = _reason(exc)

                forecasts_by_city[city] = (
                    gfs,
                    gefs,
                )

            forecast_cache[(init_time, lead)] = (
                forecasts_by_city
            )

    valid_times = [
        init_time + timedelta(hours=lead)
        for init_time in init_times
        for lead in lead_list
        if init_time + timedelta(hours=lead) <= now
    ]

    if valid_times:

        start_date = min(valid_times).date()
        end_date = max(valid_times).date()

        for city in city_list:

            latitude, longitude = CITIES[city]

            try:
                observation_cache[city] = (
                    get_historical_observations(
                        latitude,
                        longitude,
                        start_date,
                        end_date,
                    )
                )

            except Exception as exc:

                observation_cache[city] = None

                for init_time in init_times:
                    for lead in lead_list:

                        valid_time = (
                            init_time
                            + timedelta(hours=lead)
                        )

                        if valid_time <= now:
                            skipped.append(
                                {
                                    "city": city,
                                    "forecast_init_time": init_time,
                                    "valid_time": valid_time,
                                    "lead_hours": lead,
                                    "reason": _reason(exc),
                                }
                            )

    for init_time in init_times:

        for lead in lead_list:

            valid_time = (
                init_time
                + timedelta(hours=lead)
            )

            if valid_time > now:
                continue

            for city in city_list:

                observation = _observation_at(
                    observation_cache.get(city) or {},
                    valid_time,
                )

                if observation is None:

                    if observation_cache.get(city) is not None:
                        skipped.append(
                            {
                                "city": city,
                                "forecast_init_time": init_time,
                                "valid_time": valid_time,
                                "lead_hours": lead,
                                "reason": (
                                    "no observation at exact valid_time"
                                ),
                            }
                        )

                    continue

                gfs, gefs = forecast_cache[
                    (init_time, lead)
                ][city]

                for variable in variable_list:

                    field = VARIABLE_FIELDS[variable]

                    if observation.get(field) is None:

                        skipped.append(
                            {
                                "city": city,
                                "variable": variable,
                                "forecast_init_time": init_time,
                                "valid_time": valid_time,
                                "lead_hours": lead,
                                "reason": (
                                    f"observation field "
                                    f"'{field}' is unavailable"
                                ),
                            }
                        )

                        continue

                    records.append(
                        build_verification_record(
                            init_time,
                            valid_time,
                            city,
                            lead,
                            variable,
                            {
                                "gfs": gfs,
                                "gefs": gefs,
                            },
                            observation,
                        )
                    )

    result = {
        "records": records,
        "skipped": skipped,
    }

    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(
                _json_value(result),
                indent=2,
            ),
            encoding="utf-8",
        )

    return result


def load_historical_records(
    path: str | Path,
) -> dict[str, list[dict[str, Any]]]:

    payload = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    for collection in (
        payload.get("records", []),
        payload.get("skipped", []),
    ):

        for item in collection:

            for key in (
                "forecast_init_time",
                "valid_time",
            ):

                if key in item:
                    item[key] = _utc(
                        item[key],
                        key,
                    )

    return payload