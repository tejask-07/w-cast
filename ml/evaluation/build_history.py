"""Build matched historical GFS, GEFS, and observation records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ml.evaluation.historical_skill import build_verification_record
from ml.preprocessing.align import INDIA_LOCATIONS, LOCATION_METADATA
from ml.preprocessing.download import GFSDownloadError, download_gfs_subsets
from ml.preprocessing.gefs import download_gefs_subsets, extract_gefs_point
from ml.preprocessing.gfs_extract import extract_gfs_point
from ml.preprocessing.observations import get_historical_observations

DEFAULT_CITIES = ("Mumbai", "Delhi", "Kolkata", "Chennai")
DEFAULT_LEAD_HOURS = (24, 48, 72)
DEFAULT_OUTPUT = "data/processed/history_30d_20locations_multilead.json"
DEFAULT_VARIABLES = ("temperature", "precipitation", "wind_speed")
VARIABLE_FIELDS = {
    "temperature": "temperature_C",
    "precipitation": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}


def _utc(value: datetime | str, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _observation_at(observations: Mapping[str, Any], valid_time: datetime) -> dict[str, Any] | None:
    timestamps = observations.get("hourly", [])
    for index, timestamp in enumerate(timestamps):
        if _utc(timestamp, "observation timestamp") == valid_time:
            return {
                field: observations.get(field, [None] * len(timestamps))[index]
                for field in VARIABLE_FIELDS.values()
            }
    return None


def _reason(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _location_names(cities: Iterable[str] | None, locations: int | Iterable[str] | None) -> list[str]:
    if cities is not None and locations is not None:
        raise ValueError("Pass either cities or locations, not both")
    if cities is not None:
        return list(cities)
    if locations is None or locations == 4:
        return list(DEFAULT_CITIES)
    if locations == 20:
        return list(INDIA_LOCATIONS)
    if isinstance(locations, int):
        raise ValueError("locations must be 4 or 20 when passed as a count")
    return list(locations)


def historical_initialization_dates(days: int = 30, end_date: date | None = None, init_hour: int = 6) -> list[datetime]:
    """Return recent valid GFS initialization times without hardcoded dates."""
    if days <= 0:
        raise ValueError("days must be positive")
    if init_hour not in range(24):
        raise ValueError("init_hour must be between 0 and 23")
    latest_date = end_date or (datetime.now(timezone.utc).date() - timedelta(days=5))
    latest = datetime.combine(latest_date, datetime.min.time(), tzinfo=timezone.utc).replace(hour=init_hour)
    return [latest - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def _record_key(record: Mapping[str, Any]) -> tuple[datetime, str, int, str]:
    return (
        _utc(record["forecast_init_time"], "forecast_init_time"),
        str(record["city"]),
        int(record["lead_hours"]),
        str(record["variable"]),
    )


def _deduplicate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[datetime, str, int, str], dict[str, Any]] = {}
    for record in records:
        unique.setdefault(_record_key(record), dict(record))
    return list(unique.values())


def _write_history(output_path: str | Path | None, result: Mapping[str, Any]) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(result), indent=2), encoding="utf-8")


def _print_summary(result: Mapping[str, Any]) -> None:
    print(f"RECORDS {len(result['records'])}")
    print(f"SKIPPED {len(result['skipped'])}")
    print(f"MISSING GFS {len(result['missing_gfs'])}")
    print(f"MISSING GEFS {len(result['missing_gefs'])}")
    print(f"GFS SOURCE FAILURES {len(result['gfs_source_failures'])}")
    print(f"GEFS ARCHIVE UNAVAILABLE {len(result['gefs_archive_unavailable'])}")
    for title, values in (
        ("BY CITY", Counter(record["city"] for record in result["records"])),
        ("BY LEAD", Counter(record["lead_hours"] for record in result["records"])),
        ("BY VARIABLE", Counter(record["variable"] for record in result["records"])),
        ("BY REGION", Counter(record["region"] for record in result["records"])),
    ):
        print(title)
        for key in sorted(values, key=str):
            print(f"  {key}: {values[key]}")


def build_historical_records(
    dates: Iterable[datetime | str],
    cities: Iterable[str] | None = None,
    lead_hours: int | Iterable[int] = 24,
    variables: Iterable[str] | None = None,
    output_path: str | Path | None = None,
    locations: int | Iterable[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    city_list = _location_names(cities, locations)
    variable_list = list(DEFAULT_VARIABLES if variables is None else variables)
    lead_list = [lead_hours] if isinstance(lead_hours, int) else list(lead_hours)
    if not lead_list:
        raise ValueError("At least one lead time is required")
    if any(not isinstance(lead, int) or lead < 0 for lead in lead_list):
        raise ValueError("lead_hours must contain non-negative integers")
    unknown_cities = [city for city in city_list if city not in LOCATION_METADATA]
    if unknown_cities:
        raise ValueError(f"Unknown city '{unknown_cities[0]}'")
    unknown_variables = [variable for variable in variable_list if variable not in VARIABLE_FIELDS]
    if unknown_variables:
        raise ValueError(f"Unsupported variable '{unknown_variables[0]}'")

    init_times = list(dict.fromkeys(_utc(value, "dates") for value in dates))
    now = datetime.now(timezone.utc)
    if any(init_time > now for init_time in init_times):
        raise ValueError("historical dates cannot be in the future")

    existing = load_historical_records(output_path) if resume and output_path is not None and Path(output_path).is_file() else {}
    records = _deduplicate_records(existing.get("records", []))
    skipped = list(existing.get("skipped", []))
    missing_gfs = list(existing.get("missing_gfs", []))
    missing_gefs = list(existing.get("missing_gefs", []))
    gfs_source_failures = list(existing.get("gfs_source_failures", []))
    gefs_archive_unavailable = list(existing.get("gefs_archive_unavailable", []))
    completed_keys = {_record_key(record) for record in records}

    valid_pairs = [
        (init_time, lead)
        for init_time in init_times
        for lead in lead_list
        if init_time + timedelta(hours=lead) <= now
    ]
    requested_keys = {
        (init_time, city, lead, variable)
        for init_time, lead in valid_pairs
        for city in city_list
        for variable in variable_list
    }
    completed_count = len(completed_keys & requested_keys)
    pending_pairs = [
        (init_time, lead)
        for init_time, lead in valid_pairs
        if not all(
            (init_time, city, lead, variable) in completed_keys
            for city in city_list
            for variable in variable_list
        )
    ]
    total_combinations = len(requested_keys)
    print(f"completed combinations: {completed_count} / {total_combinations}")
    print(f"remaining combinations: {total_combinations - completed_count}")

    observation_cache: dict[str, Mapping[str, Any] | None] = {}
    valid_times = [init_time + timedelta(hours=lead) for init_time, lead in pending_pairs]
    if valid_times:
        start_date = min(valid_times).date()
        end_date = max(valid_times).date()
        for city in city_list:
            location = LOCATION_METADATA[city]
            try:
                observation_cache[city] = get_historical_observations(
                    location["latitude"], location["longitude"], start_date, end_date
                )
            except Exception as exc:
                observation_cache[city] = None
                for init_time, lead in pending_pairs:
                    skipped.append({
                        "city": city,
                        "forecast_init_time": init_time,
                        "valid_time": init_time + timedelta(hours=lead),
                        "lead_hours": lead,
                        "reason": _reason(exc),
                    })

    def current_result() -> dict[str, Any]:
        return {
            "records": records,
            "skipped": skipped,
            "missing_gfs": missing_gfs,
            "missing_gefs": missing_gefs,
            "gfs_source_failures": gfs_source_failures,
            "gefs_archive_unavailable": gefs_archive_unavailable,
        }

    for init_time, lead in pending_pairs:
        print(f"Processing {init_time:%Y-%m-%d %H:%M} UTC F{lead:03d}")
        gfs_files = None
        try:
            gfs_files = download_gfs_subsets(init_time, lead)
        except GFSDownloadError as exc:
            gfs_source_failures.extend(exc.source_failures)
            missing_gfs.append({"forecast_init_time": init_time, "lead_hours": lead, "reason": _reason(exc), "missing_subsets": exc.missing_subsets})
            print(f"  GFS unavailable: {_reason(exc)}")
        except Exception as exc:
            missing_gfs.append({"forecast_init_time": init_time, "lead_hours": lead, "reason": _reason(exc)})
            print(f"  GFS unavailable: {_reason(exc)}")
        else:
            gfs_source_failures.extend(getattr(gfs_files, "source_failures", []))
            missing_subsets = getattr(gfs_files, "missing_subsets", [])
            if missing_subsets:
                missing_gfs.append({"forecast_init_time": init_time, "lead_hours": lead, "reason": "Some GFS subsets were unavailable", "missing_subsets": missing_subsets})

        gefs_files = None
        try:
            gefs_files = download_gefs_subsets(init_time, lead)
        except Exception as exc:
            missing_gefs.append({"forecast_init_time": init_time, "lead_hours": lead, "reason": _reason(exc)})
            print(f"  GEFS unavailable: {_reason(exc)}")
        else:
            if gefs_files is None:
                archive_item = {"forecast_init_time": init_time, "lead_hours": lead, "reason": "GEFS archive or index unavailable"}
                gefs_archive_unavailable.append(archive_item)
                missing_gefs.append(archive_item)

        valid_time = init_time + timedelta(hours=lead)
        for city in city_list:
            location = LOCATION_METADATA[city]
            gfs = None
            gefs = None
            if gfs_files is not None:
                try:
                    gfs = extract_gfs_point(gfs_files, location["latitude"], location["longitude"], lead)
                except Exception:
                    pass
            if gefs_files is not None:
                try:
                    gefs = extract_gefs_point(gefs_files, location["latitude"], location["longitude"], lead)
                except Exception:
                    pass
            observation = _observation_at(observation_cache.get(city) or {}, valid_time)
            if observation is None:
                if observation_cache.get(city) is not None:
                    skipped.append({"city": city, "forecast_init_time": init_time, "valid_time": valid_time, "lead_hours": lead, "reason": "no observation at exact valid_time"})
                continue
            for variable in variable_list:
                field = VARIABLE_FIELDS[variable]
                if observation.get(field) is None:
                    skipped.append({"city": city, "variable": variable, "forecast_init_time": init_time, "valid_time": valid_time, "lead_hours": lead, "reason": f"observation field '{field}' is unavailable"})
                    continue
                record = build_verification_record(init_time, valid_time, city, lead, variable, {"gfs": gfs, "gefs": gefs}, observation)
                key = _record_key(record)
                if key not in completed_keys:
                    records.append(record)
                    completed_keys.add(key)
                    if key in requested_keys:
                        completed_count += 1

        _write_history(output_path, current_result())
        print(f"  completed combinations: {completed_count} / {total_combinations}")
        print(f"  remaining combinations: {total_combinations - completed_count}")
        print(f"  missing GFS: {len(missing_gfs)}")
        print(f"  missing GEFS: {len(missing_gefs)}")

    result = current_result()
    _write_history(output_path, result)
    return result


def load_historical_records(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for collection in (payload.get("records", []), payload.get("skipped", [])):
        for item in collection:
            for key in ("forecast_init_time", "valid_time"):
                if key in item:
                    item[key] = _utc(item[key], key)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical W-CAST verification records")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--locations", type=int, choices=(4, 20), default=20)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = build_historical_records(
        historical_initialization_dates(args.days),
        locations=args.locations,
        lead_hours=DEFAULT_LEAD_HOURS,
        output_path=args.output,
        resume=args.resume,
    )
    _print_summary(result)


if __name__ == "__main__":
    main()