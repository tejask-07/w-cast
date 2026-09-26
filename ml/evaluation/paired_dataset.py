"""Build a strict paired GFS/GEFS/observation dataset for offline verification."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ml.preprocessing.align import LOCATION_METADATA
from ml.preprocessing.download import download_gfs_subsets
from ml.preprocessing.gfs_extract import extract_gfs_point
from ml.preprocessing.historical_gefs import (
    GEFS_RESOLUTION_DEGREES,
    GEFS_SOURCE_DESCRIPTION,
    HistoricalGEFSAdapter,
    HistoricalGEFSUnavailable,
    validate_pair_timestamps,
)
from ml.preprocessing.observations import get_historical_observations

INITIAL_LOCATIONS = ("Mumbai", "Delhi", "Pune", "Kolkata", "Chennai")
VARIABLE_FIELDS = {
    "temperature": "temperature_C",
    "rainfall": "precipitation_mm",
    "wind_speed": "wind_speed_ms",
}
LEAD_HOURS = (24, 48, 72)
DEFAULT_DAYS = 10
DEFAULT_CYCLE_HOUR = 6
DEFAULT_OUTPUT = Path("data/processed/historical_paired_validation.jsonl")
GFS_RESOLUTION_DEGREES = 0.25
GFS_SOURCE_DESCRIPTION = "NOAA GFS operational pgrb2.0p25 via Herbie"
GFS_REQUIRED_FIELDS = {
    "temperature": ("temperature",),
    "rainfall": ("precipitation",),
    "wind_speed": ("wind_u", "wind_v"),
}


class PairedRecordRejected(ValueError):
    """A candidate cannot form a fully observed, timestamp-matched pair."""


def _as_utc(value: datetime | str, name: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite_value(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise PairedRecordRejected(f"{name}_missing")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise PairedRecordRejected(f"{name}_invalid") from exc
    if not math.isfinite(numeric):
        raise PairedRecordRejected(f"{name}_non_finite")
    return numeric


def build_paired_record(
    *,
    location_name: str,
    latitude: float,
    longitude: float,
    region: str,
    forecast_cycle: datetime | str,
    variable: str,
    lead_hours: int,
    gfs_value: Any,
    gefs_value: Any,
    observation_value: Any,
    observation_valid_time: datetime | str | None,
    gefs_source: str = GEFS_SOURCE_DESCRIPTION,
    gfs_source: str = GFS_SOURCE_DESCRIPTION,
) -> dict[str, Any]:
    """Validate one candidate and return the strict paired-record schema."""
    if variable not in VARIABLE_FIELDS:
        raise PairedRecordRejected("unsupported_variable")
    if lead_hours not in LEAD_HOURS:
        raise PairedRecordRejected("unsupported_lead_hours")
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise PairedRecordRejected("invalid_coordinates") from exc
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise PairedRecordRejected("invalid_coordinates")

    cycle = _as_utc(forecast_cycle, "forecast_cycle")
    valid_time = cycle + timedelta(hours=lead_hours)
    if observation_valid_time is None:
        raise PairedRecordRejected("observation_missing")
    observed_time = _as_utc(observation_valid_time, "observation_valid_time")
    if observed_time != valid_time:
        raise PairedRecordRejected("timestamp_mismatch")

    gfs = _finite_value(gfs_value, "gfs")
    gefs = _finite_value(gefs_value, "gefs")
    observation = _finite_value(observation_value, "observation")
    gfs_error = gfs - observation
    gefs_error = gefs - observation

    return {
        "location_name": location_name,
        "latitude": lat,
        "longitude": lon,
        "region": region,
        "forecast_date": cycle.date().isoformat(),
        "forecast_cycle": cycle.isoformat().replace("+00:00", "Z"),
        "valid_time": valid_time.isoformat().replace("+00:00", "Z"),
        "variable": variable,
        "lead_hours": lead_hours,
        "gfs_value": gfs,
        "gfs_source": gfs_source,
        "gfs_resolution_degrees": GFS_RESOLUTION_DEGREES,
        "gefs_value": gefs,
        "gefs_source": gefs_source,
        "gefs_resolution_degrees": GEFS_RESOLUTION_DEGREES,
        "observation_value": observation,
        "gfs_error": gfs_error,
        "gefs_error": gefs_error,
        "gfs_abs_error": abs(gfs_error),
        "gefs_abs_error": abs(gefs_error),
    }


def historical_forecast_cycles(
    days: int = DEFAULT_DAYS,
    cycle_hour: int = DEFAULT_CYCLE_HOUR,
    end_date: date | None = None,
) -> list[datetime]:
    """Return multiple completed, recent UTC initialization cycles."""
    if days <= 0:
        raise ValueError("days must be positive")
    if cycle_hour not in range(24):
        raise ValueError("cycle_hour must be between 0 and 23")
    latest_date = end_date or (datetime.now(timezone.utc).date() - timedelta(days=5))
    return [
        datetime.combine(latest_date - timedelta(days=offset), time(cycle_hour), tzinfo=timezone.utc)
        for offset in range(days - 1, -1, -1)
    ]


def _gfs_metadata_for_variable(
    gfs_files: Mapping[str, Any],
    variable: str,
    cycle: datetime,
    lead_hours: int,
) -> tuple[datetime, datetime, str]:
    expected = _as_utc(cycle, "forecast_cycle")
    fields = GFS_REQUIRED_FIELDS[variable]
    sources = getattr(gfs_files, "subset_sources", {})
    actual_sources: set[str] = set()
    for field in fields:
        if field not in gfs_files:
            raise PairedRecordRejected(f"gfs_{field}_unavailable")
        path = Path(gfs_files[field])
        match = re.search(r"gfs\.t(?P<cycle>\d{2})z\.pgrb2\.0p25\.f(?P<lead>\d{3})", path.name)
        date_match = next((part for part in path.parts if re.fullmatch(r"\d{8}", part)), None)
        if match is None or date_match is None:
            raise PairedRecordRejected("gfs_cycle_metadata_unavailable")
        actual_init = datetime.strptime(date_match + match.group("cycle"), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        actual_lead = int(match.group("lead"))
        if actual_init != expected:
            raise PairedRecordRejected("gfs_cycle_mismatch")
        if actual_lead != lead_hours:
            raise PairedRecordRejected("gfs_lead_mismatch")
        source = sources.get(field)
        if source:
            actual_sources.add(str(source))
    source_name = f"{GFS_SOURCE_DESCRIPTION} ({'+'.join(sorted(actual_sources))})" if actual_sources else GFS_SOURCE_DESCRIPTION
    return expected, expected + timedelta(hours=lead_hours), source_name


def _observation_index(payload: Mapping[str, Any]) -> dict[datetime, dict[str, Any]]:
    indexed: dict[datetime, dict[str, Any]] = {}
    timestamps = payload.get("hourly", [])
    for index, timestamp in enumerate(timestamps):
        normalized = _as_utc(timestamp, "observation timestamp")
        indexed[normalized] = {
            field: payload.get(field, [None] * len(timestamps))[index]
            for field in VARIABLE_FIELDS.values()
        }
    return indexed


def _reject(
    rejected: list[dict[str, Any]],
    cycle: datetime,
    location: str,
    variable: str,
    lead: int,
    reason: str,
) -> None:
    rejected.append({
        "location_name": location,
        "forecast_cycle": cycle.isoformat().replace("+00:00", "Z"),
        "variable": variable,
        "lead_hours": lead,
        "reason": reason,
    })


def summarize_dataset(
    records: list[Mapping[str, Any]],
    rejected: list[Mapping[str, Any]],
    candidate_records: int,
    locations: Iterable[str],
    variables: Iterable[str],
    leads: Iterable[int],
    cycles: Iterable[datetime],
) -> dict[str, Any]:
    by_location = Counter(record["location_name"] for record in records)
    by_variable = Counter(record["variable"] for record in records)
    by_lead = Counter(record["lead_hours"] for record in records)
    by_date = Counter(record["forecast_date"] for record in records)
    coverage = {
        location: {
            variable: {
                str(lead): sum(
                    record["location_name"] == location
                    and record["variable"] == variable
                    and record["lead_hours"] == lead
                    for record in records
                )
                for lead in leads
            }
            for variable in variables
        }
        for location in locations
    }
    return {
        "total_candidates": candidate_records,
        "candidate_records": candidate_records,
        "valid_records": len(records),
        "valid_paired_records": len(records),
        "rejected_records": len(rejected),
        "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
        "records_by_location": dict(by_location),
        "records_by_variable": dict(by_variable),
        "records_by_lead_hours": {str(key): value for key, value in by_lead.items()},
        "records_by_forecast_date": dict(by_date),
        "coverage": coverage,
        "forecast_dates_attempted": [cycle.date().isoformat() for cycle in cycles],
        "gfs_sources": dict(Counter(record.get("gfs_source", "unknown") for record in records)),
        "gefs_sources": dict(Counter(record.get("gefs_source", "unknown") for record in records)),
    }


def build_paired_dataset(
    cycles: Iterable[datetime | str],
    locations: Iterable[str] = INITIAL_LOCATIONS,
    variables: Iterable[str] = VARIABLE_FIELDS,
    leads: Iterable[int] = LEAD_HOURS,
    output_path: str | Path | None = DEFAULT_OUTPUT,
    historical_gefs_adapter: HistoricalGEFSAdapter | None = None,
) -> dict[str, Any]:
    """Download and pair real forecasts with exact-valid-time observations."""
    cycle_list = list(dict.fromkeys(_as_utc(cycle, "forecast_cycle") for cycle in cycles))
    location_list = list(dict.fromkeys(locations))
    variable_list = list(dict.fromkeys(variables))
    lead_list = list(dict.fromkeys(leads))
    if not cycle_list:
        raise ValueError("At least one forecast cycle is required")
    unknown = [location for location in location_list if location not in LOCATION_METADATA]
    if unknown:
        raise ValueError(f"Unknown location '{unknown[0]}'")
    if any(variable not in VARIABLE_FIELDS for variable in variable_list):
        raise ValueError("Unsupported variable")
    if any(lead not in LEAD_HOURS for lead in lead_list):
        raise ValueError("Unsupported lead_hours")

    candidate_records = len(cycle_list) * len(location_list) * len(variable_list) * len(lead_list)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    gefs_adapter = historical_gefs_adapter or HistoricalGEFSAdapter()
    observations: dict[str, dict[datetime, dict[str, Any]] | None] = {}
    observation_errors: dict[str, str] = {}
    valid_times = [cycle + timedelta(hours=lead) for cycle in cycle_list for lead in lead_list]
    observation_start = min(valid_times).date()
    observation_end = max(valid_times).date()

    for name in location_list:
        location = LOCATION_METADATA[name]
        try:
            payload = get_historical_observations(
                location["latitude"],
                location["longitude"],
                observation_start,
                observation_end,
            )
            observations[name] = _observation_index(payload)
        except Exception as exc:
            observations[name] = None
            observation_errors[name] = f"observation_source_error: {type(exc).__name__}: {exc}"

    for cycle in cycle_list:
        for lead in lead_list:
            gfs_files: Mapping[str, Any] | None = None
            gfs_error: str | None = None
            try:
                gfs_files = download_gfs_subsets(cycle, lead)
            except Exception as exc:
                gfs_error = f"gfs_unavailable: {type(exc).__name__}: {exc}"

            for name in location_list:
                location = LOCATION_METADATA[name]
                gfs_point = None
                location_gfs_error = None
                if gfs_files is not None:
                    try:
                        gfs_point = extract_gfs_point(
                            gfs_files,
                            location["latitude"],
                            location["longitude"],
                            lead,
                        )
                    except Exception as exc:
                        location_gfs_error = f"gfs_extraction_error: {type(exc).__name__}: {exc}"
                else:
                    location_gfs_error = gfs_error

                for variable in variable_list:
                    field = VARIABLE_FIELDS[variable]
                    for candidate_lead in (lead,):
                        valid_time = cycle + timedelta(hours=candidate_lead)
                        if observations[name] is None:
                            _reject(
                                rejected,
                                cycle,
                                name,
                                variable,
                                candidate_lead,
                                observation_errors.get(name, "observation_unavailable"),
                            )
                            continue
                        observation = observations[name].get(valid_time)
                        if observation is None:
                            _reject(rejected, cycle, name, variable, candidate_lead, "observation_timestamp_unavailable")
                            continue
                        gfs_value = None if gfs_point is None else gfs_point.get(field)
                        try:
                            if gfs_files is None:
                                raise PairedRecordRejected(gfs_error or "gfs_unavailable")
                            gfs_init, gfs_valid, gfs_source = _gfs_metadata_for_variable(
                                gfs_files,
                                variable,
                                cycle,
                                candidate_lead,
                            )
                            gefs_result = gefs_adapter.get_forecast(
                                forecast_date=cycle.date(),
                                forecast_cycle=cycle.hour,
                                lead_hours=candidate_lead,
                                variable=variable,
                                latitude=location["latitude"],
                                longitude=location["longitude"],
                            )
                            validate_pair_timestamps(
                                gfs_init,
                                gefs_result["initialization_time"],
                                gfs_valid,
                                gefs_result["valid_time"],
                                candidate_lead,
                            )
                            records.append(build_paired_record(
                                location_name=name,
                                latitude=location["latitude"],
                                longitude=location["longitude"],
                                region=str(location["region"]),
                                forecast_cycle=cycle,
                                variable=variable,
                                lead_hours=candidate_lead,
                                gfs_value=gfs_value,
                                gefs_value=gefs_result["forecast_value"],
                                observation_value=observation.get(field),
                                observation_valid_time=valid_time,
                                gefs_source=gefs_result["gefs_source"],
                                gfs_source=gfs_source,
                            ))
                        except PairedRecordRejected as exc:
                            reason = location_gfs_error if str(exc).startswith("gfs_") and location_gfs_error else str(exc)
                            _reject(rejected, cycle, name, variable, candidate_lead, reason)
                        except HistoricalGEFSUnavailable as exc:
                            _reject(rejected, cycle, name, variable, candidate_lead, f"gefs_unavailable: {exc}")
                        except ValueError as exc:
                            message = str(exc)
                            reason = (
                                "cycle_mismatch"
                                if "initialization times do not match" in message
                                else "valid_time_mismatch"
                            )
                            _reject(rejected, cycle, name, variable, candidate_lead, f"{reason}: {message}")

    report = summarize_dataset(
        records,
        rejected,
        candidate_records,
        location_list,
        variable_list,
        lead_list,
        cycle_list,
    )
    report["locations_attempted"] = location_list
    report["variables_attempted"] = variable_list
    report["lead_hours_attempted"] = lead_list

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        report_path = output.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return {"records": records, "rejected": rejected, "report": report}


def print_report(report: Mapping[str, Any]) -> None:
    print(f"Total candidate records: {report['candidate_records']}")
    print(f"Valid paired records: {report['valid_paired_records']}")
    print(f"Rejected records: {report['rejected_records']}")
    print(f"Forecast dates attempted: {len(report['forecast_dates_attempted'])}")
    print("Rejection reasons:")
    for reason, count in sorted(report["rejection_reasons"].items()):
        print(f"  {reason}: {count}")
    for label, key in (
        ("Records by location", "records_by_location"),
        ("Records by variable", "records_by_variable"),
        ("Records by lead time", "records_by_lead_hours"),
        ("Records by forecast date", "records_by_forecast_date"),
    ):
        print(f"{label}:")
        for value, count in sorted(report[key].items(), key=lambda item: str(item[0])):
            print(f"  {value}: {count}")
    print("Coverage (location × variable × lead_hours):")
    for location, variables in report["coverage"].items():
        for variable, leads in variables.items():
            print(f"  {location} / {variable}: " + ", ".join(f"{lead}h={count}" for lead, count in leads.items()))
    print("GFS sources:")
    for source, count in sorted(report.get("gfs_sources", {}).items()):
        print(f"  {source}: {count}")
    print("GEFS sources:")
    for source, count in sorted(report.get("gefs_sources", {}).items()):
        print(f"  {source}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired historical GFS/GEFS verification JSONL")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--cycle-hour", type=int, default=DEFAULT_CYCLE_HOUR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cycles = historical_forecast_cycles(args.days, args.cycle_hour)
    print("Attempting forecast cycles:")
    for cycle in cycles:
        print(f"  {cycle.isoformat()}")
    result = build_paired_dataset(cycles, output_path=args.output)
    print_report(result["report"])
    print(f"JSONL dataset: {args.output}")
    print(f"Report: {args.output.with_suffix('.report.json')}")


if __name__ == "__main__":
    main()