"""Audit historical GEFS index and ensemble-mean field availability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.preprocessing.gefs import GEFS_MEMBER, GEFS_MODEL, GEFS_PRODUCT, GEFS_PRIORITY

LEADS = (24, 48, 72)
REQUIRED_FIELDS = {
    "temperature": ("TMP", "2 m above ground"),
    "wind_u": ("UGRD", "10 m above ground"),
    "wind_v": ("VGRD", "10 m above ground"),
    "precipitation": ("APCP", "surface"),
}


def _classify_exception(error: BaseException) -> str:
    message = str(error).lower()
    error_name = type(error).__name__.lower()
    if any(token in message for token in ("no index file", "index file", "did not find", "not found", "index is unavailable")):
        return "archive_or_index_unavailable"
    if any(token in message for token in ("timeout", "timed out", "connection", "dns", "http", "url")):
        return "network_error"
    if "invalid" in message or "unsupported" in message or "valueerror" in error_name:
        return "invalid_request"
    if "column" in message or "inventory" in message or "keyerror" in error_name:
        return "malformed_inventory"
    return "unexpected_error"


def _inventory_status(inventory: Any) -> tuple[str, dict[str, str], list[str]]:
    required_columns = {"variable", "level", "search_this"}
    columns = set(getattr(inventory, "columns", ()))
    if not required_columns.issubset(columns):
        missing = sorted(required_columns - columns)
        raise ValueError(f"Inventory missing columns: {missing}")

    fields: dict[str, str] = {}
    missing: list[str] = []
    optional_missing: list[str] = []
    for name, (variable, level) in REQUIRED_FIELDS.items():
        matches = inventory[
            (inventory["variable"] == variable)
            & (inventory["level"] == level)
            & inventory["search_this"].astype(str).str.contains("ens mean", na=False)
        ]
        if len(matches) == 1:
            fields[name] = str(matches.iloc[0]["search_this"])
        elif name == "precipitation" and len(matches) == 0:
            optional_missing.append(name)
        else:
            missing.append(name)

    if missing:
        return "partial", fields, missing + optional_missing
    if optional_missing:
        return "partial", fields, [f"optional_missing:{name}" for name in optional_missing]
    return "available", fields, []


def _audit_candidate(init_time: datetime) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date": init_time.date().isoformat(),
        "cycle": init_time.strftime("%H"),
        "leads": {},
    }
    try:
        from herbie import Herbie
    except Exception as exc:
        category = _classify_exception(exc)
        for lead in LEADS:
            result["leads"][str(lead)] = {
                "status": category,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return result

    for lead in LEADS:
        try:
            forecast = Herbie(
                date=init_time.replace(tzinfo=None),
                model=GEFS_MODEL,
                product=GEFS_PRODUCT,
                member=GEFS_MEMBER,
                fxx=lead,
                priority=GEFS_PRIORITY,
            )
            if getattr(forecast, "idx", True) in (None, False):
                raise RuntimeError("GEFS index is unavailable")
            inventory = forecast.inventory()
            status, fields, missing = _inventory_status(inventory)
            result["leads"][str(lead)] = {
                "status": status,
                "missing_fields": missing,
                "fields": fields,
            }
        except Exception as exc:
            result["leads"][str(lead)] = {
                "status": _classify_exception(exc),
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return result


def _candidate_dates(days: int, end_date: date | None) -> list[datetime]:
    if days <= 0:
        raise ValueError("days must be positive")
    last = end_date or (datetime.now(timezone.utc).date() - timedelta(days=1))
    dates = [last - timedelta(days=offset) for offset in range(days - 1, -1, -1)]
    return [
        datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).replace(hour=cycle)
        for day in dates
        for cycle in (0, 6, 12, 18)
    ]


def _print_report(results: list[dict[str, Any]]) -> None:
    counts = Counter(
        lead_result["status"]
        for result in results
        for lead_result in result["leads"].values()
    )
    print("DATE       CYCLE F24                         F48                         F72")
    print("---------- ----- --------------------------- --------------------------- ---------------------------")
    for result in results:
        statuses = [result["leads"][str(lead)]["status"] for lead in LEADS]
        print(f"{result['date']} {result['cycle']:>5} " + " ".join(f"{status:<27}" for status in statuses))
    usable = sum(all(result["leads"][str(lead)]["status"] == "available" for lead in LEADS) for result in results)
    partial = sum(
        any(result["leads"][str(lead)]["status"] == "available" for lead in LEADS)
        and not all(result["leads"][str(lead)]["status"] == "available" for lead in LEADS)
        for result in results
    )
    unavailable = len(results) - usable - partial
    print("\nSUMMARY")
    print(f"candidates: {len(results)}")
    print(f"usable dates/cycles: {usable}")
    print(f"partially usable dates/cycles: {partial}")
    print(f"unavailable dates/cycles: {unavailable}")
    print("lead status counts:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print("unavailable/partial reasons:")
    for result in results:
        for lead, lead_result in result["leads"].items():
            if lead_result["status"] in {"available"}:
                continue
            detail = lead_result.get("reason") or ", ".join(lead_result.get("missing_fields", []))
            print(f"  {result['date']} cycle {result['cycle']} F{lead}: {lead_result['status']}: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit GEFS historical index availability without downloading GRIB files")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    results = [_audit_candidate(candidate) for candidate in _candidate_dates(args.days, args.end_date)]
    _print_report(results)
    if args.json_output is not None:
        output = args.json_output if args.json_output.is_absolute() else REPO_ROOT / args.json_output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"json_output: {output}")


if __name__ == "__main__":
    main()
