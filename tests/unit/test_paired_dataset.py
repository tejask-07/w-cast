from datetime import datetime, timedelta, timezone

import pytest

from ml.evaluation import paired_dataset
from ml.preprocessing.download import GFSDownloadResult


INIT = datetime(2026, 9, 10, 6, tzinfo=timezone.utc)


def _record_kwargs(**overrides):
    values = {
        "location_name": "Mumbai",
        "latitude": 19.076,
        "longitude": 72.8777,
        "region": "west_coast",
        "forecast_cycle": INIT,
        "variable": "temperature",
        "lead_hours": 24,
        "gfs_value": 22.0,
        "gefs_value": 24.0,
        "observation_value": 20.0,
        "observation_valid_time": INIT + timedelta(hours=24),
    }
    values.update(overrides)
    return values


def test_paired_record_schema_errors_and_absolute_errors():
    record = paired_dataset.build_paired_record(**_record_kwargs())

    assert set(record) == {
        "location_name", "latitude", "longitude", "region", "forecast_date",
        "forecast_cycle", "valid_time", "variable", "lead_hours", "gfs_value",
        "gefs_value", "observation_value", "gfs_error", "gefs_error",
        "gfs_abs_error", "gefs_abs_error", "gfs_source", "gfs_resolution_degrees",
        "gefs_source", "gefs_resolution_degrees",
    }
    assert record["forecast_date"] == "2026-09-10"
    assert record["valid_time"] == "2026-09-11T06:00:00Z"
    assert record["gfs_error"] == 2.0
    assert record["gefs_error"] == 4.0
    assert record["gfs_abs_error"] == 2.0
    assert record["gefs_abs_error"] == 4.0


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"gfs_value": None}, "gfs_missing"),
        ({"gefs_value": None}, "gefs_missing"),
        ({"observation_value": None}, "observation_missing"),
        ({"observation_valid_time": INIT + timedelta(hours=25)}, "timestamp_mismatch"),
        ({"gfs_value": float("nan")}, "gfs_non_finite"),
        ({"gefs_value": float("inf")}, "gefs_non_finite"),
        ({"observation_value": float("nan")}, "observation_non_finite"),
        ({"latitude": float("nan")}, "invalid_coordinates"),
        ({"longitude": 181}, "invalid_coordinates"),
    ],
)
def test_invalid_candidate_is_rejected(overrides, reason):
    with pytest.raises(paired_dataset.PairedRecordRejected, match=reason):
        paired_dataset.build_paired_record(**_record_kwargs(**overrides))


@pytest.mark.parametrize("variable", ["temperature", "rainfall", "wind_speed"])
@pytest.mark.parametrize("lead", [24, 48, 72])
def test_multiple_variables_and_supported_leads(variable, lead):
    valid_time = INIT + timedelta(hours=lead)
    record = paired_dataset.build_paired_record(
        **_record_kwargs(
            variable=variable,
            lead_hours=lead,
            observation_valid_time=valid_time,
        )
    )
    assert record["variable"] == variable
    assert record["lead_hours"] == lead
    assert record["valid_time"] == valid_time.isoformat().replace("+00:00", "Z")


def test_coverage_summary_counts_location_variable_and_lead():
    records = [
        {"location_name": "Mumbai", "variable": "temperature", "lead_hours": 24, "forecast_date": "2026-09-10"},
        {"location_name": "Mumbai", "variable": "temperature", "lead_hours": 24, "forecast_date": "2026-09-11"},
        {"location_name": "Pune", "variable": "rainfall", "lead_hours": 72, "forecast_date": "2026-09-10"},
    ]
    report = paired_dataset.summarize_dataset(
        records,
        [],
        4,
        ["Mumbai", "Pune"],
        ["temperature", "rainfall"],
        [24, 72],
        [INIT],
    )
    assert report["candidate_records"] == 4
    assert report["total_candidates"] == 4
    assert report["valid_paired_records"] == 3
    assert report["valid_records"] == 3
    assert report["records_by_forecast_date"] == {"2026-09-10": 2, "2026-09-11": 1}
    assert report["coverage"]["Mumbai"]["temperature"] == {"24": 2, "72": 0}
    assert report["coverage"]["Pune"]["rainfall"] == {"24": 0, "72": 1}


def test_build_dataset_pairs_real_source_boundaries_and_writes_jsonl(monkeypatch, tmp_path):
    cycles = [INIT, INIT + timedelta(days=1)]
    leads = [24, 48, 72]
    valid_times = [cycle + timedelta(hours=lead) for cycle in cycles for lead in leads]

    class FakeAdapter:
        def get_forecast(self, forecast_date, forecast_cycle, lead_hours, variable, latitude, longitude):
            init = datetime(
                forecast_date.year,
                forecast_date.month,
                forecast_date.day,
                forecast_cycle,
                tzinfo=timezone.utc,
            )
            return {
                "forecast_value": {
                    "temperature": 22.0,
                    "rainfall": 2.0,
                    "wind_speed": 4.0,
                }[variable],
                "source": "noaa-gefs-operational-aws",
                "gefs_source": "GEFS operational archive (NOAA AWS Open Data)",
                "initialization_time": init,
                "valid_time": init + timedelta(hours=lead_hours),
                "spatial_resolution_degrees": 0.25,
            }

    def download_gfs(cycle, lead):
        root = tmp_path / cycle.strftime("%Y%m%d")
        fields = {
            name: root / f"subset__gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}"
            for name in ("temperature", "wind_u", "wind_v", "precipitation")
        }
        return GFSDownloadResult(fields, [], [], {name: "aws" for name in fields})

    monkeypatch.setattr(paired_dataset, "download_gfs_subsets", download_gfs)
    monkeypatch.setattr(paired_dataset, "HistoricalGEFSAdapter", FakeAdapter)
    monkeypatch.setattr(
        paired_dataset,
        "extract_gfs_point",
        lambda *_args: {"temperature_C": 21.0, "precipitation_mm": 1.0, "wind_speed_ms": 3.0},
    )
    monkeypatch.setattr(
        paired_dataset,
        "get_historical_observations",
        lambda lat, lon, start, end: {
            "hourly": valid_times,
            "temperature_C": [20.0] * len(valid_times),
            "precipitation_mm": [0.5] * len(valid_times),
            "wind_speed_ms": [2.0] * len(valid_times),
        },
    )
    output = tmp_path / "paired.jsonl"

    result = paired_dataset.build_paired_dataset(
        cycles,
        locations=["Mumbai"],
        output_path=output,
    )

    assert result["report"]["candidate_records"] == 18
    assert result["report"]["valid_paired_records"] == 18
    assert result["report"]["rejected_records"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 18
    assert output.with_suffix(".report.json").is_file()
    assert result["records"][0]["gfs_source"].endswith("(aws)")
    assert result["records"][0]["gfs_resolution_degrees"] == 0.25


def test_partial_date_failure_continues_and_rejection_is_counted(monkeypatch, tmp_path):
    cycles = [INIT, INIT + timedelta(days=1)]
    valid_times = [cycle + timedelta(hours=24) for cycle in cycles]

    class FakeAdapter:
        def get_forecast(self, forecast_date, forecast_cycle, lead_hours, variable, latitude, longitude):
            init = datetime(forecast_date.year, forecast_date.month, forecast_date.day, forecast_cycle, tzinfo=timezone.utc)
            return {
                "forecast_value": 25.0,
                "gefs_source": "GEFS operational archive (NOAA AWS Open Data)",
                "initialization_time": init,
                "valid_time": init + timedelta(hours=lead_hours),
                "spatial_resolution_degrees": 0.25,
            }

    def download_gfs(cycle, lead):
        if cycle == cycles[0]:
            raise RuntimeError("fixture GFS cycle missing")
        root = tmp_path / cycle.strftime("%Y%m%d")
        paths = {name: root / f"subset__gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}" for name in ("temperature", "wind_u", "wind_v", "precipitation")}
        return GFSDownloadResult(paths, [], [], {name: "nomads" for name in paths})

    monkeypatch.setattr(paired_dataset, "download_gfs_subsets", download_gfs)
    monkeypatch.setattr(paired_dataset, "HistoricalGEFSAdapter", FakeAdapter)
    monkeypatch.setattr(paired_dataset, "extract_gfs_point", lambda *_args: {"temperature_C": 23.0})
    monkeypatch.setattr(
        paired_dataset,
        "get_historical_observations",
        lambda *_args: {"hourly": valid_times, "temperature_C": [22.0, 22.0]},
    )

    result = paired_dataset.build_paired_dataset(
        cycles,
        locations=["Mumbai"],
        variables=["temperature"],
        leads=[24],
        output_path=tmp_path / "partial.jsonl",
    )

    assert result["report"]["candidate_records"] == 2
    assert result["report"]["valid_paired_records"] == 1
    assert result["report"]["rejected_records"] == 1
    assert any(reason.startswith("gfs_unavailable:") for reason in result["report"]["rejection_reasons"])


def test_duplicate_cycles_are_deduplicated(monkeypatch, tmp_path):
    valid_time = INIT + timedelta(hours=24)

    class FakeAdapter:
        def get_forecast(self, forecast_date, forecast_cycle, lead_hours, variable, latitude, longitude):
            init = datetime(forecast_date.year, forecast_date.month, forecast_date.day, forecast_cycle, tzinfo=timezone.utc)
            return {
                "forecast_value": 25.0,
                "gefs_source": "GEFS operational archive (NOAA AWS Open Data)",
                "initialization_time": init,
                "valid_time": init + timedelta(hours=lead_hours),
            }

    paths = {
        name: tmp_path / INIT.strftime("%Y%m%d") / f"subset__gfs.t{INIT:%H}z.pgrb2.0p25.f024"
        for name in ("temperature", "wind_u", "wind_v", "precipitation")
    }
    monkeypatch.setattr(
        paired_dataset,
        "download_gfs_subsets",
        lambda *_args: GFSDownloadResult(paths, [], [], {name: "aws" for name in paths}),
    )
    monkeypatch.setattr(paired_dataset, "HistoricalGEFSAdapter", FakeAdapter)
    monkeypatch.setattr(paired_dataset, "extract_gfs_point", lambda *_args: {"temperature_C": 23.0})
    monkeypatch.setattr(
        paired_dataset,
        "get_historical_observations",
        lambda *_args: {"hourly": [valid_time], "temperature_C": [22.0]},
    )

    result = paired_dataset.build_paired_dataset(
        [INIT, INIT],
        locations=["Mumbai"],
        variables=["temperature"],
        leads=[24],
        output_path=tmp_path / "dedup.jsonl",
    )

    assert result["report"]["candidate_records"] == 1
    assert len(result["records"]) == 1