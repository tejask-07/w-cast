"""Run one real W-CAST forecast workflow from any working directory."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "backend"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from app.services.forecast_service import generate_forecast
from app.services.spatial_forecast_service import generate_spatial_forecast_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lead-hours", type=int, choices=(24, 48, 72), required=True)
    parser.add_argument("--variable", choices=("temperature", "rainfall", "wind_speed"), required=True)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "interim" / "wcast_run.json")
    args = parser.parse_args()

    forecast = generate_forecast(args.lat, args.lon, args.lead_hours, args.variable)
    spatial_map = generate_spatial_forecast_map(
        args.lat, args.lon, args.lead_hours, args.variable
    )
    payload = {"forecast": forecast, "map": spatial_map}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "source": "real_gfs_gefs"}))


if __name__ == "__main__":
    main()
