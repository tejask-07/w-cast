from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def normalize_gfs_grid(
    grids: dict[str, Any],
) -> dict[str, xr.DataArray]:
    required = {
        "temperature",
        "wind_u",
        "wind_v",
        "precipitation",
    }

    missing = required - set(grids)

    if missing:
        raise ValueError(
            f"Missing GFS fields: {sorted(missing)}"
        )

    for name in required:
        if grids[name] is None:
            raise ValueError(
                f"GFS field is unavailable: {name}"
            )

    temperature_c = grids["temperature"] - 273.15

    wind_speed_ms = xr.apply_ufunc(
        np.hypot,
        grids["wind_u"],
        grids["wind_v"],
    )

    precipitation_mm = grids["precipitation"]

    return {
        "temperature": temperature_c,
        "wind_speed": wind_speed_ms,
        "precipitation": precipitation_mm,
    }
