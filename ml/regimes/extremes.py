"""Simple extreme-weather indicators for the W-CAST MVP."""


def detect_extremes(
    precipitation_mm: float,
    temperature_c: float,
    wind_speed_ms: float,
) -> dict[str, bool]:
    """Return rule-based extreme weather indicators."""

    return {
        "heavy_rain": precipitation_mm >= 50.0,
        "heat_wave": temperature_c >= 40.0,
        "high_wind": wind_speed_ms >= 17.0,
    }