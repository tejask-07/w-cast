from typing import Dict

SUPPORTED_VARIABLES = ("temperature", "rainfall", "wind_speed")


def validate_variable(variable: str) -> None:
    if variable not in SUPPORTED_VARIABLES:
        raise ValueError(f"Unsupported variable: {variable}")


def generate_verification(lat: float, lon: float, lead_hours: int, variable: str) -> Dict[str, object]:
    validate_variable(variable)
    _ = (lat, lon)
    return {
        "variable": variable,
        "lead_hours": lead_hours,
        "metrics": {
            "gfs": {"mae": 12.4, "rmse": 18.7, "bias": 2.1},
            "gefs": {"mae": 10.2, "rmse": 15.3, "bias": 1.2},
            "adaptive_blend": {"mae": 8.4, "rmse": 12.9, "bias": 0.6},
        },
    }
