from typing import Dict


def get_model_weights(lat: float, lon: float, lead_hours: int) -> Dict[str, float]:
    """Return deterministic mock ensemble weights for the demo region."""
    _ = (lat, lon, lead_hours)
    return {
        "gfs": 0.25,
        "gefs": 0.65,
        "baseline": 0.10,
    }
