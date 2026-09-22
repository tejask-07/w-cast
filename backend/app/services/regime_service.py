def classify_regime(lat: float, lon: float, lead_hours: int) -> str:
    """Return a mock regime label that will be replaced by ML later."""
    _ = (lat, lon, lead_hours)
    if lat > 20 and lon > 70:
        return "wet"
    if lat > 28:
        return "dry"
    return "moderate"
