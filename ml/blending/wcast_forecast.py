from ml.blending.blender import blend_forecasts
from ml.evaluation.historical_weights import build_historical_weights


def generate_wcast_forecast(
    city,
    variable,
    lead_hours,
    gfs_value,
    gefs_value,
    history_path="data/processed/history_2d.json",
):
    weights_data = build_historical_weights(history_path)

    try:
        weights = weights_data[city][variable][str(lead_hours)]["weights"]
    except KeyError:
        weights = {"gfs": 0.5, "gefs": 0.5}

    forecasts = {
        "gfs": gfs_value,
        "gefs": gefs_value,
    }

    forecast = blend_forecasts(forecasts, weights)

    return {
        "forecast": float(forecast),
        "weights": weights,
        "gfs": gfs_value,
        "gefs": gefs_value,
        "city": city,
        "variable": variable,
        "lead_hours": lead_hours,
    }