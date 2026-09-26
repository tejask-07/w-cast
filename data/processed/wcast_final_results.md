# W-CAST Final Validation

## Dataset

- Source dataset: `history_audited_gefs_2026-09-20_to_2026-09-24_20locations_multilead.json`
- Records: 1740
- Paired GFS + GEFS records: 1740
- Locations: 20
- Variables: temperature, precipitation, wind_speed
- Lead times: 24, 48, 72
- Date range: 2026-09-21T06:00:00Z to 2026-09-25T18:00:00Z

## Methodology

- The benchmark uses the existing chronological leave-one-out inverse-MAE weighting logic already implemented in `ml.evaluation.benchmark.benchmark_records`.
- For each target record, only paired historical records before the current evaluation timestamp with the same variable and lead are used to estimate the W-CAST weight.
- The target observation is excluded from its own weight calculation, and future records do not influence earlier forecasts.
- Equal Weight is computed as 50/50 GFS + GEFS blend without changing the current methodology.
- Positive improvement is defined as baseline_error - wcast_error; negative values indicate W-CAST had higher error.

## Forecast Accuracy

| Variable | Lead | GFS MAE | GEFS MAE | Equal MAE | W-CAST MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| temperature | 24h | 1.433711 | 1.380465 | 1.346150 | 1.346505 |
| temperature | 48h | 1.676702 | 1.620751 | 1.597765 | 1.598921 |
| temperature | 72h | 2.206099 | 1.906399 | 2.033256 | 2.025518 |
| precipitation | 24h | 1.949196 | 1.990714 | 1.952366 | 1.959915 |
| precipitation | 48h | 2.261688 | 1.691400 | 1.954356 | 1.939500 |
| precipitation | 72h | 1.565625 | 1.591000 | 1.530063 | 1.550541 |
| wind_speed | 24h | 7.864257 | 7.883701 | 7.865662 | 7.865470 |
| wind_speed | 48h | 8.607988 | 8.802734 | 8.695346 | 8.693651 |
| wind_speed | 72h | 9.038974 | 9.430834 | 9.233360 | 9.229132 |

| Variable | Lead | W-CAST vs GFS | W-CAST vs GEFS |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.087206 | 0.033960 |
| temperature | 48h | 0.077781 | 0.021830 |
| temperature | 72h | 0.180581 | -0.119119 |
| precipitation | 24h | -0.010719 | 0.030799 |
| precipitation | 48h | 0.322188 | -0.248100 |
| precipitation | 72h | 0.015084 | 0.040459 |
| wind_speed | 24h | -0.001213 | 0.018231 |
| wind_speed | 48h | -0.085663 | 0.109082 |
| wind_speed | 72h | -0.190158 | 0.201703 |

## Weight Adaptation

Examples from the active audited weight map (not ranked):

### Mumbai

| Variable | Lead | GFS Weight | GEFS Weight |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.475309 | 0.524691 |
| temperature | 48h | 0.468702 | 0.531298 |
| temperature | 72h | 0.492633 | 0.507367 |
| precipitation | 24h | 0.335644 | 0.664356 |
| precipitation | 48h | 0.370911 | 0.629089 |
| precipitation | 72h | 0.281915 | 0.718085 |
| wind_speed | 24h | 0.513818 | 0.486182 |
| wind_speed | 48h | 0.520210 | 0.479790 |
| wind_speed | 72h | 0.513161 | 0.486839 |

### Delhi

| Variable | Lead | GFS Weight | GEFS Weight |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.482171 | 0.517829 |
| temperature | 48h | 0.479988 | 0.520012 |
| temperature | 72h | 0.548916 | 0.451084 |
| precipitation | 24h | 0.999930 | 0.000070 |
| precipitation | 48h | 0.500000 | 0.500000 |
| precipitation | 72h | 0.500000 | 0.500000 |
| wind_speed | 24h | 0.518719 | 0.481281 |
| wind_speed | 48h | 0.524462 | 0.475538 |
| wind_speed | 72h | 0.554603 | 0.445397 |

### Chennai

| Variable | Lead | GFS Weight | GEFS Weight |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.503246 | 0.496754 |
| temperature | 48h | 0.542347 | 0.457653 |
| temperature | 72h | 0.543696 | 0.456304 |
| precipitation | 24h | 0.479843 | 0.520157 |
| precipitation | 48h | 0.526096 | 0.473904 |
| precipitation | 72h | 0.517413 | 0.482587 |
| wind_speed | 24h | 0.506609 | 0.493391 |
| wind_speed | 48h | 0.509451 | 0.490549 |
| wind_speed | 72h | 0.521696 | 0.478304 |

### Kolkata

| Variable | Lead | GFS Weight | GEFS Weight |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.518952 | 0.481048 |
| temperature | 48h | 0.494654 | 0.505346 |
| temperature | 72h | 0.508315 | 0.491685 |
| precipitation | 24h | 0.522131 | 0.477869 |
| precipitation | 48h | 0.549460 | 0.450540 |
| precipitation | 72h | 0.690942 | 0.309058 |
| wind_speed | 24h | 0.488651 | 0.511349 |
| wind_speed | 48h | 0.490360 | 0.509640 |
| wind_speed | 72h | 0.503759 | 0.496241 |

### Srinagar

| Variable | Lead | GFS Weight | GEFS Weight |
| --- | ---: | ---: | ---: |
| temperature | 24h | 0.617765 | 0.382235 |
| temperature | 48h | 0.611485 | 0.388515 |
| temperature | 72h | 0.538204 | 0.461796 |
| precipitation | 24h | 0.528669 | 0.471331 |
| precipitation | 48h | 0.468085 | 0.531915 |
| precipitation | 72h | 0.575163 | 0.424837 |
| wind_speed | 24h | 0.517857 | 0.482143 |
| wind_speed | 48h | 0.549328 | 0.450672 |
| wind_speed | 72h | 0.554288 | 0.445712 |

## Extreme Weather Validation

This validation uses the current detector thresholds exactly as implemented in the project and stores the results in [data/processed/extreme_event_validation.json](data/processed/extreme_event_validation.json).

- Some event categories had insufficient observed positive events for stable metrics.
- `heat_wave` and `heavy_rain` had no observed positive cases in the audited period, so precision, recall, and F1 are undefined for those windows.
- `high_wind` had no forecast positive events generated by the current thresholds, so precision is undefined rather than treated as a failure.

## Limitations

- Short historical window: only the audited 2026-09-20 to 2026-09-24 period is available.
- GEFS historical archive availability is uneven across dates and initialization windows.
- One primary season dominates the sample, so the record set is not a broad multi-season climatology.
- Extreme-event samples remain too small for stable classification metrics across every category.
- No blanket W-CAST improvement claim is made beyond the measured results above.

