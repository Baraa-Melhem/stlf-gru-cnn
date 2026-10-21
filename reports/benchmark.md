# Benchmark — day-ahead walk-forward backtest (Jan 2019 – Jun 2020)

All models forecast the next 24 hours at every midnight. Each of the 6 quarterly folds is retrained from scratch on all earlier data. Anomalous hours (blackouts) are excluded from scoring for every model. MW units; PICP/MPIW refer to the 90% interval.

| model | MAPE % | MAPE σ (seeds) | MAE | RMSE | sMAPE % | R² | MASE | PICP % | MPIW | Official-weeks MAPE % | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ensemble_gru_cnn_lightgbm | 3.02 | – | 36.44 | 50.92 | 2.98 | 0.93 | 0.65 | 88.00 | 149.15 | 2.90 | 3+3 |
| gru_cnn | 3.13 | 0.03 | 37.99 | 52.31 | 3.11 | 0.92 | 0.68 | 84.37 | 137.68 | 3.05 | 3 |
| lightgbm | 3.21 | 0.01 | 38.72 | 54.00 | 3.18 | 0.92 | 0.69 | 85.71 | 160.62 | 3.06 | 3 |
| operator_predispatch | 4.80 | – | 57.60 | 76.02 | 4.70 | 0.84 | 1.03 | – | – | 4.95 | 1 |
| seasonal_naive_weekly | 5.64 | – | 68.51 | 97.22 | 5.60 | 0.73 | 1.23 | 83.57 | 231.37 | 5.96 | 1 |
| seasonal_naive_daily | 5.99 | – | 74.68 | 115.06 | 6.04 | 0.62 | 1.34 | 91.36 | 456.89 | 6.71 | 1 |
| legacy_gru_cnn | 10.17 | – | 119.71 | 151.00 | 9.71 | 0.35 | 2.14 | – | – | 9.85 | 1 |

## Ablations (single seed, same folds)

| variant | MAPE % | MAE | RMSE | PICP % | Winkler |
| --- | ---: | ---: | ---: | ---: | ---: |
| gru_cnn (full model, seed 0) | 3.23 | 39.07 | 53.52 | 84.07 | 237.08 |
| ablation_no_attention | 3.31 | 39.90 | 54.99 | 84.73 | 240.70 |
| ablation_no_cnn | 3.17 | 38.28 | 52.92 | 85.17 | 234.18 |
| ablation_no_future_weather | 3.36 | 40.61 | 55.20 | 84.46 | 242.70 |
| ablation_no_gru | 3.22 | 38.94 | 53.10 | 85.26 | 236.32 |
| ablation_no_lags | 3.37 | 40.40 | 55.85 | 83.33 | 247.96 |
| ablation_no_revin | 3.68 | 44.54 | 63.35 | 82.27 | 286.84 |

## The 14 official test weeks

Weeks proposed in the dataset's metadata (Aguilar Madrid, 2021). The operator column is the official *week-ahead* pre-dispatch forecast, made with weather forecasts; the models here forecast day-ahead with observed weather, so it is a reference, not a like-for-like race.

| week starting | gru_cnn | lightgbm | ensemble_gru_cnn_lightgbm | operator_predispatch |
| --- | ---: | ---: | ---: | ---: |
| 2019-04-13 | 1.96 | 1.95 | 1.79 | 3.90 |
| 2019-05-25 | 2.26 | 2.41 | 2.25 | 3.10 |
| 2019-06-15 | 2.61 | 2.15 | 2.22 | 6.08 |
| 2019-07-20 | 3.41 | 3.96 | 3.58 | 5.55 |
| 2019-08-17 | 2.58 | 2.75 | 2.50 | 4.16 |
| 2019-09-14 | 2.94 | 3.08 | 2.97 | 4.68 |
| 2019-10-12 | 2.37 | 2.75 | 2.52 | 5.04 |
| 2019-11-02 | 2.70 | 3.19 | 2.78 | 6.65 |
| 2019-12-21 | 5.55 | 4.24 | 4.83 | 5.38 |
| 2020-01-04 | 3.50 | 3.47 | 3.19 | 4.06 |
| 2020-02-08 | 1.56 | 1.25 | 1.25 | 3.79 |
| 2020-03-07 | 2.54 | 1.93 | 1.93 | 2.93 |
| 2020-05-16 | 4.97 | 5.64 | 5.08 | 9.06 |
| 2020-06-13 | 3.72 | 4.11 | 3.77 | 4.96 |
| **mean** | 3.05 | 3.06 | 2.90 | 4.95 |
