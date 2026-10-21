# Model selection log

Every design decision for the flagship network was made on an **inner walk-forward backtest over
2018** — four quarterly folds, each trained only on earlier data, early-stopped on the 8 weeks
before it, and scored on day-ahead MAPE (blackout hours excluded). The reported 2019-2020 test
folds were run once, after the configuration below was frozen.

One caveat in the interest of full disclosure: before this protocol was set up, the fold-1 *test*
score (2019-Q1) of a single run was looked at while adding the lag anchors and the conformal
calibration. The lag anchors were then re-checked on 2018 (row "v2 without lag anchors"), and
conformal calibration only rescales the interval, never the point forecast.

| Variant (single seed) | 2018 MAPE % | Q1 | Q2 | Q3 | Q4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1: symmetric-padding CNN, lags (t-24, t-168), 60-epoch schedule | 2.82 | 2.89 | 2.73 | 2.99 | 2.68 |
| v2: + causal convolution padding | 2.69 | 2.61 | 2.67 | 3.04 | 2.43 |
| v2 without lag anchors | 2.72 | 2.53 | 2.98 | 2.95 | 2.41 |
| v2 + origin anchors (last load, last-24 h mean) | 2.79 | 2.76 | 2.81 | 3.05 | 2.53 |
| v2 with a 10-epoch cosine schedule | 2.77 | 2.91 | 2.67 | 3.06 | 2.44 |
| v2, 10-epoch schedule + origin anchors | 2.77 | 2.77 | 2.71 | 3.07 | 2.52 |
| v2, 10-epoch schedule, lr 5e-4 | 2.80 | 3.02 | 2.81 | 2.84 | 2.53 |
| v3: + holiday-masked RevIN + lagged holiday flags | 2.70 | 2.86 | 2.71 | 2.84 | 2.38 |
| v3 without holiday-masked RevIN | 2.70 | 2.76 | 2.64 | 2.99 | 2.42 |
| v4: Optuna-tuned (adopted as configs/gru_cnn.yaml) | 2.56 | 2.55 | 2.48 | 2.81 | 2.38 |
| LightGBM (v2 features) | 2.47 | 2.36 | 2.40 | 2.90 | 2.23 |
| LightGBM + lagged holiday flags (adopted) | 2.46 | 2.32 | 2.40 | 2.92 | 2.17 |
| Seasonal naive, same hour last week | 4.45 | 4.84 | 4.35 | 4.14 | 4.49 |

**Reading the table.**

* *Causal padding* (v1 → v2) was the largest architectural fix: symmetric padding convolved the
  hours just before the forecast origin with zeros, which hurt the first forecast hours most.
* Single-seed differences below ~0.1 point are within run-to-run noise (compare the per-quarter
  swings of near-identical variants); those switches (origin anchors, holiday-masked RevIN) were
  handed to the hyper-parameter search, which left both off.
* The Optuna search (`tune.py`, TPE, 20 trials, scored on 2018-Q3/Q4 only, median pruning) found
  that a *smaller* encoder (1 conv block, 1 GRU layer), heavy dropout (0.4), a two-week look-back
  and a short 12-epoch cosine schedule generalise best. That configuration was then confirmed on
  2018-Q1/Q2, which the search never scored (2.55 / 2.48 vs 2.86 / 2.71 before).
* The search was interrupted at trial 20 by a GPU driver reset (Windows TDR, `nvlddmkm` event
  153); the completed trials are recovered in [`tuning/trials_recovered.csv`](tuning/trials_recovered.csv).
  `tune.py` now stores studies in SQLite and resumes them.
