# Data

`python download_data.py` fetches the files below into `data/raw/` from
[Mendeley Data](https://doi.org/10.17632/byx7sztj59.1) and verifies their SHA-256 checksums.

| File | Content |
| --- | --- |
| `continuous dataset.csv` | 48,048 hourly rows, 2015-01-03 01:00 → 2020-06-27 00:00: national load `nat_demand` (MW), 2 m temperature / humidity / liquid water / wind for Tocumen, Santiago and David, holiday id + flag, school-term flag |
| `weekly pre-dispatch forecast.csv` | The grid operator's (CND Panama) official week-ahead load forecast, 2016-01-02 → 2020-07-31 |

Timestamps are Panama local time (UTC−05:00, no DST) and mark the **end** of each hour.

**Source and licence.** Aguilar Madrid, E. (2021). *Short-term electricity load forecasting
(Panama case study)*. Mendeley Data, V1. doi:10.17632/byx7sztj59.1 — CC BY 4.0. Also mirrored
on Kaggle as "Electricity Load Forecasting". Companion paper: Aguilar Madrid, E. & Antonio, N.
(2021). *Short-Term Electricity Load Forecasting with Machine Learning*. Information, 12(2), 50.
