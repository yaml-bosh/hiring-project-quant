import numpy as np
import pandas as pd


def time_decay_weights(
    index: pd.DatetimeIndex, ref_time: pd.Timestamp, half_life_days: float
) -> np.ndarray:
    age_days = (ref_time - index) / pd.Timedelta(days=1)
    age_days = np.maximum(age_days.to_numpy(dtype=float), 0.0)
    return np.power(0.5, age_days / half_life_days)


def _t0_start_utc(ts: pd.Timestamp, mtu_minutes: int = 15) -> pd.Timestamp:
    if ts.tz is None:
        raise ValueError("Timestamp must be timezone-aware")
    ts_utc = ts.tz_convert("UTC")
    return ts_utc.floor(f"{mtu_minutes}min")


def get_forecast_horizon(
    prediction_time: pd.Timestamp, periods: int = 8, mtu_minutes: int = 15
) -> pd.DatetimeIndex:
    t0 = _t0_start_utc(prediction_time, mtu_minutes=mtu_minutes)
    start = t0 + pd.Timedelta(minutes=mtu_minutes)  # t1
    return pd.date_range(
        start=start, periods=periods, freq=f"{mtu_minutes}min", tz="UTC"
    )


def delivery_start_for_horizon(
    feature_times: pd.DatetimeIndex, horizon: int, mtu_minutes: int = 15
) -> pd.DatetimeIndex:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if feature_times.tz is None:
        raise ValueError("feature_times must be timezone-aware")

    ft_utc = feature_times.tz_convert("UTC")
    t0 = ft_utc.floor(f"{mtu_minutes}min")
    return t0 + pd.Timedelta(minutes=mtu_minutes * horizon)


def select_row_asof(
    X: pd.DataFrame, ts: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if ts in X.index:
        return X.loc[[ts]], ts
    row = X.loc[:ts].tail(1)
    if row.empty:
        raise ValueError("Not enough data to form features at/ before prediction time")
    return row, row.index[0]


def clean_time_series(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    return s[~s.index.duplicated(keep="last")]


def asof_slice(df: pd.DataFrame | None, asof_time: pd.Timestamp) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    return df.loc[:asof_time]
