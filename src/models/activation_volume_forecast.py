from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from tqdm import tqdm

from src.utils.utils import (
    asof_slice,
    clean_time_series,
    delivery_start_for_horizon,
    get_forecast_horizon,
    select_row_asof,
    time_decay_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class ActivationVolumeForecast:
    """
    Multi-horizon regression model to predict average aFRR setpoint for the next 8 MTUs.
    """

    lags: int = 60
    train_window_days: int = 60
    half_life_days: int = 30
    retrain_every_minutes: int = 60 * 24 * 7
    use_fundamental_forecasts: bool = True
    max_horizon_periods: int = 8
    mtu_minutes: int = 15
    lgbm_kwargs: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.lgbm_kwargs is None:
            self.lgbm_kwargs = dict(
                objective="l1",
                reg_alpha=2.0,
                reg_lambda=0.0,
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,  # lightgbm warnings
            )

        self.models: dict[int, LGBMRegressor] = {
            h: LGBMRegressor(**self.lgbm_kwargs)
            for h in range(1, self.max_horizon_periods + 1)
        }

        self.feature_columns_: Optional[list[str]] = None
        self.last_fit_time_: Optional[pd.Timestamp] = None

        logger.info(
            "ActivationVolumeForecast initialised | horizons=%d | lags=%d | window_days=%d | half_life_days=%d | retrain_every_minutes=%d | use_fundamental_forecasts=%s",
            self.max_horizon_periods,
            self.lags,
            self.train_window_days,
            self.half_life_days,
            self.retrain_every_minutes,
            self.use_fundamental_forecasts,
        )

    def fit_model_if_training_due(
        self,
        *,
        prediction_time: pd.Timestamp,
        activation_min: pd.Series,
        fundamental_forecast_5min: Optional[pd.DataFrame] = None,
    ) -> bool:

        if not self._is_training_due(prediction_time):
            return False

        logger.info("Training started | prediction_time=%s", prediction_time)

        activation_min = clean_time_series(activation_min)
        if activation_min.empty:
            logger.warning("Training aborted: activation series empty")
            return False

        train_start = prediction_time - pd.Timedelta(days=self.train_window_days)
        activation_window = activation_min.loc[train_start:prediction_time]
        logger.info("Training window rows=%d", len(activation_window))

        forecast_asof = asof_slice(fundamental_forecast_5min, prediction_time)

        X_all, Y_all = self.build_training_dataset(
            activation_min=activation_window,
            fundamental_forecast_5min=forecast_asof,
        )

        X_all = self._align_feature_columns(X_all, fit_mode=True)

        weights_all = pd.Series(
            time_decay_weights(
                X_all.index,
                ref_time=prediction_time,
                half_life_days=self.half_life_days,
            ),
            index=X_all.index,
        )

        # fit per-horizon models with per-horizon as-of cutoffs
        horizons_fitted = 0
        for h in range(1, self.max_horizon_periods + 1):
            cutoff_h = prediction_time - pd.Timedelta(minutes=self.mtu_minutes * h)
            X_h = X_all.loc[:cutoff_h]
            y_h = Y_all[h].reindex(X_h.index)

            valid = X_h.notna().all(axis=1) & y_h.notna()
            X_h = X_h.loc[valid]
            y_h = y_h.loc[valid]

            w_h = weights_all.loc[X_h.index].to_numpy()

            model = LGBMRegressor(**self.lgbm_kwargs)
            model.fit(X_h, y_h, sample_weight=w_h)
            self.models[h] = model

            horizons_fitted += 1
            logger.info("Horizon model trained | h=%d | rows=%d", h, len(X_h))

        if horizons_fitted == 0:
            logger.warning("Training aborted: no horizons fitted")
            return False

        self.last_fit_time_ = prediction_time
        logger.info(
            "Training complete | horizons_fitted=%d/%d",
            horizons_fitted,
            self.max_horizon_periods,
        )
        return True

    def predict(
        self,
        *,
        prediction_time: pd.Timestamp,
        activation_min: pd.Series,
        fundamental_forecast_5min: Optional[pd.DataFrame] = None,
    ) -> pd.Series:

        activation_min = clean_time_series(activation_min)
        if activation_min.empty:
            raise ValueError("activation_min empty; cannot predict")

        retrained = self.fit_model_if_training_due(
            prediction_time=prediction_time,
            activation_min=activation_min,
            fundamental_forecast_5min=fundamental_forecast_5min,
        )

        if self.feature_columns_ is None:
            raise ValueError("Model has not been fit yet.")

        forecast_asof = asof_slice(fundamental_forecast_5min, prediction_time)

        X_pred = self.build_feature_matrix(
            activation_min=activation_min.loc[:prediction_time],
            fundamental_forecast_5min=forecast_asof,
        )

        X_row, effective_time = select_row_asof(X_pred, prediction_time)
        X_row = self._align_feature_columns(X_row, fit_mode=False)

        delivery_index = get_forecast_horizon(
            effective_time,
            periods=self.max_horizon_periods,
            mtu_minutes=self.mtu_minutes,
        )

        preds = [
            float(self.models[h].predict(X_row)[0])
            for h in range(1, self.max_horizon_periods + 1)
        ]

        return pd.Series(preds, index=delivery_index, name="prediction")

    def backtest(
        self,
        *,
        activation_min: pd.Series,
        fundamental_forecast_5min: Optional[pd.DataFrame] = None,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
        step_minutes: int = 1,
    ) -> tuple[pd.Series, pd.Series]:

        activation_min = clean_time_series(activation_min)
        if activation_min.empty:
            raise ValueError("activation_min empty; cannot backtest")

        target_q = activation_min.resample(f"{self.mtu_minutes}min").mean()

        if start_time is None:
            start_time = activation_min.index.min() + pd.Timedelta(
                days=self.train_window_days
            )
        if end_time is None:
            end_time = activation_min.index.max()

        logger.info("Backtest started | start=%s end=%s", start_time, end_time)

        self.last_fit_time_ = None
        self.feature_columns_ = None
        self.models = {
            h: LGBMRegressor(**self.lgbm_kwargs)
            for h in range(1, self.max_horizon_periods + 1)
        }

        prediction_times = activation_min.loc[start_time:end_time].index[::step_minutes]
        rows: list[pd.DataFrame] = []

        for pt in tqdm(prediction_times, desc="backtest"):
            pred = self.predict(
                prediction_time=pt,
                activation_min=activation_min,
                fundamental_forecast_5min=fundamental_forecast_5min,
            )

            realized = target_q.reindex(pred.index)
            df = pd.DataFrame(
                {
                    "prediction_time": pt,
                    "delivery_start": pred.index,
                    "prediction": pred.values,
                    "target": realized.values,
                }
            ).dropna()

            if not df.empty:
                df["time_to_delivery"] = df["delivery_start"] - df["prediction_time"]
                df["abs_error"] = (df["target"] - df["prediction"]).abs()
                df["sign_correct"] = np.sign(df["target"]) == np.sign(df["prediction"])
                rows.append(df)

        if not rows:
            raise ValueError("Backtest produced no rows.")

        out = pd.concat(rows, ignore_index=True)
        mae = out.groupby("time_to_delivery")["abs_error"].mean()

        # only compute sign accuracy where absolute target is above 10 MW
        mask = out["target"].abs() > 10.0
        sign_acc = out[mask].groupby("time_to_delivery")["sign_correct"].mean()

        logger.info("Backtest complete | rows=%d", len(out))

        return mae, sign_acc

    def build_feature_matrix(
        self,
        *,
        activation_min: pd.Series,
        fundamental_forecast_5min: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:

        activation_min = clean_time_series(activation_min)

        X_lags = self._build_activation_lag_features(activation_min)
        X_calendar = self._build_calendar_features(activation_min.index)
        X_fundamentals = self._build_fundamental_forecast_features(
            fundamental_forecast_5min, activation_min.index
        )

        return pd.concat([X_lags, X_calendar, X_fundamentals], axis=1)

    def build_training_dataset(
        self,
        *,
        activation_min: pd.Series,
        fundamental_forecast_5min: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, dict[int, pd.Series]]:

        activation_min = clean_time_series(activation_min)
        target_mtu = activation_min.resample(f"{self.mtu_minutes}min").mean()

        X = self.build_feature_matrix(
            activation_min=activation_min,
            fundamental_forecast_5min=fundamental_forecast_5min,
        )

        Y: dict[int, pd.Series] = {}
        for h in range(1, self.max_horizon_periods + 1):
            delivery_start = delivery_start_for_horizon(
                X.index, horizon=h, mtu_minutes=self.mtu_minutes
            )
            Y[h] = pd.Series(
                target_mtu.reindex(delivery_start).to_numpy(),
                index=X.index,
                name=f"target_h{h}",
            )

        X = X.loc[X.notna().all(axis=1)]
        for h in range(1, self.max_horizon_periods + 1):
            Y[h] = Y[h].loc[X.index]

        return X, Y

    def _build_calendar_features(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        df = pd.DataFrame(index=idx)
        df["weekday"] = idx.weekday
        df["hour"] = idx.hour
        return df

    def _build_activation_lag_features(self, activation_min: pd.Series) -> pd.DataFrame:
        """
        Setpoint history features (minute resolution).
        Pragmatic set: a few lags + a few diffs + a few rolling stats.
        """
        s = clean_time_series(activation_min)
        idx = s.index

        max_lag = int(self.lags)

        # dense short term + sparse longer term
        dense = list(range(1, min(max_lag, 10) + 1))
        sparse_candidates = [15, 30, 60, 120]
        sparse = [k for k in sparse_candidates if k <= max_lag]
        lag_set = sorted(set(dense + sparse))

        feats: dict[str, pd.Series] = {}

        for k in lag_set:
            feats[f"activation_lag_{k:03d}"] = s.shift(k)

        for w in [1, 5, 15]:
            if w <= max_lag:
                feats[f"activation_diff_{w:03d}m"] = s - s.shift(w)

        a1 = s.shift(1)
        for w in [5, 15, 60]:
            feats[f"activation_roll_mean_{w:03d}m"] = a1.rolling(w).mean()
            feats[f"activation_roll_std_{w:03d}m"] = a1.rolling(w).std()

        return pd.DataFrame(feats, index=idx)

    def _build_fundamental_forecast_features(
        self,
        forecast_5min: Optional[pd.DataFrame],
        minute_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        column format:
            "{tech}_forecast_mw_{horizon}"
        where horizon in {"current","1_hour","5_hour","day_ahead"}.

        feature set:
          - raw levels (ffill to minute granularity)
          - diff_15m of raw levels
          - per-tech deltas: current-1_hour, current-day_ahead
          - diff_15m of those deltas
        """
        if (
            (not self.use_fundamental_forecasts)
            or forecast_5min is None
            or forecast_5min.empty
        ):
            return pd.DataFrame(index=minute_index)

        f0 = forecast_5min.sort_index().reindex(minute_index, method="ffill").copy()
        base_cols = list(f0.columns)

        # parsing and grouping by technology and horizon
        pat = re.compile(
            r"^(?P<tech>.+)_forecast_mw_(?P<horizon>current|1_hour|5_hour|day_ahead)$"
        )
        grouped: dict[str, dict[str, str]] = {}
        for c in base_cols:
            m = pat.match(c)
            if not m:
                continue
            grouped.setdefault(m.group("tech"), {})[m.group("horizon")] = c

        new_cols: dict[str, pd.Series] = {}

        for c in base_cols:
            s = f0[c]
            new_cols[f"{c}_diff_15m"] = s - s.shift(15)

        for tech, by_h in grouped.items():
            c_cur = by_h.get("current")
            c_1h = by_h.get("1_hour")
            c_da = by_h.get("day_ahead")

            if c_cur and c_1h:
                dname = f"{tech}_delta_current_minus_1_hour"
                d = f0[c_cur] - f0[c_1h]
                new_cols[dname] = d
                new_cols[f"{dname}_diff_15m"] = d - d.shift(15)

            if c_cur and c_da:
                dname = f"{tech}_delta_current_minus_day_ahead"
                d = f0[c_cur] - f0[c_da]
                new_cols[dname] = d
                new_cols[f"{dname}_diff_15m"] = d - d.shift(15)

        feats = pd.DataFrame(new_cols, index=minute_index)
        return pd.concat([f0, feats], axis=1)

    def _is_training_due(self, prediction_time: pd.Timestamp) -> bool:
        if self.last_fit_time_ is None:
            return True
        return prediction_time >= self.last_fit_time_ + pd.Timedelta(
            minutes=self.retrain_every_minutes
        )

    def _align_feature_columns(
        self, X: pd.DataFrame, *, fit_mode: bool
    ) -> pd.DataFrame:
        if fit_mode:
            if self.feature_columns_ is None:
                self.feature_columns_ = list(X.columns)
                return X
            return X.reindex(columns=self.feature_columns_)

        if self.feature_columns_ is None:
            raise ValueError("Model has not been fit yet.")
        return X.reindex(columns=self.feature_columns_)
