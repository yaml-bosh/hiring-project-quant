from __future__ import annotations

import logging
from enum import Enum
from typing import Literal, Optional

import pandas as pd
from entsoe import Area
from typer import run

from src.fetchers.energinet_fetcher import EnerginetFetcher
from src.models.activation_volume_forecast import ActivationVolumeForecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOCAL_TZ = "Europe/Copenhagen"
UTC = "UTC"
HORIZON_MTUS = 8
MTU_MINUTES = 15


class BiddingZone(str, Enum):
    DK1 = "DK1"
    DK2 = "DK2"

    @property
    def entsoe_area(self) -> Area:
        return {BiddingZone.DK1: Area.DK_1, BiddingZone.DK2: Area.DK_2}[self]


def _parse_local_to_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(LOCAL_TZ, ambiguous=False, nonexistent="shift_forward")
    else:
        t = t.tz_convert(LOCAL_TZ)
    return t.tz_convert(UTC).floor("min")


def _fetch_inputs(
    fetcher: EnerginetFetcher,
    *,
    entsoe_area: Area,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    use_fundamental_forecasts: bool,
) -> tuple[pd.Series, Optional[pd.DataFrame]]:
    act_df = fetcher.get_afrr_setpoint(
        start=start_utc, end=end_utc, country_code=entsoe_area
    )
    activation_min = act_df.iloc[:, 0]

    forecast_5min = None
    if use_fundamental_forecasts:
        forecast_5min = fetcher.get_forecasts_5_min(
            start=start_utc - pd.Timedelta(hours=2),
            end=end_utc + pd.Timedelta(hours=6),
            country_code=entsoe_area,
        )

    return activation_min, forecast_5min


def main(
    task: Literal["predict", "backtest"],
    area: BiddingZone,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    train_window_days: int = 60,
    half_life_days: int = 30,
    retrain_every_minutes: int = 60 * 24 * 7,
    lags: int = 60,
    use_fundamental_forecasts: bool = True,
):
    fetcher = EnerginetFetcher()
    model = ActivationVolumeForecast(
        lags=lags,
        train_window_days=train_window_days,
        half_life_days=half_life_days,
        retrain_every_minutes=retrain_every_minutes,
        use_fundamental_forecasts=use_fundamental_forecasts,
        max_horizon_periods=HORIZON_MTUS,
        mtu_minutes=MTU_MINUTES,
    )

    entsoe_area = area.entsoe_area
    now_utc = pd.Timestamp("now", tz=UTC).floor("min")

    if task == "predict":
        start_utc = now_utc - pd.Timedelta(days=train_window_days, hours=2)
        end_utc = now_utc

        activation_min, forecast_5min = _fetch_inputs(
            fetcher,
            entsoe_area=entsoe_area,
            start_utc=start_utc,
            end_utc=end_utc,
            use_fundamental_forecasts=use_fundamental_forecasts,
        )

        pred_utc = model.predict(
            prediction_time=now_utc,
            activation_min=activation_min,
            fundamental_forecast_5min=forecast_5min,
        )
        pred_local = pred_utc.copy()
        pred_local.index = pred_local.index.tz_convert(LOCAL_TZ)
        print(
            f"aFRR activation prediction for {area.value} (generated at {now_utc.tz_convert(LOCAL_TZ)})"
        )
        print(pred_local.to_string())
        return

    if task == "backtest":
        if time_from is None:
            raise ValueError(
                f"time_from is required (format: 'YYYY-MM-DD HH:MM' in {LOCAL_TZ})"
            )

        scoring_start_utc = _parse_local_to_utc(time_from)
        scoring_end_utc = (
            _parse_local_to_utc(time_to) if time_to is not None else now_utc
        )
        if scoring_end_utc <= scoring_start_utc:
            raise ValueError(
                f"time_to must be after time_from. Got {scoring_start_utc=} {scoring_end_utc=}."
            )

        # fetch window: training history + targets out to horizon and some buffers
        fetch_start_utc = scoring_start_utc - pd.Timedelta(
            days=train_window_days, hours=2
        )
        fetch_end_utc = scoring_end_utc + pd.Timedelta(
            minutes=MTU_MINUTES * HORIZON_MTUS + 5
        )

        activation_min, forecast_5min = _fetch_inputs(
            fetcher,
            entsoe_area=entsoe_area,
            start_utc=fetch_start_utc,
            end_utc=fetch_end_utc,
            use_fundamental_forecasts=use_fundamental_forecasts,
        )

        if activation_min.empty:
            raise ValueError("No activation data returned for requested window.")

        # sanity checks
        required_history_start_utc = scoring_start_utc - pd.Timedelta(
            days=train_window_days
        )
        if activation_min.index.min() > required_history_start_utc:
            raise ValueError(
                "Not enough history to backtest.\n"
                f"Need data from <= {required_history_start_utc} but got {activation_min.index.min()}."
            )

        needed_end_utc = scoring_end_utc + pd.Timedelta(
            minutes=MTU_MINUTES * HORIZON_MTUS
        )
        if activation_min.index.max() < needed_end_utc - pd.Timedelta(minutes=2):
            raise ValueError(
                "Not enough data to cover scoring window with targets.\n"
                f"Need data until ~{needed_end_utc} but got {activation_min.index.max()}."
            )

        mae, sign_acc = model.backtest(
            activation_min=activation_min,
            fundamental_forecast_5min=forecast_5min,
            start_time=scoring_start_utc,
            end_time=scoring_end_utc,
            step_minutes=1,
        )

        print(f"Backtest results for {area.value}")
        print(
            "Scoring window (local): "
            f"{scoring_start_utc.tz_convert(LOCAL_TZ)} -> {scoring_end_utc.tz_convert(LOCAL_TZ)}"
        )
        print("MAE:")
        print(mae.to_string())
        print("Sign accuracy for activations > 10 MW (abs):")
        print(sign_acc.to_string())
        return

    raise ValueError("task must be 'predict' or 'backtest'")


if __name__ == "__main__":
    run(main)
