import logging
from datetime import datetime
from enum import Enum, unique
from typing import Any

import pandas as pd
from entsoe import Area
from requests import Response
from requests_cache import CachedSession


@unique
class EnerginetDataTopic(Enum):
    FORECASTS_5_MIN = "Forecasts_5Min"
    POWER_SYSTEM_RIGHT_NOW = "PowerSystemRightNow"


class EnerginetFetcher:
    ENERGINET_QUERY_URL = "https://api.energidataservice.dk/dataset"
    LOGGER = logging.getLogger(__name__)

    @staticmethod
    def get_response(url: str, params: dict[str, Any]) -> Response:
        session = CachedSession(cache_name=".cache", expire_after=30)
        response = session.get(url, params=params)
        if response.status_code != 200:
            EnerginetFetcher.LOGGER.warning(str(response.content.decode()))
            response.raise_for_status()
        return response

    def query_energi_dataservice(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        topic: EnerginetDataTopic,
        offset: int = 0,
        datetime_index: str | None = None,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        start_utc = start.tz_convert("UTC")
        end_utc = end.tz_convert("UTC")
        start_str = start_utc.strftime("%Y-%m-%dT%H:%M")
        end_str = end_utc.strftime("%Y-%m-%dT%H:%M")
        url = f"{self.ENERGINET_QUERY_URL}/{topic.value}"
        params = {
            "offset": offset,
            "start": start_str,
            "end": end_str,
            "timezone": timezone,
        }
        if datetime_index:
            params["sort"] = datetime_index
        response = self.get_response(url, params)
        response_json = response.json()
        try:
            records = response_json["records"]
        except KeyError:
            raise FileNotFoundError()
        df = pd.DataFrame.from_records(records)

        if not df.empty and datetime_index in df.columns:
            df[datetime_index] = pd.to_datetime(
                df[datetime_index], format="%Y-%m-%dT%H:%M:%S"
            ).dt.tz_localize("UTC")
            df.set_index(datetime_index, inplace=True)

        if "PriceArea" in df.columns:
            df["PriceArea"] = df["PriceArea"].replace(
                {"DK1": Area.DK_1.name, "DK2": Area.DK_2.name}
            )
        return df

    def get_afrr_setpoint(
        self, start: pd.Timestamp, end: pd.Timestamp, country_code: Area
    ) -> pd.DataFrame:
        df = self.query_energi_dataservice(
            start=start,
            end=end,
            topic=EnerginetDataTopic.POWER_SYSTEM_RIGHT_NOW,
            datetime_index="Minutes1UTC",
            timezone="UTC",
        )
        col_map = {
            Area.DK_1: "aFRR_ActivatedDK1",
            Area.DK_2: "aFRR_ActivatedDK2",
        }
        col = col_map.get(country_code)
        if col is None:
            raise ValueError(f"Unsupported country_code: {country_code.name}")
        result = df[[col]].copy()
        result.rename(columns={col: "afrr_setpoint_mw"}, inplace=True)
        return result

    def get_forecasts_5_min(
            self, start: pd.Timestamp, end: pd.Timestamp, country_code: Area
    ) -> pd.DataFrame:
        df = self.query_energi_dataservice(
            start=start,
            end=end,
            topic=EnerginetDataTopic.FORECASTS_5_MIN,
            datetime_index="Minutes5UTC",
            timezone="UTC",
        )
        df = df[df["PriceArea"] == country_code.name].copy()
        df["PriceArea"] = df["PriceArea"].str.lower()
        df["ForecastType"] = df["ForecastType"].astype(str).str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
        horizon_map = {
            "ForecastCurrent": "current",
            "Forecast1Hour": "1_hour",
            "Forecast5Hour": "5_hour",
            "ForecastDayAhead": "day_ahead",
        }
        available_horizons = [col for col in horizon_map.keys() if col in df.columns]
        if not available_horizons:
            raise ValueError("Expected forecast horizon columns not found")
        df_wide = df.pivot_table(
            index=df.index,
            columns=["ForecastType"],
            values=available_horizons,
        )
        df_wide.columns = [
            f"{ftype}_forecast_mw_{horizon_map[horizon]}"
            for horizon, ftype in df_wide.columns
        ]
        return df_wide


if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    fetcher = EnerginetFetcher()
    start = pd.Timestamp(datetime(2026, 1, 1), tz="CET")
    end = pd.Timestamp(datetime(2026, 2, 1), tz="CET")
    country_code = Area.DK_2
    forecasts_5_min = fetcher.get_forecasts_5_min(start, end, country_code)
    print(forecasts_5_min.head())
    afrr_setpoint = fetcher.get_afrr_setpoint(start, end, country_code)
    print(afrr_setpoint.head())
