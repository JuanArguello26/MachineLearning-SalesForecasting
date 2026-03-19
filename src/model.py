"""Sales forecasting model using StatsForecast."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from src.config import ModelConfig
from src.preprocessing import SalesDataPreprocessor


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    mae: float
    rmse: float
    mape: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
        }


@dataclass
class ForecastResult:
    """Container for forecast results."""

    predictions: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": self.predictions,
            "summary": self.summary,
        }


class SalesForecastingModel:
    """Sales forecasting model wrapper using StatsForecast."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.preprocessor = SalesDataPreprocessor()
        self.model: StatsForecast | None = None
        self.df: pd.DataFrame | None = None

    def train(self, df: pd.DataFrame) -> None:
        """Train the forecasting model."""
        df_prepared = self.preprocessor.prepare_for_statsforecast(df)
        models = [
            AutoARIMA(season_length=self.config.season_length),
            SeasonalNaive(season_length=self.config.season_length),
        ]
        self.model = StatsForecast(models=models, freq="D")
        self.model.fit(df_prepared)
        self.df = df

    def predict(self, horizon: int, level: list[int] | None = None) -> ForecastResult:
        """Generate forecasts for future periods."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if level is None:
            level = [self.config.confidence_level]

        forecast = self.model.predict(h=horizon, level=level)
        forecast = forecast.reset_index()

        predictions: list[dict[str, Any]] = []
        for _, row in forecast.iterrows():
            predictions.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "predicted_sales": round(float(row["AutoARIMA"]), 2),
                "lower_bound": round(float(row["AutoARIMA-lo-95"]), 2),
                "upper_bound": round(float(row["AutoARIMA-hi-95"]), 2),
            })

        return ForecastResult(
            predictions=predictions,
            summary={
                "total_predicted": round(float(forecast["AutoARIMA"].sum()), 2),
                "average_daily": round(float(forecast["AutoARIMA"].mean()), 2),
                "period_days": horizon,
            },
        )

    def evaluate(self, test_df: pd.DataFrame) -> ModelMetrics:
        """Evaluate model on test set."""
        if self.df is None:
            raise ValueError("Model not trained. Call train() first.")

        df_prepared = self.preprocessor.prepare_for_statsforecast(self.df)
        train_df, _ = self.preprocessor.split_data(df_prepared, train_size=1.0)

        test_prepared = self.preprocessor.prepare_for_statsforecast(test_df)

        models = [
            AutoARIMA(season_length=self.config.season_length),
            SeasonalNaive(season_length=self.config.season_length),
        ]
        eval_model = StatsForecast(models=models, freq="D")
        eval_model.fit(train_df)

        forecast = eval_model.predict(h=len(test_prepared))
        forecast = forecast.reset_index()

        actual = test_prepared["y"].values
        predicted = forecast["AutoARIMA"].values

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100

        return ModelMetrics(mae=mae, rmse=rmse, mape=mape)

    def full_pipeline(
        self,
        data_path: str,
        forecast_horizon: int = 30,
    ) -> dict[str, Any]:
        """Run complete training and forecasting pipeline."""
        df = self.preprocessor.load_data(data_path)
        self.train(df)
        metrics = self.evaluate(df.tail(int(len(df) * 0.2)))
        forecast = self.predict(forecast_horizon)

        return {
            "metrics": metrics.to_dict(),
            "forecast": forecast.to_dict(),
        }
