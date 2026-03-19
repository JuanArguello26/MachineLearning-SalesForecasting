"""Data preprocessing for sales forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataQualityReport:
    """Container for data quality metrics."""

    total_records: int
    missing_values: int
    date_range: tuple[str, str]
    sales_stats: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_records": self.total_records,
            "missing_values": self.missing_values,
            "date_range": self.date_range,
            "sales_stats": self.sales_stats,
        }


class SalesDataPreprocessor:
    """Handles data preprocessing for sales forecasting."""

    def __init__(self) -> None:
        self.feature_names: list[str] = ["month", "day_of_week", "year"]

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load sales data from CSV."""
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def prepare_for_statsforecast(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for StatsForecast format."""
        df_prepared = df.rename(columns={"date": "ds", "sales": "y"})
        df_prepared["unique_id"] = "sales"
        return df_prepared[["ds", "y", "unique_id"]]

    def split_data(
        self, df: pd.DataFrame, train_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_idx = int(len(df) * train_size)
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    def get_data_quality_report(self, df: pd.DataFrame) -> DataQualityReport:
        """Generate data quality report."""
        return DataQualityReport(
            total_records=int(len(df)),
            missing_values=int(df["sales"].isna().sum()),
            date_range=(str(df["date"].min()), str(df["date"].max())),
            sales_stats={
                "min": float(df["sales"].min()),
                "max": float(df["sales"].max()),
                "mean": float(df["sales"].mean()),
                "std": float(df["sales"].std()),
            },
        )

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for analysis."""
        df_features = df.copy()
        df_features["month"] = df_features["date"].dt.month
        df_features["day_of_week"] = df_features["date"].dt.dayofweek
        df_features["year"] = df_features["date"].dt.year
        return df_features

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality."""
        if len(df) == 0:
            return False
        if df["sales"].isna().any():
            return False
        if (df["sales"] <= 0).any():
            return False
        return True
