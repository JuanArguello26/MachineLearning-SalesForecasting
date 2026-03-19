"""Tests for Sales Forecasting Model."""

from __future__ import annotations

import os
import sys
from typing import Generator

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import SalesDataPreprocessor
from src.model import SalesForecastingModel


@pytest.fixture
def data_path() -> str:
    """Get path to test data."""
    return "data/sales_data.csv"


@pytest.fixture
def sample_dataframe(data_path: str) -> pd.DataFrame:
    """Load sample data for testing."""
    return pd.read_csv(data_path)


@pytest.fixture
def preprocessor() -> SalesDataPreprocessor:
    """Create preprocessor instance."""
    return SalesDataPreprocessor()


@pytest.fixture
def model() -> SalesForecastingModel:
    """Create model instance."""
    return SalesForecastingModel()


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_data_file_exists(self, data_path: str) -> None:
        """Test that data file exists."""
        assert os.path.exists(data_path), "Data file not found"

    def test_data_not_empty(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that data is not empty."""
        assert len(sample_dataframe) > 0, "Data is empty"

    def test_data_has_required_columns(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that all required columns are present."""
        assert "date" in sample_dataframe.columns, "Missing date column"
        assert "sales" in sample_dataframe.columns, "Missing sales column"

    def test_date_format(self, sample_dataframe: pd.DataFrame) -> None:
        """Test date column format."""
        assert pd.api.types.is_datetime64_any_dtype(sample_dataframe["date"]), "Date should be datetime"


class TestDataQuality:
    """Tests for data quality."""

    def test_no_missing_values(self, sample_dataframe: pd.DataFrame) -> None:
        """Test no missing values in sales."""
        assert sample_dataframe["sales"].isna().sum() == 0, "Missing values in sales"

    def test_positive_sales(self, sample_dataframe: pd.DataFrame) -> None:
        """Test all sales are positive."""
        assert (sample_dataframe["sales"] > 0).all(), "All sales should be positive"


class TestPreprocessing:
    """Tests for preprocessing functionality."""

    def test_prepare_for_statsforecast(
        self, preprocessor: SalesDataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test statsforecast format preparation."""
        df_prepared = preprocessor.prepare_for_statsforecast(sample_dataframe)
        assert "ds" in df_prepared.columns
        assert "y" in df_prepared.columns
        assert "unique_id" in df_prepared.columns

    def test_split_data(
        self, preprocessor: SalesDataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test data splitting."""
        train, test = preprocessor.split_data(sample_dataframe, train_size=0.8)
        assert len(train) > 0, "Training set is empty"
        assert len(test) > 0, "Test set is empty"
        assert len(train) + len(test) == len(sample_dataframe)

    def test_data_quality_report(
        self, preprocessor: SalesDataPreprocessor, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test data quality report generation."""
        report = preprocessor.get_data_quality_report(sample_dataframe)
        assert report.total_records > 0
        assert "sales_stats" in report.to_dict()


class TestModel:
    """Tests for model functionality."""

    def test_model_training(
        self, model: SalesForecastingModel, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test model training."""
        model.train(sample_dataframe)
        assert model.model is not None, "Model should be trained"

    def test_model_predictions(
        self, model: SalesForecastingModel, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test model predictions."""
        model.train(sample_dataframe)
        result = model.predict(horizon=30)
        
        assert len(result.predictions) == 30, "Should have 30 predictions"
        assert "predicted_sales" in result.predictions[0]
        assert "lower_bound" in result.predictions[0]
        assert "upper_bound" in result.predictions[0]

    def test_model_evaluation(
        self, model: SalesForecastingModel, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test model evaluation."""
        model.train(sample_dataframe)
        test_df = sample_dataframe.tail(50)
        metrics = model.evaluate(test_df)
        
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert metrics.mape >= 0

    def test_full_pipeline(self, model: SalesForecastingModel, data_path: str) -> None:
        """Test complete pipeline."""
        result = model.full_pipeline(data_path, forecast_horizon=7)
        
        assert "metrics" in result
        assert "forecast" in result
        assert "MAE" in result["metrics"]
        assert "RMSE" in result["metrics"]
        assert "MAPE" in result["metrics"]


class TestForecast:
    """Tests for forecast functionality."""

    def test_future_predictions(
        self, model: SalesForecastingModel, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test future predictions generation."""
        model.train(sample_dataframe)
        result = model.predict(horizon=30, level=[95])
        
        assert len(result.predictions) == 30, "Should have 30 predictions"
        assert "AutoARIMA" in str(result.predictions[0].keys()) or "predicted_sales" in result.predictions[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
