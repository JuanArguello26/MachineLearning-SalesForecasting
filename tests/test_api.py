"""Tests for Sales Forecasting API."""

from __future__ import annotations

import os
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Tests for API endpoints."""

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data

    def test_health_check(self, client: TestClient) -> None:
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    @patch("api.main.model")
    def test_predict_default_days(
        self, mock_model: MagicMock, client: TestClient
    ) -> None:
        """Test prediction with default days."""
        mock_result = MagicMock()
        mock_result.predictions = [
            {
                "date": "2024-01-01",
                "predicted_sales": 25000.0,
                "lower_bound": 23000.0,
                "upper_bound": 27000.0,
            }
        ]
        mock_result.summary = {
            "total_predicted": 25000.0,
            "average_daily": 25000.0,
            "period_days": 1,
        }
        mock_model.predict.return_value = mock_result

        response = client.post("/predict", json={})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "summary" in data
        assert len(data["predictions"]) > 0

    @patch("api.main.model")
    def test_predict_custom_days(
        self, mock_model: MagicMock, client: TestClient
    ) -> None:
        """Test prediction with custom days."""
        mock_result = MagicMock()
        mock_result.predictions = [
            {
                "date": f"2024-01-{str(i).zfill(2)}",
                "predicted_sales": 25000.0,
                "lower_bound": 23000.0,
                "upper_bound": 27000.0,
            }
            for i in range(1, 8)
        ]
        mock_result.summary = {
            "total_predicted": 175000.0,
            "average_daily": 25000.0,
            "period_days": 7,
        }
        mock_model.predict.return_value = mock_result

        response = client.post("/predict", json={"days": 7})
        assert response.status_code == 200
        data = response.json()
        assert data["summary"]["period_days"] == 7
        assert len(data["predictions"]) == 7

    def test_predict_invalid_days_negative(self, client: TestClient) -> None:
        """Test prediction with negative days."""
        response = client.post("/predict", json={"days": -1})
        assert response.status_code == 422

    def test_predict_invalid_days_zero(self, client: TestClient) -> None:
        """Test prediction with zero days."""
        response = client.post("/predict", json={"days": 0})
        assert response.status_code == 422

    def test_predict_invalid_days_too_large(self, client: TestClient) -> None:
        """Test prediction with too many days."""
        response = client.post("/predict", json={"days": 500})
        assert response.status_code == 422


class TestDataQualityEndpoint:
    """Tests for data quality endpoint."""

    def test_data_quality_endpoint(self, client: TestClient) -> None:
        """Test data quality endpoint exists."""
        response = client.get("/data-quality")
        assert response.status_code == 200
        data = response.json()
        assert "total_records" in data
        assert "missing_values" in data
        assert "date_range" in data
        assert "sales_stats" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
