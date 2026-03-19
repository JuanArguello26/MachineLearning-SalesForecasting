"""FastAPI application for Sales Forecasting."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.model import SalesForecastingModel
from src.preprocessing import SalesDataPreprocessor

app = FastAPI(
    title="Sales Forecasting API",
    description="API REST para predecir ventas futuras utilizando series temporales con StatsForecast.",
    version="1.0.0",
)

preprocessor = SalesDataPreprocessor()
model = SalesForecastingModel()

df = preprocessor.load_data("../data/sales_data.csv")
model.train(df)


class ForecastRequest(BaseModel):
    """Schema for forecast request."""

    days: int = Field(default=30, ge=1, le=365, description="Número de días a predecir (1-365)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "days": 30,
            }
        }
    }


class PredictionItem(BaseModel):
    """Schema for individual prediction."""

    date: str
    predicted_sales: float
    lower_bound: float
    upper_bound: float


class ForecastSummary(BaseModel):
    """Schema for forecast summary."""

    total_predicted: float
    average_daily: float
    period_days: int


class ForecastResponse(BaseModel):
    """Schema for forecast response."""

    predictions: list[PredictionItem]
    summary: ForecastSummary


@app.get("/", response_model=dict[str, str])
def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "Sales Forecasting API", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=ForecastResponse)
def predict_sales(request: ForecastRequest) -> ForecastResponse:
    """Generate sales forecasts."""
    result = model.predict(horizon=request.days)
    
    predictions = [
        PredictionItem(**pred) for pred in result.predictions
    ]
    
    return ForecastResponse(
        predictions=predictions,
        summary=ForecastSummary(**result.summary),
    )


@app.get("/data-quality")
def get_data_quality() -> dict[str, Any]:
    """Get data quality report."""
    report = preprocessor.get_data_quality_report(df)
    return report.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
