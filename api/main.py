from fastapi import FastAPI
from pydantic import BaseModel
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd

app = FastAPI(title="Sales Forecasting API", description="API para predecir ventas futuras")

models = [AutoARIMA(season_length=7)]
sf = StatsForecast(models=models, freq='D')

df = pd.read_csv('../data/sales_data.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'date': 'ds', 'sales': 'y'})
df['unique_id'] = 'sales'

sf.fit(df[['ds', 'y', 'unique_id']])

class ForecastRequest(BaseModel):
    days: int = 30

@app.get("/")
def root():
    return {"message": "Sales Forecasting API", "status": "running"}

@app.post("/predict")
def predict_sales(request: ForecastRequest):
    forecast = sf.predict(h=request.days, level=[95])
    forecast = forecast.reset_index()
    
    predictions = []
    for _, row in forecast.iterrows():
        predictions.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_sales": round(row['AutoARIMA'], 2),
            "lower_bound": round(row['AutoARIMA-lo-95'], 2),
            "upper_bound": round(row['AutoARIMA-hi-95'], 2)
        })
    
    return {
        "predictions": predictions,
        "summary": {
            "total_predicted": round(forecast['AutoARIMA'].sum(), 2),
            "average_daily": round(forecast['AutoARIMA'].mean(), 2),
            "period_days": request.days
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
