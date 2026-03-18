import pytest
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    df = pd.read_csv('data/sales_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'ds', 'sales': 'y'})
    df['unique_id'] = 'sales'
    return df

class TestDataLoading:
    def test_data_file_exists(self):
        assert os.path.exists('data/sales_data.csv'), "Data file not found"
    
    def test_data_not_empty(self):
        df = load_data()
        assert len(df) > 0, "Data is empty"
    
    def test_data_has_required_columns(self):
        df = load_data()
        assert 'ds' in df.columns, "Missing date column"
        assert 'y' in df.columns, "Missing sales column"
    
    def test_date_format(self):
        df = load_data()
        assert pd.api.types.is_datetime64_any_dtype(df['ds']), "Date should be datetime"

class TestDataQuality:
    def test_no_missing_values(self):
        df = load_data()
        assert df['y'].isna().sum() == 0, "Missing values in sales"
    
    def test_positive_sales(self):
        df = load_data()
        assert (df['y'] > 0).all(), "All sales should be positive"

class TestModel:
    def test_statsforecast_model_training(self):
        df = load_data()
        train = df.iloc[:int(len(df)*0.8)]
        
        models = [AutoARIMA(season_length=7)]
        sf = StatsForecast(models=models, freq='D')
        sf.fit(train[['ds', 'y', 'unique_id']])
        
        assert sf is not None, "Model should be trained"
    
    def test_statsforecast_predictions(self):
        df = load_data()
        train = df.iloc[:int(len(df)*0.8)]
        test = df.iloc[int(len(df)*0.8):]
        
        models = [AutoARIMA(season_length=7)]
        sf = StatsForecast(models=models, freq='D')
        sf.fit(train[['ds', 'y', 'unique_id']])
        
        forecast = sf.predict(h=len(test))
        forecast = forecast.reset_index()
        
        predictions = forecast['AutoARIMA'].values
        
        mae = mean_absolute_error(test['y'].values, predictions)
        rmse = np.sqrt(mean_squared_error(test['y'].values, predictions))
        
        assert mae < 10000, f"MAE too high: {mae}"
        assert rmse < 15000, f"RMSE too high: {rmse}"

class TestForecast:
    def test_future_predictions(self):
        df = load_data()
        
        models = [AutoARIMA(season_length=7)]
        sf = StatsForecast(models=models, freq='D')
        sf.fit(df[['ds', 'y', 'unique_id']])
        
        forecast = sf.predict(h=30, level=[95])
        forecast = forecast.reset_index()
        
        assert len(forecast) == 30, "Should have 30 predictions"
        assert 'AutoARIMA' in forecast.columns, "Missing prediction column"
        assert forecast['AutoARIMA'].notna().all(), "Missing predictions"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
