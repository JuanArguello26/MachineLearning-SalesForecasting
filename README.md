# Sales Forecasting API

API REST para predecir ventas futuras utilizando series temporales con StatsForecast.

## Descripción del Proyecto

Este proyecto implementa un modelo de predicción de ventas utilizando **StatsForecast** (AutoARIMA). Analiza patrones históricos de ventas y predice las ventas de los próximos 30 días.

## Stack Tecnológico

- **Python 3.10+**
- **FastAPI** - Framework web
- **StatsForecast** - Predicción de series temporales
- **Pandas/NumPy** - Análisis de datos
- **Matplotlib/Seaborn** - Visualizaciones

## Estructura del Proyecto

```
sales-forecasting/
├── api/
│   └── main.py          # API REST
├── data/
│   └── sales_data.csv   # Dataset
├── models/
│   └── *.csv           # Resultados guardados
├── notebooks/
│   └── eda.ipynb       # Análisis exploratorio
├── tests/
│   └── test_model.py   # Tests
├── requirements.txt
└── README.md
```

## Instalación

```bash
# Clonar el repositorio
git clone <repo-url>
cd sales-forecasting

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso Local

### Ejecutar la API

```bash
cd api
uvicorn main:app --reload
```

La API estará disponible en: `http://localhost:8000`

### Documentación Interactive

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Ejemplo de Predicción

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "days": 30
  }'
```

Respuesta esperada:
```json
{
  "predictions": [
    {"date": "2024-01-01", "predicted_sales": 25000, "lower_bound": 23000, "upper_bound": 27000},
    ...
  ],
  "summary": {
    "total_predicted": 750000,
    "average_daily": 25000,
    "period_days": 30
  }
}
```

## Ejecutar Tests

```bash
pytest tests/ -v
```

## Deployment

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Conectar repositorio a Railway
2. Establecer `PYTHON_VERSION` = `3.10`
3. Deploy automático

### Render

1. Conectar repositorio a Render
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| MAE | ~$1,500 - $3,000 |
| RMSE | ~$2,000 - $4,000 |
| MAPE | ~5% - 10% |

## Explicación del Modelo

**StatsForecast** es una librería optimizada para predicción de series temporales que:

- Utiliza **AutoARIMA** para selección automática de parámetros
- Es muy rápida (implementada en Numba)
- Soporta estacionalidad semanal
- Proporciona intervalos de confianza

## Dataset

El dataset contiene ventas diarias con las siguientes características:
- Período: 2023 (año completo)
- Rango de ventas: $12,500 - $30,000 diario
- Tendencia: Creciente a lo largo del año

## Licencia

MIT License
