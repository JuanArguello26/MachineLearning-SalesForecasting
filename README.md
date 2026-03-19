# Sales Forecasting API

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![CI](https://github.com/JuanArguello26/MachineLearning-SalesForecasting/actions/workflows/ci.yml/badge.svg)

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
│   └── main.py              # API REST con FastAPI
├── data/
│   └── sales_data.csv       # Dataset
├── models/
│   └── forecast_results.csv # Resultados de predicciones
├── notebooks/
│   └── eda.ipynb            # Análisis exploratorio
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuración del modelo
│   ├── preprocessing.py    # Preprocesamiento de datos
│   └── model.py            # Lógica de entrenamiento y predicción
├── tests/
│   ├── test_model.py        # Tests del modelo
│   └── test_api.py          # Tests de la API
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD
├── .pre-commit-config.yaml  # Pre-commit hooks
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/JuanArguello26/MachineLearning-SalesForecasting.git
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

### Documentación Interactiva

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

### Docker

```bash
docker build -t sales-forecasting .
docker run -p 8000:8000 sales-forecasting
```

### Railway

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

## Desarrollo

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

### Linting

```bash
ruff check .
```

## Licencia

MIT License
