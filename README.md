# Analisis Predictivo de Produccion de Azucar en Mexico

Proyecto universitario de **Big Data** que implementa un pipeline ETL completo, un modelo de regresion lineal, analisis de series temporales, y un dashboard interactivo para predecir la produccion total de azucar en Mexico utilizando datos abiertos del sistema **Infocana (CONADESUCA)**.

---

## Estructura del Proyecto

```
proyecto_bigdata/
├── data/
│   ├── raw/                    # Archivos CSV descargados (2012-2026)
│   └── processed/              # Datos limpios y transformados
├── scripts/
│   ├── download_data.py        # Descarga de datos desde datos.gob.mx
│   └── etl_pipeline.py         # Pipeline ETL completo
├── models/
│   └── linear_regression.py    # Modelo de regresion lineal + series de tiempo
├── dashboard/
│   └── app.py                  # Dashboard con Shiny for Python + Plotly
├── output/
│   ├── figures/                # Graficas generadas
│   └── metrics/                # Metricas del modelo (JSON)
├── docs/
│   └── informe_final.md        # Documentacion completa del proyecto
├── requirements.txt            # Dependencias de Python
├── README.md                   # Este archivo
└── .gitignore
```

## Requisitos

- Python 3.10+
- pip

## Instalacion

```bash
# 1. Clonar o copiar el repositorio
cd proyecto_bigdata

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Ejecucion

### 1. Descargar datos

```bash
python scripts/download_data.py
```

### 2. Ejecutar pipeline ETL

```bash
python scripts/etl_pipeline.py
```

### 3. Entrenar modelo y generar graficas

```bash
python models/linear_regression.py
```

### 4. Lanzar dashboard

```bash
python -m shiny run dashboard/app.py
```

Abrir en el navegador: `http://localhost:8000`

## Dashboard

El dashboard incluye:
- **Filtros interactivos** por zafra e ingenio
- **4 indicadores KPI** (produccion, rendimiento, R², MAE)
- **Grafica de regresion** cana vs azucar
- **Scatter plot** reales vs predichos
- **Serie temporal** de produccion 2012-2026
- **Top ingenios** por produccion
- **Tabla de datos** filtrable
- **Prediccion personalizada** ingresando cana molida
- **Ecuacion del modelo** con interpretacion

## Datos

Fuente: [datos.gob.mx - Infocana](https://www.datos.gob.mx/dataset/avance_produccion_cana_azucar_infocana)

Institucion: Comite Nacional para el Desarrollo Sustentable de la Cana de Azucar (CONADESUCA)

Periodo: 2015-2016 a 2025-2026 (11 zafras)

## Modelo

Regresion lineal simple:
```
Y = β0 + β1·X
```
- **Y**: azucar_producida_total (toneladas) — variable dependiente
- **X**: cana_molida_neta (toneladas) — variable independiente
- **β0**: intercepto (~18,791 ton)
- **β1**: coeficiente (~0.0995, rendimiento de extraccion ~9.95%)

**Metricas (2015-2026):**
- R² prueba: **0.7684** (76.8% varianza explicada)
- MAE: **16,674 ton**
- RMSE: **33,154 ton**

## Tecnologias

- **Python**: pandas, numpy, scikit-learn, statsmodels
- **Visualizacion**: plotly, matplotlib, seaborn
- **Dashboard**: Shiny for Python
- **Datos**: Fuente abierta datos.gob.mx
