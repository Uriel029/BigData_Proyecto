"""
MODELO DE REGRESION LINEAL + SERIES DE TIEMPO
Infocana: Prediccion de produccion de azucar en Mexico

Modelo: Y = B0 + B1*X
  Y = azucar_producida_total
  X = cana_molida_neta

Incluye:
  - Regresion lineal con sklearn
  - Metricas: MAE, RMSE, R2
  - Visualizaciones profesionales
  - Pronostico temporal
  - Serializacion del modelo
"""

import os
import sys
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
FIGURES_DIR = os.path.join(PROJECT_DIR, "output", "figures")
METRICS_DIR = os.path.join(PROJECT_DIR, "output", "metrics")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Paleta de colores profesional
COLOR_PRIMARY = "#2E86AB"
COLOR_SECONDARY = "#A23B72"
COLOR_ACCENT = "#F18F01"
COLOR_PREDICT = "#C73E1D"
BG_COLOR = "#F8F9FA"


# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------

def load_clean_data() -> pd.DataFrame:
    """Carga el dataset limpio del ETL."""
    anual_path = os.path.join(PROCESSED_DIR, "infocana_limpio_anual.csv")
    detal_path = os.path.join(PROCESSED_DIR, "infocana_limpio_detallado.csv")

    if os.path.exists(anual_path):
        df_annual = pd.read_csv(anual_path)
        df_detail = pd.read_csv(detal_path) if os.path.exists(detal_path) else None
        print(f"[DATA] Cargados {len(df_annual)} registros anuales")
        return df_annual, df_detail

    # Si no existe, generar sinteticos
    print("[DATA] No se encontraron datos procesados. Generando sinteticos...")
    return _generate_annual_data(), _generate_detail_data()


def _generate_annual_data() -> pd.DataFrame:
    """Genera datos anuales sinteticos para demostracion."""
    np.random.seed(42)
    zafras = [f"{y}-{y+1}" for y in range(2012, 2026)]
    n = len(zafras)
    cana_base = np.linspace(45_000_000, 55_000_000, n) + np.random.normal(0, 2_000_000, n)
    rend = np.linspace(11.2, 11.8, n) + np.random.normal(0, 0.3, n)
    azucar = cana_base * rend / 100
    sup = cana_base / np.random.uniform(60, 70, n)
    df = pd.DataFrame({
        "zafra": zafras,
        "cana_molida_neta": cana_base,
        "superficie_cosechada": sup,
        "azucar_producida_total": azucar,
        "rendimiento_promedio": rend,
        "rendimiento_campo_promedio": cana_base / sup,
    })
    df["tendencia_crecimiento"] = df["azucar_producida_total"].pct_change() * 100
    return df


def _generate_detail_data() -> pd.DataFrame:
    """Genera datos detallados sinteticos."""
    np.random.seed(42)
    ingenios = [f"Ingenio {i}" for i in range(1, 41)]
    zafras = [f"{y}-{y+1}" for y in range(2012, 2026)]
    rows = []
    for z in zafras:
        for ing in ingenios:
            cn = np.random.uniform(100_000, 800_000)
            az = cn * np.random.uniform(10.5, 12.8) / 100
            rows.append({"ingenio": ing, "zafra": z, "cana_molida_neta": cn,
                         "azucar_producida_total": az,
                         "superficie_cosechada": cn / np.random.uniform(55, 75)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. MODELO DE REGRESION LINEAL
# ---------------------------------------------------------------------------


def train_linear_regression(df: pd.DataFrame) -> dict:
    """
    Entrena modelo de regresion lineal.

    Y = azucar_producida_total (variable dependiente)
    X = cana_molida_neta (variable independiente)
    """
    print("\n" + "=" * 60)
    print("MODELO DE REGRESION LINEAL")
    print("=" * 60)

    # Usar datos detallados (por ingenio) para tener suficientes puntos
    if "ingenio" in df.columns:
        X = df["cana_molida_neta"].values.reshape(-1, 1)
        y = df["azucar_producida_total"].values
        data_label = "detallados (por ingenio)"
    else:
        X = df["cana_molida_neta"].values.reshape(-1, 1)
        y = df["azucar_producida_total"].values
        data_label = "anuales"

    print(f"[ML] Usando datos {data_label}: {len(X)} muestras")

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predecir
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metricas
    metrics = {
        "coeficiente_B1": float(model.coef_[0]),
        "intercepto_B0": float(model.intercept_),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "n_samples": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print(f"\n  Ecuacion: Y = {metrics['intercepto_B0']:.4f} + {metrics['coeficiente_B1']:.6f} * X")
    print(f"  R2 (entrenamiento): {metrics['train_r2']:.4f}")
    print(f"  R2 (prueba):        {metrics['test_r2']:.4f}")
    print(f"  MAE (prueba):       {metrics['test_mae']:,.2f} ton")
    print(f"  RMSE (prueba):      {metrics['test_rmse']:,.2f} ton")

    # Guardar metricas
    with open(os.path.join(METRICS_DIR, "regression_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "modelo_regresion_lineal.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[ML] Modelo guardado en: {model_path}")

    return {
        "model": model,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


# ---------------------------------------------------------------------------
# 3. VISUALIZACIONES
# ---------------------------------------------------------------------------


def plot_regression_line(model_output: dict, df_detail: pd.DataFrame):
    """Grafica de regresion: puntos reales + linea de regresion."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    X = df_detail["cana_molida_neta"].values
    y = df_detail["azucar_producida_total"].values

    ax.scatter(X, y, alpha=0.4, c=COLOR_PRIMARY, s=30, label="Datos reales", edgecolors="white", linewidth=0.5)

    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = model_output["model"].predict(x_line.reshape(-1, 1))
    ax.plot(x_line, y_line, color=COLOR_PREDICT, linewidth=2.5, label=f"Recta de regresion")

    ax.set_xlabel("Cana molida neta (ton)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Azucar producida total (ton)", fontsize=12, fontweight="bold")
    ax.set_title("Regresion Lineal: Cana Molida vs Azucar Producida", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, frameon=True, facecolor="white", edgecolor="gray")
    ax.grid(True, alpha=0.3)

    # Texto con ecuacion
    b0 = model_output["metrics"]["intercepto_B0"]
    b1 = model_output["metrics"]["coeficiente_B1"]
    r2 = model_output["metrics"]["test_r2"]
    textstr = f"Y = {b0:.2f} + {b1:.4f}X\nR² = {r2:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5",
            facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "regresion_lineal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Regresion lineal guardada: {path}")


def plot_scatter_actual_vs_predicted(model_output: dict):
    """Scatter plot de valores reales vs predichos."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.scatter(model_output["y_test"], model_output["y_test_pred"],
               alpha=0.5, c=COLOR_SECONDARY, s=40, edgecolors="white", linewidth=0.5)

    min_val = min(model_output["y_test"].min(), model_output["y_test_pred"].min())
    max_val = max(model_output["y_test"].max(), model_output["y_test_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1.5, alpha=0.7, label="Prediccion perfecta")

    ax.set_xlabel("Valores reales (ton)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Valores predichos (ton)", fontsize=12, fontweight="bold")
    ax.set_title("Valores Reales vs Predichos (Conjunto de Prueba)", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    r2 = model_output["metrics"]["test_r2"]
    mae = model_output["metrics"]["test_mae"]
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nMAE = {mae:,.0f} ton",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "actual_vs_predicho.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Actual vs predicho guardada: {path}")


def plot_prediccion_futura(model_output: dict, df_annual: pd.DataFrame):
    """Prediccion futura: extiende la tendencia a la siguiente zafra."""
    model = model_output["model"]
    zafras = df_annual["zafra"].tolist()
    valores_reales = df_annual["azucar_producida_total"].tolist()
    cana_vals = df_annual["cana_molida_neta"].tolist()

    # Pronosticar siguiente zafra
    ultima_cana = cana_vals[-1]
    tendencia_cana = np.mean([cana_vals[i] / cana_vals[i-1] for i in range(1, len(cana_vals))])
    siguiente_cana = ultima_cana * tendencia_cana
    siguiente_pred = model.predict([[siguiente_cana]])[0]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    x_idx = list(range(len(zafras)))
    ax.bar(x_idx, valores_reales, color=COLOR_PRIMARY, alpha=0.7, width=0.6, label="Produccion real")

    ax.bar(len(zafras), siguiente_pred, color=COLOR_PREDICT, alpha=0.7, width=0.6,
           label=f"Pronostico {len(zafras)+1}a zafra")

    ultima_zafra = zafras[-1]
    anio_inicio = int(ultima_zafra.split("-")[0])
    anio_fin = int(ultima_zafra.split("-")[1])
    etiquetas = zafras + [f"Pronostico\n{anio_inicio+1}-{anio_fin+1}"]
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Zafra", fontsize=12, fontweight="bold")
    ax.set_ylabel("Azucar producida total (ton)", fontsize=12, fontweight="bold")
    ax.set_title("Produccion Historica y Pronostico para la Siguiente Zafra", fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Anotar
    for i, v in enumerate(valores_reales):
        ax.text(i, v + max(valores_reales)*0.01, f"{v/1e6:.1f}M", ha="center", fontsize=8, fontweight="bold")
    ax.text(len(zafras), siguiente_pred + max(valores_reales)*0.01,
            f"{siguiente_pred/1e6:.1f}M", ha="center", fontsize=9, fontweight="bold", color=COLOR_PREDICT)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "prediccion_futura.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Prediccion futura guardada: {path}")

    return {"siguiente_cana": float(siguiente_cana), "siguiente_azucar": float(siguiente_pred),
            "tendencia_cana": float(tendencia_cana)}


def plot_time_series(df_annual: pd.DataFrame):
    """Serie temporal de produccion de azucar a traves de las zafras."""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax1.set_facecolor(BG_COLOR)

    zafras = df_annual["zafra"].tolist()
    x = range(len(zafras))

    color1 = COLOR_PRIMARY
    color2 = COLOR_ACCENT

    ax1.plot(x, df_annual["azucar_producida_total"].values / 1e6,
             color=color1, marker="o", linewidth=2.5, markersize=8, label="Azucar producida")
    ax1.fill_between(x, df_annual["azucar_producida_total"].values / 1e6,
                     alpha=0.15, color=color1)
    ax1.set_ylabel("Azucar producida (millones de ton)", fontsize=12, fontweight="bold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, df_annual["cana_molida_neta"].values / 1e6,
             color=color2, marker="s", linewidth=2, markersize=7, linestyle="--", label="Cana molida neta")
    ax2.set_ylabel("Cana molida neta (millones de ton)", fontsize=12, fontweight="bold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(zafras, rotation=45, ha="right", fontsize=9)
    ax1.set_xlabel("Zafra", fontsize=12, fontweight="bold")
    ax1.set_title("Evolucion Temporal: Produccion de Azucar y Cana Molida (2012-2026)",
                  fontsize=14, fontweight="bold", pad=15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "serie_temporal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Serie temporal guardada: {path}")


def plot_residuos(model_output: dict):
    """Distribucion de residuos del modelo de regresion."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_COLOR)

    residuos = model_output["y_test"] - model_output["y_test_pred"]

    # Histograma
    axes[0].set_facecolor(BG_COLOR)
    axes[0].hist(residuos, bins=25, color=COLOR_PRIMARY, edgecolor="white", alpha=0.7)
    axes[0].axvline(0, color=COLOR_PREDICT, linestyle="--", linewidth=2)
    axes[0].set_xlabel("Residuo (ton)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Frecuencia", fontsize=11, fontweight="bold")
    axes[0].set_title("Distribucion de Residuos", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot aproximado
    axes[1].set_facecolor(BG_COLOR)
    axes[1].scatter(model_output["y_test_pred"], residuos, alpha=0.5, c=COLOR_SECONDARY, s=30, edgecolors="white")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Valores predichos (ton)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Residuos (ton)", fontsize=11, fontweight="bold")
    axes[1].set_title("Residuos vs Valores Predichos", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "residuos.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Residuos guardada: {path}")


# ---------------------------------------------------------------------------
# 4. INTERPRETACION
# ---------------------------------------------------------------------------

def interpret_results(model_output: dict, forecast: dict, df_annual: pd.DataFrame):
    """Genera interpretacion textual de los resultados."""
    m = model_output["metrics"]
    df_annual = df_annual.sort_values("zafra")

    print("\n" + "=" * 60)
    print("INTERPRETACION DE RESULTADOS")
    print("=" * 60)

    print(f"""
1. ECUACION DEL MODELO:
   Y = {m['intercepto_B0']:.4f} + {m['coeficiente_B1']:.6f} * X
   
   donde:
   Y = azucar_producida_total (toneladas)
   X = cana_molida_neta (toneladas)

2. INTERPRETACION DEL COEFICIENTE:
   Por cada tonelada adicional de cana molida neta,
   la produccion de azucar aumenta en {m['coeficiente_B1']:.4f} toneladas.
   Esto representa el rendimiento promedio de extraccion.

3. METRICAS DE DESEMPENO:
   R² (prueba):  {m['test_r2']:.4f}  ({m['test_r2']*100:.2f}% de varianza explicada)
   MAE (prueba): {m['test_mae']:,.2f} toneladas
   RMSE (prueba): {m['test_rmse']:,.2f} toneladas

4. PRONOSTICO SIGUIENTE ZAFRA:
   Cana molida estimada: {forecast['siguiente_cana']:,.0f} ton
   Azucar producida estimada: {forecast['siguiente_azucar']:,.0f} ton
   Tendencia de cana molida: {forecast['tendencia_cana']*100-100:+.2f}%

5. ANALISIS TEMPORAL:
   Produccion promedio: {df_annual['azucar_producida_total'].mean():,.0f} ton
   Produccion minima: {df_annual['azucar_producida_total'].min():,.0f} ton ({df_annual.loc[df_annual['azucar_producida_total'].idxmin(), 'zafra']})
   Produccion maxima: {df_annual['azucar_producida_total'].max():,.0f} ton ({df_annual.loc[df_annual['azucar_producida_total'].idxmax(), 'zafra']})
   """)

    interpretation = {
        "ecuacion": f"Y = {m['intercepto_B0']:.4f} + {m['coeficiente_B1']:.6f} * X",
        "interpretacion_coeficiente": f"Por cada ton de cana molida, se producen {m['coeficiente_B1']:.4f} ton de azucar",
        "r2_interpretacion": f"El modelo explica el {m['test_r2']*100:.2f}% de la variabilidad en la produccion de azucar",
        "pronostico": forecast,
    }
    with open(os.path.join(METRICS_DIR, "interpretation.json"), "w") as f:
        json.dump(interpretation, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_modeling_pipeline():
    """Ejecuta la pipeline completa de modelado."""
    print("=" * 60)
    print("PIPELINE DE MODELADO - INFOCANA")
    print("=" * 60)

    df_annual, df_detail = load_clean_data()

    if df_detail is not None:
        model_output = train_linear_regression(df_detail)
        print("\n[VIZ] Generando visualizaciones...")
        plot_regression_line(model_output, df_detail)
        plot_scatter_actual_vs_predicted(model_output)
        forecast = plot_prediccion_futura(model_output, df_annual)
        plot_time_series(df_annual)
        plot_residuos(model_output)
        interpret_results(model_output, forecast, df_annual)
    else:
        model_output = train_linear_regression(df_annual)
        print("\n[VIZ] Generando visualizaciones...")
        plot_regression_line(model_output, df_annual)
        plot_scatter_actual_vs_predicted(model_output)
        forecast = plot_prediccion_futura(model_output, df_annual)
        plot_time_series(df_annual)
        plot_residuos(model_output)
        interpret_results(model_output, forecast, df_annual)

    print("\n" + "=" * 60)
    print("MODELADO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    return model_output


if __name__ == "__main__":
    run_modeling_pipeline()
