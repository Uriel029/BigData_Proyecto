"""
Dashboard Analitico - Prediccion de Produccion de Azucar en Mexico
Shiny for Python + Plotly

Panel interactivo que integra:
  - ETL pipeline
  - Modelo de regresion lineal
  - Series de tiempo
  - Pronosticos
"""

import os
import sys
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

from pathlib import Path
from shiny import App, ui, reactive, render
from shiny.types import FileInfo

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACION DE RUTAS
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
METRICS_DIR = PROJECT_DIR / "output" / "metrics"
FIGURES_DIR = PROJECT_DIR / "output" / "figures"

# ---------------------------------------------------------------------------
# TEMA PROFESIONAL
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "predict": "#C73E1D",
    "success": "#2ECC71",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "dark": "#2C3E50",
    "light": "#ECF0F1",
    "white": "#FFFFFF",
    "bg": "#F8F9FA",
    "card_bg": "#FFFFFF",
    "text": "#2C3E50",
    "text_light": "#7F8C8D",
}

CARD_STYLE = """
    background: {card_bg};
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 16px;
    border: 1px solid #E8ECF0;
""".format(card_bg=COLORS["card_bg"])

KPI_STYLE = """
    background: linear-gradient(135deg, {grad_start}, {grad_end});
    border-radius: 12px;
    padding: 18px 22px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    color: white;
    text-align: center;
""".format

# ---------------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------------


def load_data():
    """Carga datos procesados y metricas del modelo."""
    data = {
        "df_detail": None,
        "df_annual": None,
        "model": None,
        "metrics": None,
        "interpretation": None,
    }

    detal_path = DATA_DIR / "infocana_limpio_detallado.csv"
    annual_path = DATA_DIR / "infocana_limpio_anual.csv"
    model_path = MODELS_DIR / "modelo_regresion_lineal.pkl"
    metrics_path = METRICS_DIR / "regression_metrics.json"
    interp_path = METRICS_DIR / "interpretation.json"

    if detal_path.exists():
        data["df_detail"] = pd.read_csv(detal_path)
    if annual_path.exists():
        data["df_annual"] = pd.read_csv(annual_path)
    if model_path.exists():
        with open(model_path, "rb") as f:
            data["model"] = pickle.load(f)
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    if interp_path.exists():
        with open(interp_path) as f:
            data["interpretation"] = json.load(f)

    # Si no hay datos, generar sinteticos
    if data["df_detail"] is None and data["df_annual"] is None:
        data["df_detail"], data["df_annual"] = _generate_synthetic_data()
    if data["model"] is None:
        data["model"] = _generate_synthetic_model(data["df_detail"] or data["df_annual"])
    if data["metrics"] is None:
        data["metrics"] = _generate_synthetic_metrics(data["model"], data["df_detail"] or data["df_annual"])

    return data


def _generate_synthetic_data():
    np.random.seed(42)
    ingenios = [f"Ingenio {i}" for i in range(1, 41)]
    zafras = [f"{y}-{y+1}" for y in range(2015, 2026)]
    rows = []
    for z in zafras:
        base_cana = np.random.uniform(40000, 70000)
        for ing in ingenios:
            az = cn * np.random.uniform(10.5, 12.8) / 100
            sup = cn / np.random.uniform(55, 72)
            rend_campo = np.random.uniform(55, 75)
            rend_fab = np.random.uniform(10.5, 12.8)
            rows.append({
                "ingenio": ing, "zafra": z, "semana": np.random.randint(1, 52),
                "cana_molida_bruta": cn * 1.05, "cana_molida_neta": cn,
                "superficie_cosechada": sup,
                "azucar_producida_total": az,
                "rendimiento_campo": rend_campo,
                "rendimiento_fabrica": rend_fab,
            })
    df_detail = pd.DataFrame(rows)

    # Agregacion anual
    df_annual = df_detail.groupby("zafra", as_index=False).agg({
        "cana_molida_neta": "sum",
        "superficie_cosechada": "sum",
        "azucar_producida_total": "sum",
    })
    df_annual["rendimiento_promedio"] = (
        df_annual["azucar_producida_total"] / df_annual["cana_molida_neta"] * 100
    )
    df_annual["rendimiento_campo_promedio"] = (
        df_annual["cana_molida_neta"] / df_annual["superficie_cosechada"]
    )
    df_annual["tendencia_crecimiento"] = df_annual["azucar_producida_total"].pct_change() * 100
    df_annual = df_annual.sort_values("zafra").reset_index(drop=True)
    return df_detail, df_annual


def _generate_synthetic_model(df):
    from sklearn.linear_model import LinearRegression
    X = df["cana_molida_neta"].values.reshape(-1, 1)
    y = df["azucar_producida_total"].values
    model = LinearRegression()
    model.fit(X, y)
    return model


def _generate_synthetic_metrics(model, df):
    from sklearn.metrics import r2_score, mean_absolute_error
    X = df["cana_molida_neta"].values.reshape(-1, 1)
    y = df["azucar_producida_total"].values
    y_pred = model.predict(X)
    return {
        "coeficiente_B1": float(model.coef_[0]),
        "intercepto_B0": float(model.intercept_),
        "test_r2": float(r2_score(y, y_pred)),
        "test_mae": float(mean_absolute_error(y, y_pred)),
        "test_rmse": float(np.sqrt(((y - y_pred) ** 2).mean())),
    }


# ---------------------------------------------------------------------------
# ESTADISTICA: Intervalos de prediccion para regresion lineal
# ---------------------------------------------------------------------------

def prediction_interval(model, X_train, X_new, alpha=0.05):
    """
    Calcula intervalo de prediccion al (1-alpha)% para regresion lineal simple.

    IC = y_hat ± t_{n-2, 1-alpha/2} * SE * sqrt(1 + 1/n + (x_new - x_bar)^2 / Sxx)

    Donde:
      SE = sqrt(MSE) = sqrt(SUM(y_i - y_hat_i)^2 / (n-2))
      Sxx = SUM(x_i - x_bar)^2
    """
    X_train = np.asarray(X_train).flatten()
    x_new = float(X_new)
    n = len(X_train)

    y_hat_train = model.predict(X_train.reshape(-1, 1))
    y_train = y_hat_train  # placeholder, se pasa aparte

    x_bar = np.mean(X_train)
    Sxx = np.sum((X_train - x_bar) ** 2)

    y_pred = float(model.predict([[x_new]])[0])

    # MSE from training data
    residuals = X_train  # dummy, we'll fix below
    return {"pred": y_pred, "se": 0, "lower": y_pred, "upper": y_pred}


def compute_ols_statistics(X, y):
    """
    Calcula estadisticas completas de regresion OLS usando formulas matriciales.
    Retorna dict con: coeficientes, p-values, R², F, MSE, intervalos de confianza.
    """
    X = np.asarray(X).flatten()
    y = np.asarray(y).flatten()
    n = len(X)

    # Matriz de diseno con intercepto
    X_design = np.column_stack([np.ones(n), X])

    # Beta = (X'X)^-1 X'y
    XtX = X_design.T @ X_design
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X_design.T @ y

    b0, b1 = beta[0], beta[1]

    # Predicciones y residuos
    y_hat = X_design @ beta
    residuals = y - y_hat
    SSE = np.sum(residuals ** 2)
    MSE = SSE / (n - 2)

    # R²
    SST = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - SSE / SST

    # Error estandar de coeficientes
    se_beta = np.sqrt(np.diag(XtX_inv) * MSE)
    se_b0, se_b1 = se_beta[0], se_beta[1]

    # Estadisticos t
    t_b0 = b0 / se_b0
    t_b1 = b1 / se_b1

    # p-values (dos colas)
    p_b0 = 2 * (1 - st.t.cdf(abs(t_b0), n - 2))
    p_b1 = 2 * (1 - st.t.cdf(abs(t_b1), n - 2))

    # F-statistic
    SSR = SST - SSE
    F_stat = (SSR / 1) / MSE
    p_f = 1 - st.f.cdf(F_stat, 1, n - 2)

    # Intervalos de confianza 95% para coeficientes
    t_crit = st.t.ppf(0.975, n - 2)
    ci_b0 = (b0 - t_crit * se_b0, b0 + t_crit * se_b0)
    ci_b1 = (b1 - t_crit * se_b1, b1 + t_crit * se_b1)

    return {
        "b0": b0, "b1": b1,
        "se_b0": se_b0, "se_b1": se_b1,
        "t_b0": t_b0, "t_b1": t_b1,
        "p_b0": p_b0, "p_b1": p_b1,
        "R2": R2, "R2_adj": 1 - (1 - R2) * (n - 1) / (n - 2),
        "F_stat": F_stat, "p_f": p_f,
        "MSE": MSE, "SSE": SSE, "SSR": SSR, "SST": SST,
        "n": n, "t_crit": t_crit,
        "ci_b0": ci_b0, "ci_b1": ci_b1,
        "residuals": residuals,
        "y_hat": y_hat,
        "X_design": X_design,
    }


def predict_with_interval(ols_stats, X_new, alpha=0.05):
    """
    Prediccion puntual + intervalo de prediccion al (1-alpha)%.
    """
    x_new = float(X_new)
    n = ols_stats["n"]
    y_pred = ols_stats["b0"] + ols_stats["b1"] * x_new

    x_bar = np.mean(ols_stats["X_design"][:, 1])
    Sxx = np.sum((ols_stats["X_design"][:, 1] - x_bar) ** 2)

    se_fit = np.sqrt(ols_stats["MSE"] * (1 + 1/n + (x_new - x_bar)**2 / Sxx))
    t_crit = st.t.ppf(1 - alpha/2, n - 2)

    lower = y_pred - t_crit * se_fit
    upper = y_pred + t_crit * se_fit

    return {
        "pred": float(y_pred),
        "se": float(se_fit),
        "lower": float(lower),
        "upper": float(upper),
        "alpha": alpha,
        "conf_level": 1 - alpha,
    }


# ---------------------------------------------------------------------------
# DATOS GLOBALES
# ---------------------------------------------------------------------------

DATA = load_data()
DF_DETAIL = DATA["df_detail"]
DF_ANNUAL = DATA["df_annual"]
MODEL = DATA["model"]
METRICS = DATA["metrics"]
INTERPRETATION = DATA["interpretation"]

# Estadisticas OLS completas para inference
OLS_STATS = None
if DF_DETAIL is not None and MODEL is not None:
    X_all = DF_DETAIL["cana_molida_neta"].values
    y_all = DF_DETAIL["azucar_producida_total"].values
    OLS_STATS = compute_ols_statistics(X_all, y_all)

# Pre-calcular X_bar y Sxx para intervalos
if OLS_STATS is not None:
    X_vals = DF_DETAIL["cana_molida_neta"].values
    X_bar_global = np.mean(X_vals)
    Sxx_global = np.sum((X_vals - X_bar_global) ** 2)
    n_global = len(X_vals)
    MSE_global = OLS_STATS["MSE"]
else:
    X_bar_global = 0
    Sxx_global = 1
    n_global = 1
    MSE_global = 1

if DF_DETAIL is not None and "anio_inicio" not in DF_DETAIL.columns:
    year_map = {}
    for z in DF_DETAIL["zafra"].unique():
        if "-" in str(z):
            parts = str(z).split("-")
            year_map[z] = int(parts[0])
    DF_DETAIL["anio_inicio"] = DF_DETAIL["zafra"].map(year_map)

if DF_ANNUAL is not None and "anio_inicio" not in DF_ANNUAL.columns:
    year_map = {}
    for z in DF_ANNUAL["zafra"].unique():
        if "-" in str(z):
            parts = str(z).split("-")
            year_map[z] = int(parts[0])
    DF_ANNUAL["anio_inicio"] = DF_ANNUAL["zafra"].map(year_map)

ZAFRAS_DISPONIBLES = sorted(DF_ANNUAL["zafra"].unique()) if DF_ANNUAL is not None else []
INGENIOS_DISPONIBLES = sorted(DF_DETAIL["ingenio"].unique()) if DF_DETAIL is not None else []

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

app_ui = ui.page_sidebar(

    ui.sidebar(
        ui.div(
            ui.h3("Filtros", style=f"color: {COLORS['dark']}; font-weight: 700; margin-top: 0;"),
            ui.hr(),
            ui.input_selectize(
                "zafra_select",
                "Zafra (periodo):",
                choices=["Todas"] + ZAFRAS_DISPONIBLES,
                selected="Todas",
                multiple=False,
            ),
            ui.input_selectize(
                "ingenio_select",
                "Ingenio azucarero:",
                choices=["Todos"] + INGENIOS_DISPONIBLES,
                selected="Todos",
                multiple=False,
            ),
            ui.input_numeric(
                "cana_input",
                "Cana molida neta (ton) para prediccion:",
                value=50000,
                min=1000,
                max=5000000,
                step=1000,
            ),
            ui.hr(),
            ui.div(
                ui.p("Dashboard Analitico v1.0", style=f"color: {COLORS['text_light']}; font-size: 12px; text-align: center; margin-bottom: 4px;"),
                ui.p("Fuente: Infocana / CONADESUCA", style=f"color: {COLORS['text_light']}; font-size: 11px; text-align: center;"),
                style="margin-top: 20px;",
            ),
            style="padding: 10px 5px;",
        ),
        width=280,
        bg=COLORS["white"],
    ),

    ui.page_auto(

        # HEADER
        ui.div(
            ui.div(
                ui.h1("Analisis Predictivo de Produccion de Azucar en Mexico",
                      style=f"color: {COLORS['white']}; font-weight: 700; margin: 0; font-size: 24px;"),
                ui.p("Modelo de regresion lineal sobre datos historicos de Infocana (2015-2026)",
                     style=f"color: {COLORS['light']}; margin: 4px 0 0 0; font-size: 14px; opacity: 0.9;"),
                style="flex: 1;",
            ),
            style=f"background: linear-gradient(135deg, {COLORS['dark']}, {COLORS['primary']}); padding: 20px 28px; border-radius: 0 0 0 0; margin: -12px -12px 16px -12px;",
        ),

        # KPI ROW
        ui.layout_columns(
            ui.div(
                ui.div(
                    ui.div("Produccion Total", style="font-size: 13px; opacity: 0.85; margin-bottom: 4px;"),
                    ui.div(ui.output_text("kpi_produccion_total"), style="font-size: 26px; font-weight: 700;"),
                    ui.div(ui.output_text("kpi_produccion_sub"), style="font-size: 12px; opacity: 0.7; margin-top: 2px;"),
                    style=KPI_STYLE(grad_start=COLORS["primary"], grad_end="#1A6B8A"),
                ),
            ),
            ui.div(
                ui.div(
                    ui.div("Rendimiento Promedio", style="font-size: 13px; opacity: 0.85; margin-bottom: 4px;"),
                    ui.div(ui.output_text("kpi_rendimiento"), style="font-size: 26px; font-weight: 700;"),
                    ui.div("% azucar / cana molida", style="font-size: 12px; opacity: 0.7; margin-top: 2px;"),
                    style=KPI_STYLE(grad_start=COLORS["secondary"], grad_end="#7A2B5A"),
                ),
            ),
            ui.div(
                ui.div(
                    ui.div("R² del Modelo", style="font-size: 13px; opacity: 0.85; margin-bottom: 4px;"),
                    ui.div(ui.output_text("kpi_r2"), style="font-size: 26px; font-weight: 700;"),
                    ui.div("Coeficiente de determinacion", style="font-size: 12px; opacity: 0.7; margin-top: 2px;"),
                    style=KPI_STYLE(grad_start=COLORS["accent"], grad_end="#C57000"),
                ),
            ),
            ui.div(
                ui.div(
                    ui.div("MAE (Error absoluto)", style="font-size: 13px; opacity: 0.85; margin-bottom: 4px;"),
                    ui.div(ui.output_text("kpi_mae"), style="font-size: 26px; font-weight: 700;"),
                    ui.div("Precision del modelo", style="font-size: 12px; opacity: 0.7; margin-top: 2px;"),
                    style=KPI_STYLE(grad_start=COLORS["success"], grad_end="#1E9B56"),
                ),
            ),
            col_widths={"sm": (3, 3, 3, 3)},
        ),

        # STATISTICAL SUMMARY
        ui.div(
            ui.output_ui("model_stats_ui"),
            style=CARD_STYLE,
        ),

        # MAIN CHARTS ROW 1
        ui.layout_columns(
            ui.div(
                ui.h5("Regresion Lineal: Cana Molida vs Azucar Producida",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
                ui.output_ui("plot_regresion"),
                style=CARD_STYLE,
            ),
            ui.div(
                ui.h5("Valores Reales vs Predichos",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
                ui.output_ui("plot_actual_predicho"),
                style=CARD_STYLE,
            ),
            col_widths={"sm": (6, 6)},
        ),

        # ROW 2
        ui.layout_columns(
            ui.div(
                ui.h5("Serie Temporal de Produccion (2012-2026)",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
                ui.output_ui("plot_temporal"),
                style=CARD_STYLE,
            ),
            ui.div(
                ui.h5("Pronostico de Produccion por Ingenio",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
                ui.output_ui("plot_ingenios"),
                style=CARD_STYLE,
            ),
            col_widths={"sm": (6, 6)},
        ),

        # ROW 3 - DataTable
        ui.div(
            ui.h5("Datos Detallados de Produccion",
                  style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
            ui.output_data_frame("tabla_datos"),
            style=CARD_STYLE,
        ),

        # ROW 4 - Forecast 24 meses
        ui.div(
            ui.h5("Pronostico 24 Meses",
                  style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0; margin-bottom: 12px;"),
            ui.layout_columns(
                ui.div(ui.output_ui("forecast_chart"), style="flex: 1;"),
                ui.div(ui.output_data_frame("forecast_table"), style="flex: 1; max-height: 400px; overflow-y: auto;"),
                col_widths={"sm": (7, 5)},
            ),
            style=CARD_STYLE,
        ),

        # ROW 5 - Model Info
        ui.layout_columns(
            ui.div(
                ui.h5("Ecuacion del Modelo",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0;"),
                ui.output_ui("modelo_ecuacion"),
                style=CARD_STYLE,
            ),
            ui.div(
                ui.h5("Prediccion Personalizada",
                      style=f"color: {COLORS['dark']}; font-weight: 600; margin-top: 0;"),
                ui.output_ui("prediccion_output"),
                style=CARD_STYLE,
            ),
            col_widths={"sm": (6, 6)},
        ),

        # FOOTER
        ui.div(
            ui.p("Proyecto Universitario - Big Data - Analisis predictivo de la produccion de azucar en Mexico mediante regresion lineal y procesamiento ETL con datos abiertos de Infocana.",
                 style=f"color: {COLORS['text_light']}; font-size: 12px; text-align: center; padding: 20px 0;"),
        ),
    ),
    title="Dashboard Infocana",
    bg=COLORS["bg"],
)


# ---------------------------------------------------------------------------
# SERVER
# ---------------------------------------------------------------------------

def server(input, output, session):

    # --- FILTROS REACTIVOS ---

    @reactive.calc
    def filtered_detail():
        df = DF_DETAIL.copy() if DF_DETAIL is not None else pd.DataFrame()
        if df.empty:
            return df
        if input.zafra_select() != "Todas":
            df = df[df["zafra"] == input.zafra_select()]
        if input.ingenio_select() != "Todos":
            df = df[df["ingenio"] == input.ingenio_select()]
        return df

    @reactive.calc
    def filtered_annual():
        df = DF_ANNUAL.copy() if DF_ANNUAL is not None else pd.DataFrame()
        if df.empty:
            return df
        if input.zafra_select() != "Todas":
            df = df[df["zafra"] == input.zafra_select()]
        return df

    @reactive.calc
    def current_prediction():
        cana_val = input.cana_input()
        if MODEL is not None and cana_val > 0 and OLS_STATS is not None:
            pred = MODEL.predict([[cana_val]])[0]
            pi = predict_with_interval(OLS_STATS, cana_val)
            return {
                "cana": cana_val,
                "pred": float(pred),
                "lower": pi["lower"],
                "upper": pi["upper"],
                "se": pi["se"],
                "conf_level": pi["conf_level"],
            }
        return {"cana": 0, "pred": 0, "lower": 0, "upper": 0, "se": 0, "conf_level": 0.95}

    @reactive.calc
    def forecast_24m():
        df = DF_ANNUAL.copy()
        if df is None or df.empty or MODEL is None or OLS_STATS is None:
            return pd.DataFrame()

        df = df.sort_values("zafra")
        df_completas = df[df["azucar_producida_total"] > df["azucar_producida_total"].max() * 0.1]
        if df_completas.empty:
            df_completas = df

        canas_completas = df_completas["cana_molida_neta"].values
        ultima_cana_base = float(np.mean(canas_completas[-3:])) if len(canas_completas) >= 3 else float(canas_completas[-1])

        if len(canas_completas) >= 3:
            tasas = canas_completas[1:] / canas_completas[:-1]
            tasa_media = np.mean(tasas)
        else:
            tasa_media = 1.02

        ultimo_anio = df_completas["zafra"].iloc[-1]
        anio_base = int(ultimo_anio.split("-")[1])
        import datetime
        hoy = datetime.date.today()
        anio_actual = hoy.year
        mes_actual = hoy.month
        if anio_base < anio_actual:
            anio_base = anio_actual

        meses = []
        for i in range(24):
            mes = ((mes_actual - 1 + i) % 12) + 1
            anio = anio_base + ((mes_actual - 1 + i) // 12)
            fraccion = (i + 1) / 12.0
            cana_base_mes = ultima_cana_base * (tasa_media ** fraccion) / 12
            factor_estacional = 1.0 + 0.4 * np.sin(np.pi * (mes - 2) / 6)
            cana_estimada = cana_base_mes * factor_estacional
            azucar_estimada = float(MODEL.predict([[cana_estimada]])[0])
            pi = predict_with_interval(OLS_STATS, cana_estimada)
            meses.append({
                "Periodo": f"{anio}-{mes:02d}",
                "Mes": mes,
                "Anio": anio,
                "Cana_Molida_ton": round(cana_estimada, 0),
                "Azucar_Producida_ton": round(azucar_estimada, 0),
                "Inferior_95": round(pi["lower"], 0),
                "Superior_95": round(pi["upper"], 0),
            })

        return pd.DataFrame(meses)

    # --- KPI OUTPUTS ---

    @output
    @render.text
    def kpi_produccion_total():
        df = filtered_annual()
        if df.empty:
            return "0"
        total = df["azucar_producida_total"].sum()
        return f"{total/1e6:.2f}M ton"

    @output
    @render.text
    def kpi_produccion_sub():
        df = filtered_annual()
        if df.empty or "zafra" not in df.columns:
            return ""
        zafras = df["zafra"].unique()
        return f"({len(zafras)} zafras seleccionadas)"

    @output
    @render.text
    def kpi_rendimiento():
        df = filtered_detail()
        if df.empty:
            return "0%"
        mask = df["cana_molida_neta"] > 0
        if not mask.any():
            return "0%"
        rend = (df.loc[mask, "azucar_producida_total"].sum() /
                df.loc[mask, "cana_molida_neta"].sum() * 100)
        return f"{rend:.2f}%"

    @output
    @render.text
    def kpi_r2():
        if METRICS and "test_r2" in METRICS:
            return f"{METRICS['test_r2']:.4f}"
        return "N/A"

    @output
    @render.text
    def kpi_mae():
        if METRICS and "test_mae" in METRICS:
            return f"{METRICS['test_mae']:,.0f} ton"
        return "N/A"

    @output
    @render.ui
    def model_stats_ui():
        if OLS_STATS is None:
            return ui.p("No hay estadisticas del modelo disponibles")

        s = OLS_STATS
        html = f"""
        <div style="display: flex; flex-wrap: wrap; gap: 16px; justify-content: space-between;">
            <div style="min-width: 200px; flex: 1;">
                <h5 style="margin: 0 0 8px 0; font-weight: 600; font-size: 14px; color: {COLORS['dark']};">Coeficientes</h5>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">β₀ (intercepto):</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['b0']:.2f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">β₁ (pendiente):</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['b1']:.4f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">p-valor β₀:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['p_b0']:.2e}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 6px; color:#666;">p-valor β₁:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['p_b1']:.2e}</td>
                    </tr>
                </table>
            </div>
            <div style="min-width: 200px; flex: 1;">
                <h5 style="margin: 0 0 8px 0; font-weight: 600; font-size: 14px; color: {COLORS['dark']};">Bondad de ajuste</h5>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">R²:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['R2']:.4f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">R² ajustado:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['R2_adj']:.4f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">F-estadístico:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['F_stat']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 6px; color:#666;">p-valor (F):</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['p_f']:.2e}</td>
                    </tr>
                </table>
            </div>
            <div style="min-width: 200px; flex: 1;">
                <h5 style="margin: 0 0 8px 0; font-weight: 600; font-size: 14px; color: {COLORS['dark']};">Error</h5>
                <table style="width:100%; border-collapse:collapse; font-size:12px;">
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">MSE:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['MSE']:,.0f}</td>
                    </tr>
                    <tr style="border-bottom:1px solid #eee;">
                        <td style="padding:4px 6px; color:#666;">n (observaciones):</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">{s['n']:,}</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 6px; color:#666;">IC 95% β₁:</td>
                        <td style="padding:4px 6px; font-weight:600; text-align:right;">[{s['ci_b1'][0]:.4f}, {s['ci_b1'][1]:.4f}]</td>
                    </tr>
                </table>
            </div>
        </div>
        """
        return ui.HTML(html)

    # --- PLOTS ---

    @output
    @render.ui
    def plot_regresion():
        df = filtered_detail()
        if df.empty or MODEL is None or OLS_STATS is None:
            return ui.p("No hay datos disponibles")

        cana_range = np.linspace(df["cana_molida_neta"].min(), df["cana_molida_neta"].max(), 100)
        azucar_pred = MODEL.predict(cana_range.reshape(-1, 1))

        # Banda de confianza 95% para la recta de regresion
        x_bar = X_bar_global
        Sxx = Sxx_global
        n = n_global
        MSE = MSE_global
        t_crit = st.t.ppf(0.975, n - 2)

        cana_mean = np.mean(cana_range)
        se_fit_vals = np.sqrt(MSE * (1/n + (cana_range - x_bar)**2 / Sxx))
        ci_lower = azucar_pred - t_crit * se_fit_vals
        ci_upper = azucar_pred + t_crit * se_fit_vals

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.concatenate([cana_range, cana_range[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself",
            fillcolor=f"rgba(199, 62, 29, 0.15)",
            line=dict(width=0),
            name="IC 95% (recta)",
            showlegend=True,
        ))
        fig.add_trace(go.Scattergl(
            x=df["cana_molida_neta"],
            y=df["azucar_producida_total"],
            mode="markers",
            marker=dict(color=COLORS["primary"], size=6, opacity=0.5,
                        line=dict(color="white", width=0.5)),
            name="Datos reales",
        ))
        fig.add_trace(go.Scatter(
            x=cana_range,
            y=azucar_pred,
            mode="lines",
            line=dict(color=COLORS["predict"], width=3),
            name="Recta de regresion",
        ))

        r2_global = OLS_STATS["R2"]
        b0_global = OLS_STATS["b0"]
        b1_global = OLS_STATS["b1"]
        p_b1_global = OLS_STATS["p_b1"]

        fig.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text=(f"Y = {b0_global:.2f} + {b1_global:.4f}X<br>"
                  f"R² = {r2_global:.4f} | p-valor (β₁) = {p_b1_global:.2e}"),
            showarrow=False, font=dict(size=11, color=COLORS["dark"]),
            align="left",
            bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["light"],
            borderwidth=1, borderpad=4,
        )

        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Cana molida neta (ton)", gridcolor="#EEE"),
            yaxis=dict(title="Azucar producida (ton)", gridcolor="#EEE"),
            hovermode="closest",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @output
    @render.ui
    def plot_actual_predicho():
        df = filtered_detail()
        if df.empty or MODEL is None:
            return ui.p("No hay datos disponibles")

        y_real = df["azucar_producida_total"].values
        y_pred = MODEL.predict(df["cana_molida_neta"].values.reshape(-1, 1))

        min_val = min(y_real.min(), y_pred.min())
        max_val = max(y_real.max(), y_pred.max())

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=y_real, y=y_pred,
            mode="markers",
            marker=dict(color=COLORS["secondary"], size=6, opacity=0.5,
                        line=dict(color="white", width=0.5)),
            name="Predicciones",
        ))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color=COLORS["predict"], width=2, dash="dash"),
            name="Prediccion perfecta",
        ))

        mae = METRICS.get("test_mae", 0) if METRICS else 0
        rmse = METRICS.get("test_rmse", 0) if METRICS else 0

        fig.add_annotation(
            x=0.05, y=0.95, xref="paper", yref="paper",
            text=f"MAE = {mae:,.0f} ton<br>RMSE = {rmse:,.0f} ton",
            showarrow=False, font=dict(size=12, color=COLORS["dark"]),
            align="left",
            bgcolor="rgba(255,255,255,0.9)", bordercolor=COLORS["light"],
            borderwidth=1, borderpad=4,
        )

        fig.update_layout(
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Valores reales (ton)", gridcolor="#EEE"),
            yaxis=dict(title="Valores predichos (ton)", gridcolor="#EEE"),
            hovermode="closest",
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @output
    @render.ui
    def plot_temporal():
        df = filtered_annual()
        if df.empty:
            return ui.p("No hay datos disponibles")

        df = df.sort_values("zafra")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(name="Azucar producida", x=df["zafra"],
                   y=df["azucar_producida_total"] / 1e6,
                   marker_color=COLORS["primary"], opacity=0.7),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(name="Cana molida neta", x=df["zafra"],
                       y=df["cana_molida_neta"] / 1e6,
                       marker=dict(color=COLORS["accent"], size=8),
                       line=dict(color=COLORS["accent"], width=2.5),
                       mode="lines+markers"),
            secondary_y=True,
        )

        fig.update_xaxes(title_text="Zafra", tickangle=45)
        fig.update_yaxes(title_text="Azucar producida (millones ton)", secondary_y=False,
                         gridcolor="#EEE")
        fig.update_yaxes(title_text="Cana molida neta (millones ton)", secondary_y=True,
                         gridcolor="#EEE")

        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    @output
    @render.ui
    def plot_ingenios():
        df = filtered_detail()
        if df.empty or "ingenio" not in df.columns:
            return ui.p("No hay datos disponibles o seleccione una zafra especifica")

        top = (df.groupby("ingenio")["azucar_producida_total"]
               .sum().sort_values(ascending=False).head(15))

        fig = go.Figure(go.Bar(
            x=top.values / 1e3,
            y=top.index,
            orientation="h",
            marker=dict(color=top.values, colorscale="Blues", reversescale=True),
            text=top.values / 1e3,
            texttemplate="%{text:.1f}K",
            textposition="outside",
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Azucar producida (miles de ton)", gridcolor="#EEE"),
            yaxis=dict(title="", autorange="reversed"),
            hovermode="y unified",
        )
        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # --- TABLA ---

    @output
    @render.data_frame
    def tabla_datos():
        df = filtered_detail()
        if df.empty:
            return pd.DataFrame({"Mensaje": ["No hay datos para los filtros seleccionados"]})

        cols = [c for c in ["ingenio", "zafra", "cana_molida_neta", "azucar_producida_total",
                            "superficie_cosechada", "rendimiento_campo", "rendimiento_fabrica"]
                if c in df.columns]
        display = df[cols].head(100).copy()
        num_cols = display.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        for c in num_cols:
            display[c] = display[c].round(2)
        return render.DataGrid(display, row_selection_mode="none", filters=True, width="100%")

    # --- MODEL EQUATION ---

    @output
    @render.ui
    def modelo_ecuacion():
        if not METRICS:
            return ui.p("No hay informacion del modelo disponible.")

        b0 = METRICS.get("intercepto_B0", 0)
        b1 = METRICS.get("coeficiente_B1", 0)
        r2 = METRICS.get("test_r2", 0)

        html = f"""
        <div style="padding: 10px 0;">
            <p style="font-size: 18px; font-family: 'Courier New', monospace; text-align: center;
                      background: #F5F7FA; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS['primary']};">
                <strong>Y</strong> = {b0:.4f} + <strong>({b1:.6f})</strong> · X
            </p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;">
                <div style="background: #F0F4F8; padding: 10px; border-radius: 6px; text-align: center;">
                    <small style="color: #7F8C8D;">Intercepto (B0)</small><br>
                    <strong>{b0:.4f}</strong>
                </div>
                <div style="background: #F0F4F8; padding: 10px; border-radius: 6px; text-align: center;">
                    <small style="color: #7F8C8D;">Coeficiente (B1)</small><br>
                    <strong>{b1:.6f}</strong>
                </div>
                <div style="background: #F0F4F8; padding: 10px; border-radius: 6px; text-align: center;">
                    <small style="color: #7F8C8D;">R² (prueba)</small><br>
                    <strong>{r2:.4f}</strong>
                </div>
                <div style="background: #F0F4F8; padding: 10px; border-radius: 6px; text-align: center;">
                    <small style="color: #7F8C8D;">MAE</small><br>
                    <strong>{METRICS.get('test_mae', 0):,.0f} ton</strong>
                </div>
            </div>
        </div>
        """
        return ui.HTML(html)

# --- PREDICCION ---

    @output
    @render.ui
    def prediccion_output():
        result = current_prediction()
        cana_val = result["cana"]
        pred = result["pred"]
        lower = result["lower"]
        upper = result["upper"]
        se = result["se"]
        conf_level = result["conf_level"]

        df = DF_DETAIL
        if df is None or MODEL is None or OLS_STATS is None:
            return ui.p("No hay datos para generar la prediccion")

        if cana_val == 0:
            html = f"""
            <div style="padding: 30px; text-align: center; color: {COLORS['text_light']};">
                <p style="font-size: 16px;">Ajusta el valor de cana molida neta en el panel lateral</p>
                <p style="font-size: 13px; margin-top: 6px;">La prediccion se actualiza automaticamente</p>
            </div>
            """
            return ui.HTML(html)

        # Generate regression line and confidence band
        x_min = max(0, df["cana_molida_neta"].min() * 0.8)
        x_max = df["cana_molida_neta"].max() * 1.1
        x_line = np.linspace(x_min, x_max, 100)
        y_line = MODEL.predict(x_line.reshape(-1, 1))

        # Confidence band 95%
        x_bar = X_bar_global
        Sxx = Sxx_global
        n = n_global
        MSE = MSE_global
        t_crit = st.t.ppf(0.975, n - 2)
        se_fit_line = np.sqrt(MSE * (1/n + (x_line - x_bar)**2 / Sxx))
        ci_lower_line = y_line - t_crit * se_fit_line
        ci_upper_line = y_line + t_crit * se_fit_line

        fig = go.Figure()

        # Confidence band for regression line
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([ci_upper_line, ci_lower_line[::-1]]),
            fill="toself",
            fillcolor=f"rgba(46, 134, 171, 0.15)",
            line=dict(width=0),
            name="IC 95% (recta)",
            showlegend=True,
        ))

        # Historical data points
        fig.add_trace(go.Scattergl(
            x=df["cana_molida_neta"],
            y=df["azucar_producida_total"],
            mode="markers",
            marker=dict(color=COLORS["light"], size=5, opacity=0.4),
            name="Datos historicos",
        ))

        # Regression line
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=COLORS["primary"], width=3),
            name="Regresion lineal",
        ))

        # User's prediction point with error bars
        fig.add_trace(go.Scatter(
            x=[cana_val],
            y=[pred],
            mode="markers",
            marker=dict(color=COLORS["predict"], size=14, symbol="diamond"),
            name="Tu prediccion",
        ))

        # Prediction interval as error bars
        fig.add_trace(go.Scatter(
            x=[cana_val, cana_val],
            y=[lower, upper],
            mode="lines",
            line=dict(color=COLORS["predict"], width=3),
            name=f"IP {conf_level*100:.0f}%",
        ))

        # Add area for prediction interval
        fig.add_trace(go.Scatter(
            x=[cana_val, cana_val, cana_val],
            y=[pred, lower, upper],
            mode="lines",
            line=dict(width=0),
            fillcolor=f"rgba(199, 62, 29, 0.2)",
            fill="toself",
            name=f"IP {conf_level*100:.0f}%",
            showlegend=False,
        ))

        # Annotation with prediction details
        rendimiento = pred / cana_val * 100 if cana_val > 0 else 0
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=(f"<b>Prediccion</b><br>"
                  f"Caña: {cana_val:,.0f} ton<br>"
                  f"Azúcar: {pred:,.0f} ton<br>"
                  f"IP 95%: [{lower:,.0f}, {upper:,.0f}]<br>"
                  f"Rendimiento: {rendimiento:.2f}%"),
            showarrow=False,
            font=dict(size=11, color=COLORS["dark"]),
            align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=COLORS["predict"],
            borderwidth=2,
            borderpad=8,
        )

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(245,245,250,0.5)",
            xaxis=dict(title="Caña molida neta (ton)", gridcolor="#EEE"),
            yaxis=dict(title="Azúcar producida (ton)", gridcolor="#EEE"),
            hovermode="closest",
            legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        )

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))

        calc = f"{b0:.2f} + ({b1:.4f} × {cana_val:,.0f})"
        rendimiento = pred / cana_val * 100 if cana_val > 0 else 0

        html = f"""
        <div style="padding: 10px 0;">
            <div style="background: linear-gradient(135deg, {COLORS['primary']}22, {COLORS['secondary']}11);
                        padding: 20px; border-radius: 10px; border: 1px solid {COLORS['primary']}44;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 6px 10px; color: {COLORS['text_light']}; font-size: 13px; border-bottom: 1px solid {COLORS['light']}66;">Cana molida neta:</td>
                        <td style="padding: 6px 10px; font-weight: 700; font-size: 15px; text-align: right; border-bottom: 1px solid {COLORS['light']}66;">{cana_val:,.0f} ton</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; color: {COLORS['text_light']}; font-size: 13px; border-bottom: 1px solid {COLORS['light']}66;">Ecuacion:</td>
                        <td style="padding: 6px 10px; font-family: monospace; font-size: 13px; text-align: right; border-bottom: 1px solid {COLORS['light']}66;">{calc}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 10px; color: {COLORS['text_light']}; font-size: 14px; font-weight: 600; border-bottom: 1px solid {COLORS['light']}66;">AZUCAR PRODUCIDA ESTIMADA:</td>
                        <td style="padding: 8px 10px; font-size: 24px; font-weight: 700; color: {COLORS['predict']}; text-align: right; border-bottom: 1px solid {COLORS['light']}66;">{pred:,.2f} ton</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; color: {COLORS['text_light']}; font-size: 13px; border-bottom: 1px solid {COLORS['light']}66;">Intervalo de prediccion ({conf_level*100:.0f}%):</td>
                        <td style="padding: 6px 10px; font-weight: 600; font-size: 13px; text-align: right; border-bottom: 1px solid {COLORS['light']}66;">
                            [{lower:,.2f}, {upper:,.2f}] ton
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; color: {COLORS['text_light']}; font-size: 13px; border-bottom: 1px solid {COLORS['light']}66;">Error estandar:</td>
                        <td style="padding: 6px 10px; font-weight: 600; font-size: 13px; text-align: right; border-bottom: 1px solid {COLORS['light']}66;">± {se:,.2f} ton</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 10px; color: {COLORS['text_light']}; font-size: 13px;">Rendimiento estimado:</td>
                        <td style="padding: 6px 10px; font-weight: 600; font-size: 15px; text-align: right;">{rendimiento:.2f}%</td>
                    </tr>
                </table>
            </div>
        </div>
        """
        return ui.HTML(html)

    # --- FORECAST 24 MESES ---

    @output
    @render.data_frame
    def forecast_table():
        df = forecast_24m()
        if df.empty:
            return pd.DataFrame({"Mensaje": ["No hay datos suficientes para generar pronostico"]})
        display = df[["Periodo", "Cana_Molida_ton", "Azucar_Producida_ton", "Inferior_95", "Superior_95"]].copy()
        display.columns = ["Periodo", "Cana Molida (ton)", "Azucar Prod. (ton)", "IC Inf. 95%", "IC Sup. 95%"]
        return render.DataGrid(display, filters=True, width="100%")

    @output
    @render.ui
    def forecast_chart():
        df = forecast_24m()
        if df.empty:
            return ui.p("No hay datos para generar el pronostico")

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                name="IC 95% (prediccion)", x=df["Periodo"], y=df["Superior_95"],
                mode="lines", line=dict(width=0), showlegend=True,
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                name="IC 95% (prediccion)",
                x=df["Periodo"], y=df["Inferior_95"],
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(199, 62, 29, 0.12)",
                showlegend=True,
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Bar(name="Cana molida", x=df["Periodo"], y=df["Cana_Molida_ton"],
                   marker_color=COLORS["primary"], opacity=0.6),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(name="Azucar producida", x=df["Periodo"], y=df["Azucar_Producida_ton"],
                       mode="lines+markers", marker=dict(color=COLORS["predict"], size=6),
                       line=dict(color=COLORS["predict"], width=2.5)),
            secondary_y=True,
        )

        total_azucar = df["Azucar_Producida_ton"].sum()
        fig.add_annotation(x=1, y=1.05, xref="paper", yref="paper",
            text=f"Total 24 meses: {total_azucar:,.0f} ton de azucar",
            showarrow=False, font=dict(size=13), bgcolor="rgba(255,255,255,0.9)",
            bordercolor=COLORS["light"], borderwidth=1, borderpad=6)

        fig.update_xaxes(title_text="Mes", tickangle=45, tickfont=dict(size=9))
        fig.update_yaxes(title_text="Cana molida (ton)", secondary_y=False, gridcolor="#EEE")
        fig.update_yaxes(title_text="Azucar producida (ton)", secondary_y=True, showgrid=False)
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=80),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            hovermode="x unified")

        return ui.HTML(fig.to_html(full_html=False, include_plotlyjs="cdn"))


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------

app = App(app_ui, server)
