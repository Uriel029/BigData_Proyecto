"""
Pipeline ETL completo para datos de produccion de cana y azucar (Infocana 2012-2026).

Flujo:
  1. EXTRACT: Lee todos los CSV descargados
  2. TRANSFORM: Normaliza columnas, limpia nulos, deriva variables
  3. LOAD: Guarda dataset limpio en CSV, Pandas DataFrame y SQLite
"""

import os
import sys
import glob
import sqlite3
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. EXTRACT
# ---------------------------------------------------------------------------

def extract_raw_data(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Lee todos los CSV del directorio raw y los concatena en un solo DataFrame."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        print("[WARN] No se encontraron archivos CSV. Ejecuta primero download_data.py")
        # Usar datos sinteticos para demostracion
        return _generate_sample_data()

    frames = []
    for fpath in csv_files:
        zafra_label = os.path.basename(fpath).replace("infocana_", "").replace(".csv", "").replace("_", "-")
        print(f"[EXTRACT] Leyendo {os.path.basename(fpath)}")
        try:
            df = pd.read_csv(fpath, encoding="utf-8", low_memory=False)
            # Estandarizar nombres a minusculas
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
            frames.append(df)
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")

    if not frames:
        print("[WARN] No se pudo leer ningun archivo. Usando datos sinteticos.")
        return _generate_sample_data()

    df_raw = pd.concat(frames, ignore_index=True)
    print(f"[EXTRACT] Total registros combinados: {len(df_raw):,}")
    return df_raw


def _generate_sample_data() -> pd.DataFrame:
    """Genera datos sinteticos basados en los datos reales de Infocana para propositos de prueba."""
    np.random.seed(42)
    ingenios = [
        "Adolfo Lopez Mateos", "Alianza Popular", "Atencingo", "Bella Vista",
        "Central Motzorongo", "Cholula", "Ciudad Mendoza", "Cuatotolapam",
        "El Carmen", "El Molino", "El Potrero", "Eldorado",
        "Emiliano Zapata", "Gabriel", "Huixtla", "Jose Maria Martinez",
        "La Joya", "La Primavera", "Mahuixtlan", "Melchor Ocampo",
        "Nacional", "Plan de Ayala", "Plan de San Luis", "Pujiltic",
        "Queseria", "San Cristobal", "San Francisco", "San Jose de Abajo",
        "San Miguel", "San Pedro", "San Rafael", "Santa Clara",
        "Santa Rosalia", "Tala", "Tamazula", "Tres Valles",
        "Zapoapita", "La Gloria", "El Refugio", "San Nicolas"
    ]
    rows = []
    zafras = [
        "2015-2016", "2016-2017", "2017-2018", "2018-2019", "2019-2020",
        "2020-2021", "2021-2022", "2022-2023", "2023-2024", "2024-2025",
        "2025-2026",
    ]
    base_cana = {z: np.random.randint(1_500_000, 2_500_000) for z in zafras}
    rend_base = {z: np.random.uniform(10.5, 12.5) for z in zafras}

    for zafra in zafras:
        n_ingenios = np.random.randint(35, 40)
        selected = np.random.choice(ingenios, n_ingenios, replace=False)
        total_cana = base_cana[zafra]
        total_azucar = 0
        cana_asignada = np.random.dirichlet(np.ones(n_ingenios)) * total_cana
        for i, ing in enumerate(selected):
            cn = cana_asignada[i]
            rend = rend_base[zafra] + np.random.uniform(-1.5, 1.5)
            azucar = cn * rend / 100
            total_azucar += azucar
            sup = cn / np.random.uniform(55, 75)
            rows.append({
                "ingenio": ing,
                "zafra": zafra,
                "cana_molida_bruta": cn * np.random.uniform(1.02, 1.08),
                "cana_molida_neta": cn,
                "superficie_cosechada": sup,
                "azucar_producida_total": azucar,
                "azucar_producida_refinada": azucar * np.random.uniform(0.3, 0.5),
                "azucar_producida_estandar": azucar * np.random.uniform(0.3, 0.5),
                "azucar_producida_blanca_especial": azucar * np.random.uniform(0.05, 0.15),
                "azucar_producida_mascabado": azucar * np.random.uniform(0.02, 0.08),
                "rendimiento_campo": sup / cn * 100 if cn > 0 else 0,
                "rendimiento_fabrica": azucar / cn * 100 if cn > 0 else 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. TRANSFORM
# ---------------------------------------------------------------------------

COLUMN_MAPPING = {
    "azucar_total": "azucar_producida_total",
    "azucar_refinada": "azucar_producida_refinada",
    "azucar_estandar": "azucar_producida_estandar",
    "azucar_blanca_especial": "azucar_producida_blanca_especial",
    "azucar_mascabado": "azucar_producida_mascabado",
    "azucar_pol_menor_99_2": "toneladas_azucar_con_pol_menor_99_2",
    "no_s_zafra": "semana",
    "cania_molida_neta": "cana_molida_neta",
    "cania_molida_bruta": "cana_molida_bruta",
    "karbe_ton_cania_neta_teorico": "karbe_cana_neta",
    "karbe_ton_cana_bruta_teorico": "karbe_cana_bruta",
}

COLUMNS_TO_KEEP = [
    "ingenio", "zafra", "semana",
    "cana_molida_bruta", "cana_molida_neta", "superficie_cosechada",
    "azucar_producida_total", "azucar_producida_refinada",
    "azucar_producida_estandar", "azucar_producida_blanca_especial",
    "azucar_producida_mascabado",
    "rendimiento_campo", "rendimiento_fabrica",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a un schema comun."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Fusionar columnas que se mapean al mismo destino
    for old, new in COLUMN_MAPPING.items():
        if old in df.columns and old != new:
            if new in df.columns:
                # El destino ya existe: combinar (rellenar NaN con la columna vieja)
                df[new] = df[new].fillna(df[old])
                df.drop(columns=[old], inplace=True)
            else:
                df.rename(columns={old: new}, inplace=True)

    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df


def _clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia valores nulos: rellena numericos con 0, elimina filas sin cana o azucar."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(0)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")
    df = df.dropna(subset=["cana_molida_neta", "azucar_producida_total"], how="all")
    return df


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas a tipos optimos."""
    df = df.copy()

    # Intentar convertir columnas de tipo string a numerico
    str_cols = df.select_dtypes(include=["object", "str", "string"]).columns
    for col in str_cols:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted.fillna(0)
        except (ValueError, TypeError, Exception):
            pass

    for col in ["cana_molida_bruta", "cana_molida_neta", "superficie_cosechada",
                 "azucar_producida_total"]:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            except Exception:
                pass
    return df


def _create_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables derivadas para el analisis."""
    df = df.copy()
    mask = (df["cana_molida_neta"] > 0) & (df["superficie_cosechada"] > 0)

    df["rendimiento_agroindustrial"] = np.where(
        mask,
        (df["azucar_producida_total"] / df["cana_molida_neta"]) * 100,
        0
    )
    df["eficiencia_extraccion"] = np.where(
        df["cana_molida_bruta"] > 0,
        (df["cana_molida_neta"] / df["cana_molida_bruta"]) * 100,
        0
    )
    zafras_validas = [z for z in df["zafra"].unique() if isinstance(z, str) and "-" in z]
    year_map = {}
    for z in zafras_validas:
        parts = z.split("-")
        try:
            year_map[z] = int(parts[0])
        except (ValueError, IndexError):
            continue
    df["anio_inicio"] = df["zafra"].map(year_map)
    return df


def _deduplicate_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """
    Los datos semanales de Infocana son ACUMULATIVOS por zafra.
    Por cada (ingenio, zafra) debemos conservar SOLO la ultima semana
    (la que contiene los totales acumulados finales).
    """
    if "semana" not in df.columns:
        return df
    df = df.copy()
    df["semana"] = pd.to_numeric(df["semana"], errors="coerce").fillna(0)
    idx = df.groupby(["ingenio", "zafra"])["semana"].idxmax()
    df = df.loc[idx.dropna()].reset_index(drop=True)
    print(f"[TRANSFORM] Despues de deduplicar acumulados: {len(df)} registros")
    df.drop(columns=["semana"], inplace=True)
    return df


def _aggregate_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega datos por zafra (anio) para el analisis temporal."""
    agg = df.groupby("zafra", as_index=False).agg({
        "cana_molida_bruta": "sum",
        "cana_molida_neta": "sum",
        "superficie_cosechada": "sum",
        "azucar_producida_total": "sum",
        "azucar_producida_refinada": "sum",
        "azucar_producida_estandar": "sum",
        "azucar_producida_blanca_especial": "sum",
        "azucar_producida_mascabado": "sum",
    })
    mask_agg = agg["cana_molida_neta"] > 0
    agg["rendimiento_promedio"] = np.where(
        mask_agg,
        (agg["azucar_producida_total"] / agg["cana_molida_neta"]) * 100,
        0
    )
    agg["rendimiento_campo_promedio"] = np.where(
        agg["superficie_cosechada"] > 0,
        agg["cana_molida_neta"] / agg["superficie_cosechada"],
        0
    )
    agg = agg.sort_values("zafra").reset_index(drop=True)
    agg["tendencia_crecimiento"] = agg["azucar_producida_total"].pct_change() * 100
    agg["tendencia_crecimiento"] = agg["tendencia_crecimiento"].fillna(0)
    return agg


def transform(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta toda la pipeline de transformacion."""
    print("[TRANSFORM] Normalizando columnas...")
    df = _normalize_columns(df_raw)

    # Filtrar solo zafras del periodo 2015-2026
    if "zafra" in df.columns:
        zafras_validas = [z for z in df["zafra"].unique() if isinstance(z, str) and len(z) == 9]
        zafras_filtradas = sorted([z for z in zafras_validas if z >= "2015-2016" and z <= "2025-2026"])
        df = df[df["zafra"].isin(zafras_filtradas)].copy()
        print(f"[TRANSFORM] Zafras en periodo 2015-2026: {len(zafras_filtradas)}")

    print("[TRANSFORM] Seleccionando columnas relevantes...")
    available_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[available_cols].copy()

    print("[TRANSFORM] Convirtiendo tipos de datos...")
    df = _convert_dtypes(df)

    print("[TRANSFORM] Limpiando valores nulos...")
    df = _clean_nulls(df)

    print("[TRANSFORM] Deduplicando datos acumulativos (ultima semana por ingenio)...")
    df = _deduplicate_cumulative(df)

    print("[TRANSFORM] Creando variables derivadas...")
    df = _create_derived_variables(df)

    print("[TRANSFORM] Agregando por zafra...")
    df_annual = _aggregate_annual(df)

    print(f"[TRANSFORM] Registros detallados: {len(df):,}")
    print(f"[TRANSFORM] Registros agregados (por zafra): {len(df_annual)}")
    return df, df_annual


# ---------------------------------------------------------------------------
# 3. LOAD
# ---------------------------------------------------------------------------

def load_to_csv(df: pd.DataFrame, df_annual: pd.DataFrame) -> str:
    """Guarda los datasets limpios en CSV."""
    detal_path = os.path.join(PROCESSED_DIR, "infocana_limpio_detallado.csv")
    annual_path = os.path.join(PROCESSED_DIR, "infocana_limpio_anual.csv")

    df.to_csv(detal_path, index=False, encoding="utf-8")
    print(f"[LOAD] CSV detallado guardado: {detal_path}")

    df_annual.to_csv(annual_path, index=False, encoding="utf-8")
    print(f"[LOAD] CSV anual guardado: {annual_path}")

    return detal_path, annual_path


def load_to_sqlite(df: pd.DataFrame, df_annual: pd.DataFrame) -> str:
    """Guarda los datasets en una base de datos SQLite."""
    db_path = os.path.join(PROCESSED_DIR, "infocana.db")
    conn = sqlite3.connect(db_path)
    df.to_sql("produccion_detallada", conn, if_exists="replace", index=False)
    df_annual.to_sql("produccion_anual", conn, if_exists="replace", index=False)
    conn.close()
    print(f"[LOAD] SQLite guardado: {db_path}")
    return db_path


def load(df: pd.DataFrame, df_annual: pd.DataFrame) -> dict:
    """Ejecuta todas las cargas."""
    result = {"dataframe_detallado": df, "dataframe_anual": df_annual}
    result["csv_detallado"], result["csv_anual"] = load_to_csv(df, df_annual)

    try:
        result["sqlite"] = load_to_sqlite(df, df_annual)
    except Exception as e:
        print(f"[WARN] SQLite no disponible: {e}")
        result["sqlite"] = None

    report = {
        "total_registros": len(df),
        "total_zafras": df["zafra"].nunique() if "zafra" in df.columns else 0,
        "total_ingenios": df["ingenio"].nunique() if "ingenio" in df.columns else 0,
        "columnas": list(df.columns),
    }
    pd.Series(report).to_json(os.path.join(OUTPUT_DIR, "etl_report.json"), indent=2)
    return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_etl_pipeline() -> dict:
    """Ejecuta la pipeline ETL completa."""
    print("=" * 60)
    print("PIPELINE ETL - INFOCANA")
    print("=" * 60)

    df_raw = extract_raw_data()
    df_clean, df_annual = transform(df_raw)
    artifacts = load(df_clean, df_annual)

    print("=" * 60)
    print("ETL COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    return artifacts


if __name__ == "__main__":
    artifacts = run_etl_pipeline()
