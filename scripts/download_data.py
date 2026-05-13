"""
Script: download_data.py
Proposito: Descarga los archivos CSV historicos de Infocana desde datos.gob.mx
"""

import os
import requests
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = [
    ("2025-2026", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/Infocana_25_26_resumen.csv"),
    ("2024-2025", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/Infocana_24_25_resumen_ok.csv"),
    ("2023-2024", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_23_24_resumen_ok.csv"),
    ("2022-2023", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_22_23_resumen_ok.csv"),
    ("2021-2022", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_21_22_resumen_ok.csv"),
    ("2020-2021", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_20_21_resumen_ok.csv"),
    ("2019-2020", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_19_20_resumen_ok.csv"),
    ("2018-2019", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_18_19_resumen_ok.csv"),
    ("2017-2018", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_17_18_resumen_ok.csv"),
    ("2016-2017", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_16_17_resumen_ok.csv"),
    ("2015-2016", "https://repodatos.atdt.gob.mx/api_update/conadesuca/avance_produccion_cana_azucar_infocana/infocana_15_16_resumen_ok.csv"),
]

def download_all():
    for zafra, url in URLS:
        filename = f"infocana_{zafra.replace('-', '_')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"[OK] {filename} ya existe")
            continue
        print(f"[DL] Descargando {zafra}...")
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"[OK] {filename} guardado ({len(resp.content)} bytes)")
        except Exception as e:
            print(f"[ERR] {zafra}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    download_all()
