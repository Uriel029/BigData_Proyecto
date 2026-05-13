#!/usr/bin/env python3
"""
Punto de entrada principal del proyecto.
Ejecuta la pipeline completa: ETL -> Modelado -> Dashboard
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name: str, script_path: str):
    print(f"\n{'='*60}")
    print(f"  PASO: {step_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_path], cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"[ERROR] Fallo en: {step_name}")
        sys.exit(result.returncode)
    print(f"[OK] {step_name} completado\n")


def main():
    print("=" * 60)
    print("  PROYECTO BIG DATA - PREDICCION AZUCAR MEXICO")
    print("  Pipeline completa: ETL + Modelado + Dashboard")
    print("=" * 60)

    steps = [
        ("Descarga de datos", "scripts/download_data.py"),
        ("Pipeline ETL", "scripts/etl_pipeline.py"),
        ("Modelo de regresion lineal", "models/linear_regression.py"),
    ]

    for step_name, script_path in steps:
        full_path = os.path.join(PROJECT_DIR, script_path)
        if os.path.exists(full_path):
            run_step(step_name, full_path)
        else:
            print(f"[SKIP] {step_name}: {script_path} no encontrado")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print(f"\nPara iniciar el dashboard:")
    print(f"  python -m shiny run {os.path.join(PROJECT_DIR, 'dashboard', 'app.py')}")
    print(f"\nO desde la raiz del proyecto:")
    print(f"  shiny run dashboard/app.py")


if __name__ == "__main__":
    main()
