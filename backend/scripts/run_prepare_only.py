"""
Orchestrator parcial: solo Paso 2 (preparar datos).

Lee los 15 tickers del screener guardado en src/reports/screener_result.json
y re-ejecuta generate_dataset con la nueva logica de proxy BTC-USD/ETH-USD
para IBIT/ETHA.

Sobre la fecha de inicio del preparar-datos:
    El screener fue ejecutado con start=2014-01-01 (rango amplio para
    rankear los activos por Sharpe sobre el maximo histórico disponible).
    Sin embargo, el preparar-datos arranca en 2017-11-09 — primer dia con
    datos reales o proxy real (ETH-USD) de TODOS los activos del universo.
    Esto evita el bfill silencioso de ETHA durante 2014-2017 (~3.8 anos),
    asegurando que el dataset sea 100 % honesto desde el dia 1.

    Activos tradicionales (IVV, BND, etc.): 8.5 anos de historia real.
    Cripto (IBIT, ETHA): 8.5 anos via proxy BTC-USD/ETH-USD pre-launch
    + datos reales del ETF post-launch.

Uso:
    .venv/Scripts/python.exe scripts/run_prepare_only.py
"""

import os
import sys
import time
import json

# Forzar UTF-8 en stdout/stderr para evitar UnicodeEncodeError cuando se
# ejecuta con stdout redirigido a fichero en Windows.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(HERE)
sys.path.insert(0, BACKEND_ROOT)

from src.pipeline_getdata.data_downloader import generate_dataset


SCREENER_RESULT_PATH = os.path.join(BACKEND_ROOT, 'src', 'reports',
                                    'screener_result.json')

# Fecha de inicio efectiva del dataset de entrenamiento.
# Justificacion: ETH-USD (proxy de ETHA pre-launch) solo existe en Yahoo
# Finance desde 2017-11-09. Arrancar antes obligaria a rellenar ETHA con
# bfill (constante) durante ~3.8 anos, lo que el tutor desaconsejo
# explicitamente en su revision. Mantenemos la auditoria del screener con
# fechas amplias (2014-2026) pero entrenamos sobre el subconjunto
# temporal donde TODOS los activos tienen datos reales o proxy real.
PREPARE_START_DATE = "2017-11-09"


def main():
    if not os.path.exists(SCREENER_RESULT_PATH):
        print(f"ERROR: no encontrado {SCREENER_RESULT_PATH}")
        print("Ejecuta primero scripts/run_screener_and_prepare.py.")
        sys.exit(1)

    with open(SCREENER_RESULT_PATH, 'r', encoding='utf-8') as f:
        screener = json.load(f)

    tickers = screener['candidates']
    end_date = screener['end_date']
    start_date = PREPARE_START_DATE

    print("\n" + "#" * 70)
    print("# PASO 2 - PREPARAR DATOS (solo)")
    print(f"# tickers ({len(tickers)}) = {tickers}")
    print(f"# start={start_date}  end={end_date}")
    print(f"# (screener evaluo desde {screener['start_date']}; preparar-datos")
    print(f"#  arranca en {start_date} para evitar bfill de ETHA pre-2017)")
    print("#" * 70)

    t0 = time.time()
    generate_dataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    elapsed = time.time() - t0
    print(f"\n  >>> Preparar-datos tardo {elapsed/60:.1f} min")
    print(f"  >>> CSVs generados en data/")
    print("\n" + "=" * 70)
    print("PASO 2 COMPLETADO. Listo para Paso 3 (entrenar) cuando decidas.")
    print("=" * 70)


if __name__ == "__main__":
    main()
