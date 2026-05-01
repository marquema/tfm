"""
Orchestrator: Paso 1 (screener) + Paso 2 (preparar datos).

Ejecuta el screener sobre el S&P 500 con los nuevos parametros del TFM
(start_date=2014, top_n=15, force_include=IVV/BND/IBIT/ETHA, dedup hermanos)
y a continuacion regenera los CSVs de features y precios con la nueva
logica de proxy BTC-USD/ETH-USD para IBIT/ETHA pre-launch.

Uso:
    .venv/Scripts/python.exe scripts/run_screener_and_prepare.py

Salidas:
    data/normalized_features.csv  (z-score sobre split global 80%)
    data/original_features.csv    (sin normalizar; lo usa walk-forward y EW)
    data/original_prices.csv

NOTA: este script NO actualiza la BD del sistema. Si despues quieres usar
las simulaciones del rol inversor desde el frontend, levanta el server y
re-ejecuta /admin/fase1/screener desde la UI para que la BD se sincronice.
"""

import os
import sys
import time
import json

# Forzar UTF-8 en stdout/stderr para evitar UnicodeEncodeError cuando se
# ejecuta con stdout redirigido a fichero en Windows (cp1252 por defecto
# rechaza caracteres como flechas Unicode, em-dash, etc.). Equivalente a
# definir PYTHONIOENCODING=utf-8 antes de lanzar el script.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass  # Python <3.7 o stdout ya cerrado; tirar adelante.

# Asegurar que `src` esta en el PYTHONPATH
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(HERE)
sys.path.insert(0, BACKEND_ROOT)

from src.pipeline_getdata.market_screener import MarketScreener
from src.pipeline_getdata.data_downloader import generate_dataset


START_DATE = "2014-01-01"
END_DATE = "2026-04-16"
TOP_N = 15
MAX_PER_SECTOR = 3
FORCE_INCLUDE = ['IVV', 'BND', 'IBIT', 'ETHA']


def run_step1_screener():
    print("\n" + "#" * 70)
    print("# PASO 1 - SCREENER DE MERCADO")
    print(f"# start={START_DATE}  end={END_DATE}  top_n={TOP_N}")
    print(f"# force_include={FORCE_INCLUDE}  max_per_sector={MAX_PER_SECTOR}")
    print("#" * 70)
    t0 = time.time()
    screener = MarketScreener(max_per_sector=MAX_PER_SECTOR)
    result = screener.run(
        start_date=START_DATE,
        end_date=END_DATE,
        top_n=TOP_N,
        force_include=FORCE_INCLUDE,
    )
    elapsed = time.time() - t0
    candidates = result['candidates']
    print(f"\n  >>> Screener tardo {elapsed/60:.1f} min")
    print(f"  >>> Candidatos finales ({len(candidates)}): {candidates}")

    # Persistimos el resultado del screener en un JSON para auditoria.
    out_path = os.path.join(BACKEND_ROOT, 'src', 'reports',
                            'screener_result.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        'start_date': START_DATE,
        'end_date': END_DATE,
        'top_n': TOP_N,
        'max_per_sector': MAX_PER_SECTOR,
        'force_include': FORCE_INCLUDE,
        'candidates': candidates,
        'filtered_out': result.get('filtered_out', {}),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  >>> Resultado guardado en {out_path}")
    return candidates


def run_step2_prepare_data(tickers):
    print("\n" + "#" * 70)
    print("# PASO 2 - PREPARAR DATOS")
    print(f"# tickers ({len(tickers)}) = {tickers}")
    print(f"# start={START_DATE}  end={END_DATE}")
    print("#" * 70)
    t0 = time.time()
    generate_dataset(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    elapsed = time.time() - t0
    print(f"\n  >>> Preparar-datos tardo {elapsed/60:.1f} min")
    print(f"  >>> CSVs generados en data/")


if __name__ == "__main__":
    candidates = run_step1_screener()
    run_step2_prepare_data(candidates)
    print("\n" + "=" * 70)
    print("ORCHESTRATOR COMPLETADO. Listo para Paso 3 (entrenar) cuando decidas.")
    print("=" * 70)
