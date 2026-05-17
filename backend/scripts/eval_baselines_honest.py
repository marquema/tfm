"""
Eval baselines honest sobre TEST n=17 (universo data leakage fixed).

6 estrategias: Equal-Weight, Buy & Hold, 60/40, Markowitz MV, Random Uniform, Momentum Top-K.

Coste: ~2-3 min (Markowitz dominante por scipy.optimize cada mes).

Resultado se documenta en memoria §4.4 tabla baselines honest.

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/eval_baselines_honest.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.benchmarking.baselines import run_baselines, compute_metrics  # noqa: E402


PRICES_PATH = BACKEND_ROOT / 'data' / 'original_prices.csv'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
RESULTS_PATH = REPORTS_DIR / 'baselines_honest_results.json'
TRAIN_SPLIT_PCT = 0.80


def main():
    print('=' * 70)
    print('EVAL BASELINES HONEST sobre TEST (n=17 universo data-leakage fixed)')
    print('=' * 70)

    df_p = pd.read_csv(PRICES_PATH, index_col=0)
    n_total = len(df_p)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    df_test = df_p.iloc[train_end_idx:].copy()

    print(f'Universo: {df_p.shape[1]} activos')
    print(f'TEST: filas [{train_end_idx}:{n_total}] = {len(df_test)} dias')
    print(f'Tickers: {", ".join(df_p.columns.tolist())}')
    print()

    print('Ejecutando 6 baselines...')
    series_dict = run_baselines(
        df_test,
        initial_balance=10000.0,
        commission=0.001,
        ticker_equity='IVV_Close',
        ticker_bond='BND_Close',
    )

    print('\nMetricas:')
    results = {}
    for name, series in series_dict.items():
        m = compute_metrics(series, annual_rf=0.04)
        results[name] = {
            'sharpe': m['Sharpe Ratio'],
            'sortino': m['Sortino Ratio'],
            'retorno_pct': m['Retorno Total (%)'],
            'cagr_pct': m['CAGR (%)'],
            'vol_anual_pct': m['Volatilidad Anualizada (%)'],
            'mdd_pct': m['Max Drawdown (%)'],
            'valor_final': m['Valor Final ($)'],
        }
        print(f'  {name:25s}: Sharpe {m["Sharpe Ratio"]:+.3f} | Ret {m["Retorno Total (%)"]:+7.2f}% | MDD {m["Max Drawdown (%)"]:+6.2f}%')

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump({
            'universe_size': df_p.shape[1],
            'tickers': df_p.columns.tolist(),
            'test_days': len(df_test),
            'train_end_idx': train_end_idx,
            'n_total': n_total,
            'results': results,
        }, f, indent=2)

    print('\n' + '=' * 70)
    print(f'Resultado: {RESULTS_PATH}')
    print('=' * 70)
    print('Comparativa rapida (Sharpe ordenado):')
    ranked = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)
    for i, (name, m) in enumerate(ranked, 1):
        print(f'  #{i} {name:25s} Sharpe={m["sharpe"]:+.3f}')


if __name__ == '__main__':
    main()
