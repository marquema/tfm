"""
Re-entrena agente especulativo GMM+KMeans sobre universo honest (n=17).

Motivo:
- Detector HMM previo entrenado con universo data leakage (n=15) — incompatible
  con columnas universo honest. Dashboard daba KeyError al usarlo.
- Re-entrenamiento sobre features actuales restaura baseline GMM+KMeans
  en tabla §4.4 comparativa.

Replica flujo de POST /admin/fase4/ajustar-especulativo (main.py:1240+).

Tiempo: segundos (no horas).

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/retrain_speculative_honest.py
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.unsupervised.speculative_agent import SpeculativeAgent  # noqa: E402


FEATURES_PATH = BACKEND_ROOT / 'data' / 'normalized_features.csv'
PRICES_PATH = BACKEND_ROOT / 'data' / 'original_prices.csv'
MODEL_OUT = BACKEND_ROOT / 'models' / 'speculative_gmm.pkl'
SPLIT_PCT = 0.80


def main():
    print('=' * 60)
    print('Retrain agente especulativo GMM+KMeans (universo honest n=17)')
    print('=' * 60)

    df_f = pd.read_csv(FEATURES_PATH, index_col=0)
    df_p = pd.read_csv(PRICES_PATH, index_col=0)
    split_idx = int(len(df_f) * SPLIT_PCT)
    print(f'Dataset: {len(df_f)} días, {df_f.shape[1]} features, {df_p.shape[1]} activos')
    print(f'Train [0:{split_idx}], Test [{split_idx}:{len(df_f)}]')

    # Backup viejo .pkl si existe
    if MODEL_OUT.exists():
        backup = MODEL_OUT.with_suffix('.pkl.pre_honest_bak')
        MODEL_OUT.replace(backup)
        print(f'Backup viejo modelo: {backup.name}')

    agent = SpeculativeAgent(n_regimes=3, n_clusters=3, cluster_window=60)
    print('Entrenando GMM (3 regímenes) + KMeans (3 clusters)...')
    agent.fit(df_f.iloc[:split_idx], df_p.iloc[:split_idx])
    print('Fit OK')

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(agent, f)
    print(f'Modelo guardado: {MODEL_OUT}')

    # Smoke test backtest sobre TEST
    print('\nSmoke test backtest sobre TEST...')
    spec_series = agent.backtest(
        df_f.iloc[split_idx:],
        df_p.iloc[split_idx:],
        initial_balance=10000.0,
        commission=0.001,
    )
    final_value = float(spec_series.iloc[-1])
    return_pct = (final_value / 10000.0 - 1) * 100
    print(f'Valor final TEST: ${final_value:,.2f}')
    print(f'Retorno TEST: {return_pct:+.2f}%')
    print('=' * 60)
    print('Listo. Dashboard ya puede cargar baseline GMM sin error.')


if __name__ == '__main__':
    main()
