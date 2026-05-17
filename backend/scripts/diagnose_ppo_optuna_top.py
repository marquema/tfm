"""
Diagnose pesos PPO Optuna top seed (seed 1, Sharpe 0.607) sobre TEST corregido n=17.

Genera CSV con pesos diarios para tabla §4.5 memoria.

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/diagnose_ppo_optuna_top.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.environment_trading import PortfolioEnv  # noqa: E402


FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')
MODEL_PATH = str(BACKEND_ROOT / 'models' / 'best_model_academic_OPTUNA_seed1' / 'best_model.zip')
OUT_CSV = BACKEND_ROOT / 'src' / 'reports' / 'diagnose_ppo_weights.csv'

SPLIT_PCT = 0.80
VARPHI = 0.0094
GAMMA = 0.0020


def main():
    df_p = pd.read_csv(PRICES_PATH, index_col=0)
    df_f = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df_f)
    split = int(n_total * SPLIT_PCT)
    tickers = [c.replace('_Close', '') for c in df_p.columns]
    print(f'Universo {len(tickers)}: {tickers}')
    print(f'TEST [{split}:{n_total}] = {n_total-split} dias ({df_f.index[split]} a {df_f.index[-1]})')

    print(f'Cargando: {MODEL_PATH}')
    model = PPO.load(MODEL_PATH)

    env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=split,
        end_idx=n_total,
        phi=VARPHI,
        gamma=GAMMA,
        reward_type='sharpe',
    )

    obs, _ = env.reset()
    done, trunc = False, False
    weights = []
    equity = [env.initial_balance]
    dates = [df_f.index[split]]
    step = 0
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        w = np.clip(action, 0, 1)
        w = w / (w.sum() + 1e-8)
        weights.append(w)
        obs, _, done, trunc, info = env.step(action)
        equity.append(float(info['value']))
        dates.append(df_f.index[split + 1 + step] if split + 1 + step < n_total else df_f.index[-1])
        step += 1

    w_arr = np.array(weights)
    df_w = pd.DataFrame(w_arr, columns=tickers, index=dates[:-1])
    df_w.index.name = 'date'
    df_w.to_csv(OUT_CSV)
    print(f'\nCSV: {OUT_CSV} ({len(df_w)} filas x {len(tickers)} cols)')

    print('\nEstadisticos pesos PPO Optuna seed 1 (top):')
    print(f'{"Activo":<8} {"Peso medio (%)":>15} {"Std (%)":>10} {"Mediana (%)":>13} {"Pico max (%)":>14}')
    print('-' * 65)
    stats = []
    for i, t in enumerate(tickers):
        avg = w_arr[:, i].mean() * 100
        std = w_arr[:, i].std() * 100
        med = np.median(w_arr[:, i]) * 100
        mx = w_arr[:, i].max() * 100
        stats.append((t, avg, std, med, mx))
        print(f'{t:<8} {avg:>15.2f} {std:>10.2f} {med:>13.2f} {mx:>14.2f}')

    # Ordenado por peso medio descendente
    stats_sorted = sorted(stats, key=lambda x: x[1], reverse=True)
    print('\n--- ORDENADO POR PESO MEDIO ---')
    for t, avg, std, med, mx in stats_sorted:
        print(f'{t:<8} {avg:>15.2f} {std:>10.2f} {med:>13.2f} {mx:>14.2f}')

    final = equity[-1]
    sharpe_check = pd.Series(equity).pct_change().dropna()
    sh = (sharpe_check.mean() - 0.04/252) / sharpe_check.std() * np.sqrt(252)
    print(f'\nValor final: ${final:,.2f} | Retorno {(final/equity[0]-1)*100:+.2f}% | Sharpe verif: {sh:.3f}')


if __name__ == '__main__':
    main()
