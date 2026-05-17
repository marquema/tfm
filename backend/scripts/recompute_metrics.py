"""
Recompute métricas Retorno/MDD/Vol/CAGR a partir de modelos guardados.

Reparación post-hoc por bug de keys (`compute_metrics` usa
'Retorno Total (%)' y no 'Retorno Total', etc.) que causó NaN
en JSONs de optuna_retrain_results.json, a2c_multiseed_results.json,
sac_multiseed_results.json.

Carga cada modelo guardado en disco, re-eval sobre TEST intocable,
escribe métricas correctas en el JSON. Sharpes existentes coinciden,
solo se reparan los campos NaN.

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/recompute_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.environment_trading import PortfolioEnv  # noqa: E402
from src.benchmarking.baselines import compute_metrics  # noqa: E402


FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')
MODELS_DIR = BACKEND_ROOT / 'models'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
TRAIN_SPLIT_PCT = 0.80

# Profiles para varphi/gamma (env params)
PROFILES = {
    'low_turnover': dict(varphi=0.02, gamma=0.020),
    'aggressive':   dict(varphi=0.01, gamma=0.005),
}


def eval_model(model, varphi: float, gamma: float, train_end_idx: int, n_total: int) -> dict:
    """Backtest modelo sobre TEST. Devuelve metrics dict completo con keys correctas."""
    env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=train_end_idx,
        end_idx=n_total,
        phi=varphi,
        gamma=gamma,
        reward_type='sharpe',
    )
    obs, _ = env.reset()
    equity = [env.initial_balance]
    done, trunc = False, False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(action)
        equity.append(float(info['value']))

    metrics = compute_metrics(pd.Series(equity), annual_rf=0.04)
    return {
        'sharpe': float(metrics.get('Sharpe Ratio', np.nan)),
        'sortino': float(metrics.get('Sortino Ratio', np.nan)),
        'retorno_pct': float(metrics.get('Retorno Total (%)', np.nan)),
        'cagr_pct': float(metrics.get('CAGR (%)', np.nan)),
        'vol_anual_pct': float(metrics.get('Volatilidad Anualizada (%)', np.nan)),
        'mdd_pct': float(metrics.get('Max Drawdown (%)', np.nan)),
    }


def repair_optuna_ppo():
    """Repara optuna_retrain_results.json (PPO Optuna × 5 seeds)."""
    json_path = REPORTS_DIR / 'optuna_retrain_results.json'
    if not json_path.exists():
        print(f'[skip] {json_path.name} no existe')
        return

    with open(json_path) as f:
        data = json.load(f)

    varphi = data['best_params']['varphi']
    sac_gamma_param_unused = data['best_params']['gamma']  # 'gamma' aquí = turnover penalty
    gamma_turnover = data['best_params']['gamma']

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    print(f'\n=== Reparando PPO Optuna retrain ({len(data["runs"])} runs) ===')
    for run in data['runs']:
        seed = run['seed']
        model_path = MODELS_DIR / f'best_model_academic_OPTUNA_seed{seed}' / 'best_model.zip'
        if not model_path.exists():
            print(f'  seed {seed}: modelo no encontrado, skip')
            continue
        print(f'  seed {seed}: re-eval...', end='', flush=True)
        model = PPO.load(str(model_path))
        m = eval_model(model, varphi, gamma_turnover, train_end_idx, n_total)
        run.update(m)
        del model
        print(f' Sharpe={m["sharpe"]:.4f} Retorno={m["retorno_pct"]:+.2f}% MDD={m["mdd_pct"]:+.2f}%')

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'JSON reparado: {json_path}')


def repair_a2c_or_sac(json_filename: str, algo_class, model_prefix: str):
    """Repara JSON A2C o SAC."""
    json_path = REPORTS_DIR / json_filename
    if not json_path.exists():
        print(f'\n[skip] {json_path.name} no existe')
        return

    with open(json_path) as f:
        data = json.load(f)

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    print(f'\n=== Reparando {json_filename} ({len(data.get("runs", []))} runs) ===')
    for run in data.get('runs', []):
        profile = run['profile']
        seed = run['seed']
        pf = PROFILES[profile]
        model_path = MODELS_DIR / f'{model_prefix}_{profile}_seed{seed}' / 'best_model.zip'
        if not model_path.exists():
            print(f'  {profile} seed {seed}: modelo no encontrado, skip')
            continue
        print(f'  {profile} seed {seed}: re-eval...', end='', flush=True)
        model = algo_class.load(str(model_path))
        m = eval_model(model, pf['varphi'], pf['gamma'], train_end_idx, n_total)
        run.update(m)
        del model
        print(f' Sharpe={m["sharpe"]:.4f} Retorno={m["retorno_pct"]:+.2f}% MDD={m["mdd_pct"]:+.2f}%')

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'JSON reparado: {json_path}')


def summarize():
    """Print resumen final de los 3 JSONs."""
    print('\n' + '=' * 70)
    print('RESUMEN MULTI-SEED (TEST honesto, N=5 seeds)')
    print('=' * 70)

    for fname, label in [
        ('optuna_retrain_results.json', 'PPO Optuna best'),
        ('a2c_multiseed_results.json', 'A2C'),
        ('sac_multiseed_results.json', 'SAC'),
    ]:
        path = REPORTS_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        runs = data.get('runs', [])
        if not runs:
            continue

        if 'profile' in runs[0]:
            for profile in ['low_turnover', 'aggressive']:
                sub = [r for r in runs if r['profile'] == profile]
                if not sub:
                    continue
                sh = np.array([r['sharpe'] for r in sub])
                ret = np.array([r['retorno_pct'] for r in sub])
                mdd = np.array([r['mdd_pct'] for r in sub])
                print(f'\n{label} ({profile}) N={len(sub)}:')
                print(f'  Sharpe:  {sh.mean():.4f} ± {sh.std():.4f}')
                print(f'  Retorno: {ret.mean():+.2f}% ± {ret.std():.2f}%')
                print(f'  MDD:     {mdd.mean():+.2f}% ± {mdd.std():.2f}%')
        else:
            sh = np.array([r['sharpe'] for r in runs])
            ret = np.array([r['retorno_pct'] for r in runs])
            mdd = np.array([r['mdd_pct'] for r in runs])
            print(f'\n{label} N={len(runs)}:')
            print(f'  Sharpe:  {sh.mean():.4f} ± {sh.std():.4f}')
            print(f'  Retorno: {ret.mean():+.2f}% ± {ret.std():.2f}%')
            print(f'  MDD:     {mdd.mean():+.2f}% ± {mdd.std():.2f}%')


def main():
    print('=' * 70)
    print('RECOMPUTE MÉTRICAS DESDE MODELOS GUARDADOS')
    print('=' * 70)

    repair_optuna_ppo()
    repair_a2c_or_sac('a2c_multiseed_results.json', A2C, 'best_model_academic_a2c')
    repair_a2c_or_sac('sac_multiseed_results.json', SAC, 'best_model_academic_sac')
    summarize()


if __name__ == '__main__':
    main()
