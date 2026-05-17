"""
Experimento extra: PPO con best Optuna hyperparams + recompensa DUAL.

Combina:
- Best hyperparams encontrados por Optuna (algoritmo PPO calibrado)
- Recompensa dual (Sharpe + log-retorno) en lugar de sharpe puro

Hipótesis: dual reward + Optuna hyperparams puede superar dual + manual
(que dio Sharpe ~0.94 single-seed sobre universo corregido) y/o PPO Optuna sharpe puro
(que dio 0.47 ± 0.11).

Coste: 1 run × 1.5M pasos ≈ 45 min.

Resultado se documenta como experimento adicional en memoria §4.7
("ablación iteración 2 ampliada con configuración Optuna").

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/retrain_ppo_optuna_dual.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from stable_baselines3 import PPO

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.environment_trading import PortfolioEnv  # noqa: E402
from src.benchmarking.baselines import compute_metrics  # noqa: E402
from src.hpo.space import validate_batch_size_compatibility  # noqa: E402


STUDY_NAME = 'ppo_hpo_tfm'
STORAGE = f'sqlite:///{(BACKEND_ROOT / "hpo" / "optuna_study_ppo.db").as_posix()}'

SEED = 0  # 1 run reproducible
TRAIN_STEPS = 1_500_000
TRAIN_SPLIT_PCT = 0.80
ALPHA = 0.5
BETA = 0.5

FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')

MODELS_DIR = BACKEND_ROOT / 'models'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
RESULTS_PATH = REPORTS_DIR / 'optuna_dual_result.json'

NET_ARCH = [256, 256]


def main():
    """Entrena PPO con best params Optuna + reward dual (alpha=beta=0.5).
    1 seed × 1.5M pasos. Eval out-of-sample sobre TEST y persiste resultado JSON."""
    print('=' * 70)
    print('PPO Optuna best + recompensa DUAL (alpha=beta=0.5)')
    print('=' * 70)

    # Cargar best Optuna
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    bp = dict(study.best_params)
    print(f'Best Sharpe val Optuna: {study.best_value:.4f} (trial #{study.best_trial.number})')
    print('Best params:')
    for k, v in bp.items():
        print(f'  {k}: {v}')

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    print(f'\nTrain [0:{train_end_idx}], Test [{train_end_idx}:{n_total}]')
    print(f'Recompensa: DUAL alpha={ALPHA} beta={BETA}')

    n_steps = bp['n_steps']
    batch_size = validate_batch_size_compatibility(n_steps, bp['batch_size'])

    # Entorno train CON recompensa dual
    train_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=0,
        end_idx=train_end_idx,
        phi=bp['varphi'],
        gamma=bp['gamma'],
        reward_type='dual',
        alpha=ALPHA,
        beta=BETA,
    )

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=bp['learning_rate'],
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=bp['clip_range'],
        ent_coef=bp['ent_coef'],
        gae_lambda=bp['gae_lambda'],
        vf_coef=bp['vf_coef'],
        max_grad_norm=bp['max_grad_norm'],
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=0,
        device='cpu',
    )

    t0 = time.time()
    print(f'\nEntrenando {TRAIN_STEPS:,} pasos (seed={SEED})...')
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
    train_time_min = (time.time() - t0) / 60
    print(f'Train OK en {train_time_min:.1f} min')

    save_path = MODELS_DIR / 'best_model_academic_OPTUNA_dual'
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path / 'best_model.zip'))
    print(f'Modelo guardado: {save_path.name}')

    # Eval TEST con recompensa dual también (consistencia)
    test_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=train_end_idx,
        end_idx=n_total,
        phi=bp['varphi'],
        gamma=bp['gamma'],
        reward_type='dual',
        alpha=ALPHA,
        beta=BETA,
    )

    obs, _ = test_env.reset()
    equity = [test_env.initial_balance]
    done, trunc = False, False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = test_env.step(action)
        equity.append(float(info['value']))

    metrics = compute_metrics(pd.Series(equity), annual_rf=0.04)

    result = {
        'config': 'PPO Optuna best + dual reward',
        'best_params_optuna': bp,
        'alpha': ALPHA,
        'beta': BETA,
        'seed': SEED,
        'train_steps': TRAIN_STEPS,
        'sharpe': float(metrics.get('Sharpe Ratio', np.nan)),
        'sortino': float(metrics.get('Sortino Ratio', np.nan)),
        'retorno_pct': float(metrics.get('Retorno Total (%)', np.nan)),
        'cagr_pct': float(metrics.get('CAGR (%)', np.nan)),
        'vol_anual_pct': float(metrics.get('Volatilidad Anualizada (%)', np.nan)),
        'mdd_pct': float(metrics.get('Max Drawdown (%)', np.nan)),
        'train_time_min': train_time_min,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(result, f, indent=2)

    print('\n' + '=' * 70)
    print('RESULTADO PPO Optuna + DUAL')
    print('=' * 70)
    print(f'Sharpe TEST:   {result["sharpe"]:.4f}')
    print(f'Sortino:       {result["sortino"]:.4f}')
    print(f'Retorno:       {result["retorno_pct"]:+.2f}%')
    print(f'CAGR:          {result["cagr_pct"]:+.2f}%')
    print(f'Vol anual:     {result["vol_anual_pct"]:.2f}%')
    print(f'MDD:           {result["mdd_pct"]:+.2f}%')
    print('=' * 70)
    print('Comparativa:')
    print(f'  PPO LT manual corregido  : Sharpe 0.469 (single seed)')
    print(f'  PPO dual manual corregido: Sharpe 0.936 (single seed)')
    print(f'  PPO Optuna sharpe N=5 : Sharpe 0.474 ± 0.107')
    print(f'  PPO Optuna + dual     : Sharpe {result["sharpe"]:.4f}  ← este experimento')
    print(f'Resultado: {RESULTS_PATH}')


if __name__ == '__main__':
    main()
