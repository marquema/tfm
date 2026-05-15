"""
Retrain SAC multi-seed × 2 perfiles para evaluación robusta inter-seed.

Decisión P3 (project_optuna_decisions.md):
- SAC NO usa Optuna (alcance TFM: PPO algoritmo central)
- Hyperparams calibrados manualmente (training_analysis.py rama SAC §4.8)
- N=5 seeds × 2 perfiles (low_turnover + aggressive) = 10 runs
- Eval cada uno sobre TEST intocable

Coste: 10 × 500k pasos ≈ 5h CPU (SAC off-policy con replay → 500k suficiente).

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/retrain_sac_multiseed.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.environment_trading import PortfolioEnv  # noqa: E402
from src.benchmarking.baselines import compute_metrics  # noqa: E402


# ── Configuración ───────────────────────────────────────────────────────────
N_SEEDS = 5
TRAIN_STEPS = 500_000  # off-policy + replay → menor presupuesto suficiente
TRAIN_SPLIT_PCT = 0.80

FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')

MODELS_DIR = BACKEND_ROOT / 'models'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
RESULTS_PATH = REPORTS_DIR / 'sac_multiseed_results.json'

# Hyperparams SAC calibrados manualmente (training_analysis.py L144-158)
SAC_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'buffer_size': 100_000,
    'learning_starts': 1_000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 4,       # calibrado vs SB3 default 1
    'gradient_steps': 1,
    'ent_coef': 'auto',    # entropía adaptativa
}

NET_ARCH = [256, 256]

PROFILES = {
    'low_turnover': dict(varphi=0.02, gamma=0.020),
    'aggressive':   dict(varphi=0.01, gamma=0.005),
}


def train_one_run(profile_name: str, seed: int, save_path: Path) -> dict:
    """Entrena 1 SAC con seed y perfil dado. Eval sobre TEST."""
    profile = PROFILES[profile_name]

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    print(f'  Profile={profile_name}, seed={seed}')
    print(f'  Train [0:{train_end_idx}], Test [{train_end_idx}:{n_total}]')
    print(f'  varphi={profile["varphi"]}, gamma={profile["gamma"]}')

    train_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=0,
        end_idx=train_end_idx,
        phi=profile['varphi'],
        gamma=profile['gamma'],
        reward_type='sharpe',
    )

    # Renombro 'gamma' SAC discount → para evitar choque con 'gamma' perfil
    sac_gamma = SAC_HYPERPARAMS['gamma']

    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=SAC_HYPERPARAMS['learning_rate'],
        buffer_size=SAC_HYPERPARAMS['buffer_size'],
        learning_starts=SAC_HYPERPARAMS['learning_starts'],
        batch_size=SAC_HYPERPARAMS['batch_size'],
        tau=SAC_HYPERPARAMS['tau'],
        gamma=sac_gamma,
        train_freq=SAC_HYPERPARAMS['train_freq'],
        gradient_steps=SAC_HYPERPARAMS['gradient_steps'],
        ent_coef=SAC_HYPERPARAMS['ent_coef'],
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=seed,
        verbose=0,
        device='cpu',
    )

    t0 = time.time()
    print(f'  Entrenando {TRAIN_STEPS:,} pasos...')
    model.learn(total_timesteps=TRAIN_STEPS, progress_bar=False)
    train_time_min = (time.time() - t0) / 60
    print(f'  Train OK en {train_time_min:.1f} min')

    save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path / 'best_model.zip'))

    # Eval sobre TEST intocable
    test_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=train_end_idx,
        end_idx=n_total,
        phi=profile['varphi'],
        gamma=profile['gamma'],
        reward_type='sharpe',
    )

    obs, _ = test_env.reset()
    equity = [test_env.initial_balance]
    done, trunc = False, False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = test_env.step(action)
        equity.append(float(info['value']))

    metrics = compute_metrics(pd.Series(equity), annual_rf=0.04)

    del model, train_env, test_env

    return {
        'algo': 'SAC',
        'profile': profile_name,
        'seed': seed,
        'sharpe': float(metrics.get('Sharpe Ratio', np.nan)),
        'sortino': float(metrics.get('Sortino Ratio', np.nan)),
        'retorno_pct': float(metrics.get('Retorno Total', np.nan)),
        'cagr_pct': float(metrics.get('CAGR', np.nan)),
        'vol_anual_pct': float(metrics.get('Volatilidad Anual', np.nan)),
        'mdd_pct': float(metrics.get('Max Drawdown', np.nan)),
        'train_time_min': train_time_min,
    }


def main():
    print('=' * 70)
    print('RETRAIN SAC × N=5 SEEDS × 2 PERFILES × 500K PASOS')
    print('=' * 70)
    print(f'Hyperparams SAC: {SAC_HYPERPARAMS}')

    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f).get('runs', [])
        done = {(r['profile'], r['seed']) for r in results}
        print(f'Previos: {len(results)} runs completos')
    else:
        done = set()

    for profile_name in ['low_turnover', 'aggressive']:
        for seed in range(N_SEEDS):
            key = (profile_name, seed)
            if key in done:
                print(f'\n[SAC {profile_name} seed {seed}] ya hecho, skip')
                continue

            save_path = MODELS_DIR / f'best_model_academic_sac_{profile_name}_seed{seed}'
            print(f'\n[SAC {profile_name} seed {seed}] → {save_path.name}')
            result = train_one_run(profile_name, seed, save_path)
            results.append(result)
            print(f'  → Sharpe TEST: {result["sharpe"]:.4f}  Retorno: {result["retorno_pct"]:+.2f}%  MDD: {result["mdd_pct"]:+.2f}%')

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, 'w') as f:
                json.dump({'hyperparams': SAC_HYPERPARAMS, 'profiles': PROFILES, 'runs': results}, f, indent=2)

    print('\n' + '=' * 70)
    print('RESULTADOS FINALES SAC')
    print('=' * 70)
    for profile_name in ['low_turnover', 'aggressive']:
        runs = [r for r in results if r['profile'] == profile_name]
        if not runs:
            continue
        sharpes = np.array([r['sharpe'] for r in runs])
        retornos = np.array([r['retorno_pct'] for r in runs])
        mdds = np.array([r['mdd_pct'] for r in runs])
        print(f'\nSAC {profile_name} (N={len(runs)} seeds):')
        for r in sorted(runs, key=lambda x: x['seed']):
            print(f'  seed {r["seed"]}: Sharpe={r["sharpe"]:.4f}  Retorno={r["retorno_pct"]:+.2f}%  MDD={r["mdd_pct"]:+.2f}%')
        print(f'  Sharpe:  {sharpes.mean():.4f} ± {sharpes.std():.4f}')
        print(f'  Retorno: {retornos.mean():+.2f}% ± {retornos.std():.2f}%')
        print(f'  MDD:     {mdds.mean():+.2f}% ± {mdds.std():.2f}%')
    print('=' * 70)
    print(f'Resultados: {RESULTS_PATH}')


if __name__ == '__main__':
    main()
