"""
Retrain A2C multi-seed × 2 perfiles para evaluación robusta inter-seed.

Decisión P3 (project_optuna_decisions.md):
- A2C/SAC NO usan Optuna (alcance TFM: PPO es algoritmo central)
- Hyperparams calibrados manualmente (copiados de training_analysis.py
  rama A2C, secciones §4.8 memoria)
- N=5 seeds × 2 perfiles (low_turnover + aggressive) = 10 runs
- Eval cada uno sobre TEST intocable [train_split:n_total]

Coste: 10 × 1.5M pasos ≈ 5h CPU.

Resume robusto: persiste resultados parciales tras cada run.

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/retrain_a2c_multiseed.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.environment_trading import PortfolioEnv  # noqa: E402
from src.benchmarking.baselines import compute_metrics  # noqa: E402


# ── Configuración ───────────────────────────────────────────────────────────
N_SEEDS = 5
TRAIN_STEPS = 1_500_000  # presupuesto pleno A2C (on-policy)
TRAIN_SPLIT_PCT = 0.80

FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')

MODELS_DIR = BACKEND_ROOT / 'models'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
RESULTS_PATH = REPORTS_DIR / 'a2c_multiseed_results.json'

# Hyperparams A2C calibrados manualmente (training_analysis.py L127-142)
A2C_HYPERPARAMS = {
    'learning_rate': 3e-4,
    'n_steps': 32,         # vs SB3 default 5
    'gae_lambda': 0.95,    # vs SB3 default 1.0
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
}

NET_ARCH = [256, 256]

# Perfiles de riesgo (varphi, gamma) — copiados de risk_profiles.py
PROFILES = {
    'low_turnover': dict(varphi=0.02, gamma=0.020),
    'aggressive':   dict(varphi=0.01, gamma=0.005),
}


def train_one_run(profile_name: str, seed: int, save_path: Path) -> dict:
    """Entrena 1 A2C con seed y perfil dado. Eval sobre TEST."""
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

    model = A2C(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=A2C_HYPERPARAMS['learning_rate'],
        n_steps=A2C_HYPERPARAMS['n_steps'],
        gae_lambda=A2C_HYPERPARAMS['gae_lambda'],
        ent_coef=A2C_HYPERPARAMS['ent_coef'],
        vf_coef=A2C_HYPERPARAMS['vf_coef'],
        max_grad_norm=A2C_HYPERPARAMS['max_grad_norm'],
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
        'algo': 'A2C',
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
    print('RETRAIN A2C × N=5 SEEDS × 2 PERFILES × 1.5M PASOS')
    print('=' * 70)
    print(f'Hyperparams A2C: {A2C_HYPERPARAMS}')

    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f).get('runs', [])
        done = {(r['profile'], r['seed']) for r in results}
        print(f'Previos: {len(results)} runs completos')
    else:
        done = set()

    # Orden: por perfil completo, luego siguiente. Permite ver convergencia
    for profile_name in ['low_turnover', 'aggressive']:
        for seed in range(N_SEEDS):
            key = (profile_name, seed)
            if key in done:
                print(f'\n[A2C {profile_name} seed {seed}] ya hecho, skip')
                continue

            save_path = MODELS_DIR / f'best_model_academic_a2c_{profile_name}_seed{seed}'
            print(f'\n[A2C {profile_name} seed {seed}] → {save_path.name}')
            result = train_one_run(profile_name, seed, save_path)
            results.append(result)
            print(f'  → Sharpe TEST: {result["sharpe"]:.4f}  Retorno: {result["retorno_pct"]:+.2f}%  MDD: {result["mdd_pct"]:+.2f}%')

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, 'w') as f:
                json.dump({'hyperparams': A2C_HYPERPARAMS, 'profiles': PROFILES, 'runs': results}, f, indent=2)

    # Reporte final por perfil
    print('\n' + '=' * 70)
    print('RESULTADOS FINALES A2C (N=5 seeds × 2 perfiles)')
    print('=' * 70)
    for profile_name in ['low_turnover', 'aggressive']:
        runs = [r for r in results if r['profile'] == profile_name]
        if not runs:
            continue
        sharpes = np.array([r['sharpe'] for r in runs])
        retornos = np.array([r['retorno_pct'] for r in runs])
        mdds = np.array([r['mdd_pct'] for r in runs])
        print(f'\nA2C {profile_name} (N={len(runs)} seeds):')
        for r in sorted(runs, key=lambda x: x['seed']):
            print(f'  seed {r["seed"]}: Sharpe={r["sharpe"]:.4f}  Retorno={r["retorno_pct"]:+.2f}%  MDD={r["mdd_pct"]:+.2f}%')
        print(f'  Sharpe:  {sharpes.mean():.4f} ± {sharpes.std():.4f}')
        print(f'  Retorno: {retornos.mean():+.2f}% ± {retornos.std():.2f}%')
        print(f'  MDD:     {mdds.mean():+.2f}% ± {mdds.std():.2f}%')
    print('=' * 70)
    print(f'Resultados: {RESULTS_PATH}')


if __name__ == '__main__':
    main()
