"""
Retrain PPO best Optuna config × N=5 seeds × 1.5M pasos.

Decision P11 (project_optuna_decisions.md):
- Carga best hyperparams de SQLite (study ppo_hpo_tfm)
- Entrena N=5 seeds independientes sobre TODO train [0, 2313]
- Eval cada modelo sobre TEST intocable [2313, 2892]
- Reporta media ± std de Sharpe sobre N=5 seeds (Henderson 2018 compliant)
- Sobre TEST: primera vez que toca ese subset (Optuna usó val interno)

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/retrain_optuna_best.py

Coste: 5 seeds × 1.5M pasos ≈ 7-9h CPU.

Resume robusto: cada seed guarda modelo. Si script muere, re-lanzar continúa
desde donde quedó (skip seeds ya entrenados).
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


# ── Configuración ───────────────────────────────────────────────────────────
STUDY_NAME = 'ppo_hpo_tfm'
STORAGE = f'sqlite:///{(BACKEND_ROOT / "hpo" / "optuna_study_ppo.db").as_posix()}'

N_SEEDS = 5
TRAIN_STEPS_FINAL = 1_500_000  # presupuesto pleno (vs 200k HPO)
TRAIN_SPLIT_PCT = 0.80

FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')

MODELS_DIR = BACKEND_ROOT / 'models'
REPORTS_DIR = BACKEND_ROOT / 'src' / 'reports'
RESULTS_PATH = REPORTS_DIR / 'optuna_retrain_results.json'

NET_ARCH = [256, 256]


def load_best_params() -> dict:
    """Carga best params Optuna SQLite."""
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    print(f'Cargado study {STUDY_NAME}: {len(study.trials)} trials')
    print(f'Best Sharpe (val Optuna): {study.best_value:.4f}')
    print(f'Best trial: #{study.best_trial.number}')
    return dict(study.best_params)


def train_one_seed(best_params: dict, seed: int, save_path: Path) -> dict:
    """
    Entrena 1 PPO sobre todo train con best params + seed.
    Eval sobre test intocable.
    """
    n_steps = best_params['n_steps']
    batch_size = validate_batch_size_compatibility(n_steps, best_params['batch_size'])

    df = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    print(f'  Train [0:{train_end_idx}), Test [{train_end_idx}:{n_total}]')

    # Entorno train
    train_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=0,
        end_idx=train_end_idx,
        phi=best_params['varphi'],
        gamma=best_params['gamma'],
        reward_type='sharpe',
    )

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=best_params['learning_rate'],
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=best_params['clip_range'],
        ent_coef=best_params['ent_coef'],
        gae_lambda=best_params['gae_lambda'],
        vf_coef=best_params['vf_coef'],
        max_grad_norm=best_params['max_grad_norm'],
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=seed,
        verbose=0,
        device='cpu',
    )

    t0 = time.time()
    print(f'  Entrenando {TRAIN_STEPS_FINAL:,} pasos (seed={seed})...')
    model.learn(total_timesteps=TRAIN_STEPS_FINAL, progress_bar=False)
    train_time_min = (time.time() - t0) / 60
    print(f'  Train OK en {train_time_min:.1f} min')

    # Save modelo
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path / 'best_model.zip'))

    # Eval sobre TEST intocable
    test_env = PortfolioEnv(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        start_idx=train_end_idx,
        end_idx=n_total,
        phi=best_params['varphi'],
        gamma=best_params['gamma'],
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
    print('RETRAIN PPO BEST OPTUNA × N=5 SEEDS × 1.5M PASOS')
    print('=' * 70)

    best_params = load_best_params()
    print('\nBest params:')
    for k, v in best_params.items():
        print(f'  {k}: {v}')

    # Resume: si ya hay resultados parciales, cargarlos
    results = []
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f).get('runs', [])
        done_seeds = {r['seed'] for r in results}
        print(f'\nResultados previos: {len(results)} seeds completados ({sorted(done_seeds)})')
    else:
        done_seeds = set()

    for seed in range(N_SEEDS):
        if seed in done_seeds:
            print(f'\n[seed {seed}] ya hecho, skip')
            continue

        save_path = MODELS_DIR / f'best_model_academic_OPTUNA_seed{seed}'
        print(f'\n[seed {seed}] modelo → {save_path.name}')
        result = train_one_seed(best_params, seed, save_path)
        results.append(result)
        print(f'  → Sharpe TEST: {result["sharpe"]:.4f}  Retorno: {result["retorno_pct"]:+.2f}%  MDD: {result["mdd_pct"]:+.2f}%')

        # Persist parcial tras cada seed (resume robusto)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump({'best_params': best_params, 'runs': results}, f, indent=2)

    # Reporte final
    sharpes = np.array([r['sharpe'] for r in results])
    retornos = np.array([r['retorno_pct'] for r in results])
    mdds = np.array([r['mdd_pct'] for r in results])

    print('\n' + '=' * 70)
    print('RESULTADOS FINALES (N=5 seeds, TEST intocable)')
    print('=' * 70)
    for r in sorted(results, key=lambda x: x['seed']):
        print(f'  seed {r["seed"]}:  Sharpe={r["sharpe"]:.4f}  Retorno={r["retorno_pct"]:+.2f}%  MDD={r["mdd_pct"]:+.2f}%')
    print('-' * 70)
    print(f'  Sharpe:  {sharpes.mean():.4f} ± {sharpes.std():.4f}')
    print(f'  Retorno: {retornos.mean():+.2f}% ± {retornos.std():.2f}%')
    print(f'  MDD:     {mdds.mean():+.2f}% ± {mdds.std():.2f}%')
    print('=' * 70)
    print(f'Resultados guardados: {RESULTS_PATH}')


if __name__ == '__main__':
    main()
