"""
Standalone runner Optuna HPO PPO para TFM.

Decisiones P1-P11 (project_optuna_decisions.md):

- P4: 50 trials TPE sampler
- P6: MedianPruner agresivo (n_startup_trials=5, n_warmup_steps=100k)
- P7: maximize Sharpe puro
- P10: Storage SQLite persistente backend/hpo/optuna_study_ppo.db

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/run_optuna.py [n_trials]

Si study existe en SQLite, lo carga y continúa desde donde quedó (resiliente
a reboots del PC). Para reiniciar desde cero: borrar backend/hpo/optuna_study_ppo.db.

Coste estimado:
- 50 trials × 9 evals (3 folds × 3 seeds) × 200k pasos
- Con MedianPruner: ~50-70% trials pruned → ~30-50h efectivo
- Sobrevive reboot vía SQLite storage
"""

import argparse
import os
import sys
from pathlib import Path

# Permitir imports desde backend/src
BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

import optuna  # noqa: E402

from src.hpo.objective_ppo import objective_ppo  # noqa: E402


# Storage SQLite persistente (P10)
DEFAULT_STORAGE_PATH = BACKEND_ROOT / 'hpo' / 'optuna_study_ppo.db'
STUDY_NAME = 'ppo_hpo_tfm'

# Trials por defecto (P4)
DEFAULT_N_TRIALS = 50


def main(n_trials: int = DEFAULT_N_TRIALS, storage_path: Path = DEFAULT_STORAGE_PATH):
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f'sqlite:///{storage_path.as_posix()}'

    # P6: MedianPruner agresivo. Compatible con TPE.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=100_000,
        interval_steps=50_000,
    )

    # TPE sampler default Optuna (P5 cobertura 10 dims con 50 trials).
    # seed=None permite resume tras reboot sin repetir trials warmup
    # (TPE reconstruye prior bayesian desde storage SQLite).
    sampler = optuna.samplers.TPESampler(seed=None)

    study = optuna.create_study(
        storage=storage_url,
        study_name=STUDY_NAME,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    print(f"Storage: {storage_url}")
    print(f"Study: {STUDY_NAME}")
    print(f"Trials completados hasta ahora: {len(study.trials)}")
    if len(study.trials) > 0 and study.best_trial is not None:
        try:
            print(f"Best Sharpe so far: {study.best_value:.4f}")
        except ValueError:
            print("Best Sharpe so far: (no completed trials yet)")

    # Rutas datasets (relativo a backend/)
    features_path = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
    prices_path = str(BACKEND_ROOT / 'data' / 'original_prices.csv')

    print(f"Lanzando optimize con {n_trials} trials adicionales...")
    study.optimize(
        lambda trial: objective_ppo(trial, features_path, prices_path),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    print(f"Trials totales: {len(study.trials)}")
    print(f"Best Sharpe: {study.best_value:.4f}")
    print(f"Best hyperparams:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optuna HPO PPO standalone runner')
    parser.add_argument(
        'n_trials', type=int, nargs='?', default=DEFAULT_N_TRIALS,
        help=f'Número de trials a ejecutar (default {DEFAULT_N_TRIALS})',
    )
    parser.add_argument(
        '--storage', type=str, default=str(DEFAULT_STORAGE_PATH),
        help='Path SQLite storage Optuna',
    )
    args = parser.parse_args()
    main(n_trials=args.n_trials, storage_path=Path(args.storage))
