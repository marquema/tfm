"""
Objective function Optuna PPO TFM.

Diseño basado en decisiones P1-P11 (project_optuna_decisions.md):

- P1 (eval set): Walk-forward CV 3 folds rolling dentro train (80% dataset).
  Test (20% final) INTOCABLE por Optuna.
- P2 (multi-seed): N=3 seeds por trial. Objective = mean Sharpe sobre folds × seeds.
- P4 (trials): 50 trials.
- P5 (space): 10 dims (8 PPO + varphi + gamma). Ver space.py.
- P6 (pruner): MedianPruner gestionado en run_optuna.py.
- P7 (métrica): Sharpe puro (single-objective).
- P8 (folds): 3 rolling, NO expanding (folds comparables → ranking limpio).

Coste por trial: 3 folds × 3 seeds = 9 entrenamientos PPO con N_HPO_STEPS cada uno.
Con MedianPruner agresivo: ~50-70% trials pruned antes de completar.

Justificación N_HPO_STEPS=200000:
- Suficiente para PPO converger razonablemente (TFM reporta R_eval positivo a 200k)
- Permite MedianPruner activar (n_warmup_steps=100000)
- Best config se re-entrena con 1.5M pasos post-Optuna (P11)

Documentación memoria final §3.X Optuna.
"""

import numpy as np
import optuna
import pandas as pd
from stable_baselines3 import PPO

from src.training_drl.environment_trading import PortfolioEnv
from src.hpo.space import sample_hyperparams, validate_batch_size_compatibility
from src.hpo.eval_metrics import evaluate_on_val


# ────────────────────────────────────────────────────────────────────────────
# Configuración global del objective. Justificación cada constante:
# ────────────────────────────────────────────────────────────────────────────

# Pasos por fold por seed durante HPO. Trade-off coste/calidad.
# 200k pasos: PPO converge razonablemente, MedianPruner puede actuar tras 100k.
# Best config post-Optuna se re-entrena con 1.5M pasos (P11).
N_HPO_STEPS = 200_000

# Seeds por trial (P2). Reduce ruido inter-seed inherente DRL (Henderson 2018).
N_SEEDS = 3

# Folds CV walk-forward (P1, P8). Rolling.
N_FOLDS = 3

# Split train/test: 80/20 (consistente con TFM existente).
# Test idx >= TRAIN_END_IDX está intocable por Optuna.
TRAIN_SPLIT_PCT = 0.8


def make_folds(n_train: int, n_folds: int = N_FOLDS) -> list:
    """
    Construye índices walk-forward rolling sobre el rango train [0, n_train).

    Justificación P8 (rolling NO expanding): folds comparables (mismo tamaño
    train) permite promedio Sharpe limpio como objective. Expanding rompe
    fold-independence por solapamiento creciente.

    Estructura 3 folds reales (n_train=2313, valores calculados):
        Fold 0: train [0:1503], val [1503:1803]
        Fold 1: train [255:1758], val [1758:2058]
        Fold 2: train [510:2013], val [2013:2313]

    Ventana train fija ~1500 días (~6 años), sliding ~255 días.

    NOTA — sliding overlapping vs anchored non-overlap:
    Esta implementación usa SLIDING window con solapamiento parcial entre
    train ventanas (e.g. train fold 2 contiene parte del val fold 1). Esto
    es estándar en literatura ML para series temporales (López de Prado
    2018, Hyndman & Athanasopoulos) cuando el objetivo es ranking de
    hyperparams: cada fold se entrena desde cero, sin transferir
    conocimiento entre folds, y el promedio de 3 Sharpes evalúa robustez
    de la config a distintas ventanas. La alternativa "pure non-overlap"
    requeriría train < 600 días por fold, insuficiente para que PPO
    converja con 200k pasos. Dentro de cada fold, train SIEMPRE precede
    temporalmente a val (no hay leakage forward).

    Parameters
    ----------
    n_train : int
        Tamaño del rango train.
    n_folds : int
        Número de folds (default 3, decisión P8).

    Returns
    -------
    list of (train_start, train_end, val_start, val_end) tuples.
    """
    train_size = int(n_train * 0.65)
    val_size = int(n_train * 0.13)

    folds = []
    if n_folds > 1:
        max_train_start = n_train - train_size - val_size
        stride = max_train_start // (n_folds - 1)
    else:
        stride = 0

    for i in range(n_folds):
        train_start = i * stride
        train_end = train_start + train_size
        val_start = train_end
        val_end = min(val_start + val_size, n_train)
        folds.append((train_start, train_end, val_start, val_end))

    return folds


def train_and_eval_one_fold(
    hyperparams: dict,
    seed: int,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    features_path: str,
    prices_path: str,
) -> float:
    """
    Una unidad de trabajo: 1 fold × 1 seed.

    1. Crea PortfolioEnv train con subset [train_start, train_end)
    2. Construye PPO con hyperparams custom + seed
    3. Entrena N_HPO_STEPS pasos
    4. Evalúa Sharpe sobre val [val_start, val_end) via eval_metrics

    Justificación PPO directo (no _build_drl_model):
    _build_drl_model solo expone lr y n_envs; resto hardcoded. Para Optuna
    necesitamos control total → instanciamos PPO de SB3 directamente.

    Returns
    -------
    float : Sharpe Ratio sobre val. -10.0 si falla.
    """
    n_steps = hyperparams['n_steps']
    batch_size = validate_batch_size_compatibility(n_steps, hyperparams['batch_size'])

    train_env = PortfolioEnv(
        features_path=features_path,
        prices_path=prices_path,
        start_idx=train_start,
        end_idx=train_end,
        phi=hyperparams['varphi'],
        gamma=hyperparams['gamma'],
        reward_type='sharpe',
    )

    # PPO con hyperparams sampleados. policy_kwargs net_arch hardcoded
    # [256, 256] (excluido del space P5 por simplicidad).
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=hyperparams['learning_rate'],
        n_steps=n_steps,
        batch_size=batch_size,
        clip_range=hyperparams['clip_range'],
        ent_coef=hyperparams['ent_coef'],
        gae_lambda=hyperparams['gae_lambda'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        verbose=0,
        device='cpu',
    )

    model.learn(total_timesteps=N_HPO_STEPS, progress_bar=False)

    sharpe = evaluate_on_val(
        model=model,
        features_path=features_path,
        prices_path=prices_path,
        start_idx=val_start,
        end_idx=val_end,
        varphi=hyperparams['varphi'],
        gamma=hyperparams['gamma'],
        reward_type='sharpe',
    )

    del model, train_env
    return sharpe


def objective_ppo(
    trial: optuna.Trial,
    features_path: str = 'data/normalized_features.csv',
    prices_path: str = 'data/original_prices.csv',
) -> float:
    """
    Función objective Optuna PPO.

    Flujo (P1+P2+P7+P8):
    1. Sample hyperparams del search space (10 dims).
    2. Cargar dataset, identificar train_end_idx (80%).
    3. Construir 3 folds walk-forward rolling.
    4. Por cada fold × seed: train PPO + eval Sharpe val.
    5. Pruner intermedio: report mean Sharpe parcial cada fold completado.
       Si pruner activa → optuna.TrialPruned.
    6. Return mean Sharpe over (folds × seeds).

    Parameters
    ----------
    trial : optuna.Trial
        Trial Optuna.
    features_path, prices_path : str
        Rutas datasets.

    Returns
    -------
    float : mean Sharpe sobre todos los (folds, seeds).

    Raises
    ------
    optuna.TrialPruned : si pruner decide matar trial temprano.
    """
    hyperparams = sample_hyperparams(trial)

    df = pd.read_csv(features_path, index_col=0)
    n_total = len(df)
    train_end_idx = int(n_total * TRAIN_SPLIT_PCT)
    del df

    folds = make_folds(train_end_idx, n_folds=N_FOLDS)

    fold_sharpes = []
    all_sharpes = []

    for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds):
        seed_sharpes = []
        for seed_idx in range(N_SEEDS):
            seed = 1000 * fold_idx + seed_idx
            sharpe = train_and_eval_one_fold(
                hyperparams=hyperparams,
                seed=seed,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                features_path=features_path,
                prices_path=prices_path,
            )
            seed_sharpes.append(sharpe)
            all_sharpes.append(sharpe)

        fold_mean = float(np.mean(seed_sharpes))
        fold_sharpes.append(fold_mean)

        # Pruner intermedio (cada fold completado).
        # step = pasos cómputo acumulados (no fold_idx) para que MedianPruner
        # con n_warmup_steps=100000 active correctamente tras fold 0.
        # Cada fold ejecuta N_SEEDS entrenamientos de N_HPO_STEPS pasos.
        steps_completed = (fold_idx + 1) * N_HPO_STEPS * N_SEEDS
        trial.report(fold_mean, step=steps_completed)
        if trial.should_prune():
            raise optuna.TrialPruned()

    final_sharpe = float(np.mean(all_sharpes))
    return final_sharpe
