"""
Evaluación de Sharpe sobre subset val durante Optuna trial.

Reutiliza:
- PortfolioEnv (environment_trading.py) — entorno backtest
- Patrón predict-loop de scripts/diagnose_ppo.py:89-101
- compute_metrics de benchmarking/baselines.py:668 — Sharpe canónico TFM

Justificación métrica:
- Sharpe Ratio anualizado con rf=4% (consistencia con compute_metrics canónico
  usado en §4.4 tabla baselines)
- Objective Optuna = Sharpe sobre val (P7 decisión: single-objective)

Ver project_optuna_decisions.md P7 y P11.
"""

import numpy as np
import pandas as pd

from src.training_drl.environment_trading import PortfolioEnv
from src.benchmarking.baselines import compute_metrics


def evaluate_on_val(
    model,
    features_path: str,
    prices_path: str,
    start_idx: int,
    end_idx: int,
    varphi: float,
    gamma: float,
    reward_type: str = 'sharpe',
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    Evalúa modelo PPO sobre subset [start_idx, end_idx) de los datos.

    Crea PortfolioEnv con subset, ejecuta predict loop determinista,
    construye equity curve, calcula Sharpe canónico TFM (rf=4% anual).

    Parameters
    ----------
    model : stable_baselines3.PPO
        Modelo entrenado.
    features_path, prices_path : str
        Rutas CSVs.
    start_idx, end_idx : int
        Subset indices del dataset (sobre features dataframe).
    varphi, gamma : float
        Hyperparams recompensa del trial (deben coincidir con los del train env).
    reward_type, alpha, beta : str, float
        Forma de la recompensa. Default 'sharpe'.

    Returns
    -------
    float : Sharpe Ratio anualizado. NaN si backtest falla.
    """
    try:
        env = PortfolioEnv(
            features_path=features_path,
            prices_path=prices_path,
            start_idx=start_idx,
            end_idx=end_idx,
            phi=varphi,
            gamma=gamma,
            reward_type=reward_type,
            alpha=alpha,
            beta=beta,
        )

        obs, _ = env.reset()
        equity_log = [env.initial_balance]
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            equity_log.append(float(info['value']))

        # Sharpe canónico TFM via compute_metrics (rf=4% anual)
        equity_series = pd.Series(equity_log)
        metrics = compute_metrics(equity_series, annual_rf=0.04)
        sharpe = metrics.get('Sharpe Ratio', np.nan)

        return float(sharpe) if not np.isnan(sharpe) else -10.0

    except Exception as e:
        # Si backtest peta, devolver penalización extrema para que TPE descarte
        print(f"  [eval_on_val] Error: {e}")
        return -10.0
