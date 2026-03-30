"""
academic_evaluation.py
======================
Academic validation module for the TFM.

Main functions:
  - compute_mvp_weights()     -> Minimum Variance Portfolio (Markowitz 1952)
  - run_three_benchmark_backtest()  -> AI vs B&H vs MVP on test set
  - load_multiseed_results()        -> Multi-seed summary table
  - load_wfv_results()              -> Walk-forward validation summary table

Can be called from app_dashboard.py or run directly as a script.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stable_baselines3 import PPO

# Support both `python reports/academic_evaluation.py` and package import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.trading_env import PortfolioEnv


# ---------------------------------------------------------------------------
# Minimum Variance Portfolio (Markowitz, 1952)
# ---------------------------------------------------------------------------

def compute_mvp_weights(train_prices: pd.DataFrame) -> np.ndarray:
    """
    Compute Minimum Variance Portfolio weights via quadratic programming.

    Optimisation problem:
        min  w' Σ w
        s.t. Σ wᵢ = 1
             wᵢ ≥ 0   (long-only, no leverage — same constraint as the AI agent)

    Args:
        train_prices: DataFrame of close prices for the training period.

    Returns:
        Optimal weight array (sums to 1, all >= 0).
    """
    rets = train_prices.pct_change().dropna()
    cov  = rets.cov().values
    n    = cov.shape[0]

    def portfolio_variance(w):
        return float(w @ cov @ w)

    result = minimize(
        portfolio_variance,
        x0          = np.ones(n) / n,
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        bounds      = [(0.0, 1.0)] * n,
        method      = 'SLSQP',
        options     = {'ftol': 1e-12, 'maxiter': 1000},
    )

    if result.success:
        weights = np.clip(result.x, 0, 1)
        weights /= weights.sum()
        return weights
    else:
        print("Warning: MVP did not converge, falling back to 1/N.")
        return np.ones(n) / n


def simulate_mvp(test_prices: pd.DataFrame, mvp_weights: np.ndarray,
                 capital: float = 10_000) -> np.ndarray:
    """
    Simulate MVP strategy on the test period with fixed weights.

    Weights are calibrated on train and applied on test (out-of-sample).
    No rebalancing — classic MVP buy-and-hold with optimal weights.

    Returns:
        Array of cumulative portfolio values, length = len(test_prices) + 1.
    """
    daily_rets  = test_prices.pct_change().dropna()
    port_return = (daily_rets * mvp_weights).sum(axis=1)
    values      = capital * (1 + port_return).cumprod()
    return np.insert(values.values, 0, capital)


# ---------------------------------------------------------------------------
# Three-benchmark backtest
# ---------------------------------------------------------------------------

def run_three_benchmark_backtest(
    model_path: str = "models/best_model/best_model.zip",
    split_pct: float = 0.8,
    capital: float = 10_000,
    reward_mode: str = 'rolling_sharpe',
) -> dict:
    """
    Full backtest comparing:
      1. AI PPO (trained model)
      2. Buy & Hold 1/N (naive benchmark)
      3. Minimum Variance Portfolio — Markowitz (strong benchmark)

    MVP is calibrated on TRAIN prices and applied to TEST only,
    ensuring a rigorous out-of-sample evaluation.

    Returns:
        dict with keys: 'values_ai', 'values_bh', 'values_mvp',
        'mvp_weights', 'metrics', 'tickers', 'ai_weights_hist'
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    df_f      = pd.read_csv('data/features_normalizadas.csv', index_col=0)
    df_p      = pd.read_csv('data/precios_originales.csv', index_col=0)
    split_idx = int(len(df_f) * split_pct)

    # --- AI PPO ---
    env_ai = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        initial_balance=capital, start_idx=split_idx, reward_mode=reward_mode,
        episode_length=None, random_start=False,  # full test window, no randomness
    )
    model = PPO.load(model_path)
    obs, _ = env_ai.reset()
    done = False
    values_ai, weights_ai = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env_ai.step(action)
        values_ai.append(info['value'])
        weights_ai.append(info.get('weights', np.zeros(env_ai.n_assets)).copy())

    # --- Buy & Hold 1/N ---
    test_prices  = df_p.iloc[split_idx:].copy()
    train_prices = df_p.iloc[:split_idx].copy()
    bh_rets      = test_prices.pct_change().dropna().mean(axis=1)
    values_bh    = np.insert((capital * (1 + bh_rets).cumprod()).values, 0, capital)
    values_bh    = values_bh[:len(values_ai)]

    # --- MVP (calibrated on train, applied on test) ---
    mvp_weights  = compute_mvp_weights(train_prices)
    values_mvp   = simulate_mvp(test_prices, mvp_weights, capital)[:len(values_ai)]

    tickers          = [c.replace("_Close", "") for c in df_p.columns]
    mvp_weights_dict = {tickers[i]: float(mvp_weights[i]) for i in range(len(tickers))}
    print("MVP weights (Minimum Variance):")
    for t, w in sorted(mvp_weights_dict.items(), key=lambda x: -x[1]):
        print(f"  {t}: {w*100:.1f}%")

    # --- Metrics ---
    def _sharpe(vals):
        r = pd.Series(vals).pct_change().dropna()
        return float((r.mean() / r.std()) * np.sqrt(252)) if r.std() > 0 else 0.0

    def _max_dd(vals):
        s = pd.Series(vals)
        return float((s / s.cummax() - 1).min() * 100)

    def _sortino(vals):
        r = pd.Series(vals).pct_change().dropna()
        d = r[r < 0].std()
        return float((r.mean() / d) * np.sqrt(252)) if d > 0 else 0.0

    def _calmar(vals):
        r   = pd.Series(vals).pct_change().dropna()
        ann = r.mean() * 252
        dd  = abs(_max_dd(vals) / 100)
        return float(ann / dd) if dd > 0 else 0.0

    metrics = pd.DataFrame({
        'Metric': ['Total Return (%)', 'Sharpe Ratio', 'Sortino Ratio',
                   'Max Drawdown (%)', 'Calmar Ratio'],
        'AI PPO': [
            f"{(values_ai[-1]/capital - 1)*100:+.2f}%",
            f"{_sharpe(values_ai):.3f}",
            f"{_sortino(values_ai):.3f}",
            f"{_max_dd(values_ai):.2f}%",
            f"{_calmar(values_ai):.3f}",
        ],
        'B&H 1/N': [
            f"{(values_bh[-1]/capital - 1)*100:+.2f}%",
            f"{_sharpe(values_bh):.3f}",
            f"{_sortino(values_bh):.3f}",
            f"{_max_dd(values_bh):.2f}%",
            f"{_calmar(values_bh):.3f}",
        ],
        'MVP (Markowitz)': [
            f"{(values_mvp[-1]/capital - 1)*100:+.2f}%",
            f"{_sharpe(values_mvp):.3f}",
            f"{_sortino(values_mvp):.3f}",
            f"{_max_dd(values_mvp):.2f}%",
            f"{_calmar(values_mvp):.3f}",
        ],
    }).set_index('Metric')

    print("\n" + "=" * 55)
    print(metrics.to_string())
    print("=" * 55)

    return {
        'values_ai':      values_ai,
        'values_bh':      values_bh,
        'values_mvp':     values_mvp,
        'mvp_weights':    mvp_weights_dict,
        'metrics':        metrics,
        'tickers':        tickers,
        'ai_weights_hist': weights_ai,
    }


# ---------------------------------------------------------------------------
# Load saved results from train.py
# ---------------------------------------------------------------------------

def load_multiseed_results(path: str = 'models/multisemilla.csv') -> tuple | None:
    """Load multi-seed results generated by train.py:entrenar_multisemilla()."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    summary = pd.DataFrame({
        'Metric': ['Sharpe mean', 'Sharpe std', 'MaxDD mean', 'MaxDD std'],
        'Value': [
            f"{df['sharpe'].mean():.3f}",
            f"+/- {df['sharpe'].std():.3f}",
            f"{df['max_drawdown'].mean():.2%}",
            f"+/- {df['max_drawdown'].std():.2%}",
        ]
    })
    return df, summary


def load_wfv_results(path: str = 'models/wfv_resultados.csv') -> pd.DataFrame | None:
    """Load walk-forward results generated by train.py:walk_forward_validation()."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df['max_drawdown_pct'] = df['max_drawdown'] * 100
    return df


# ---------------------------------------------------------------------------
# Direct script execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_three_benchmark_backtest()

    plt.style.use('ggplot')
    plt.figure(figsize=(13, 6))
    plt.plot(result['values_ai'],  label='AI PPO',          color='#1f77b4', lw=2)
    plt.plot(result['values_bh'],  label='B&H 1/N',         color='#ff7f0e', ls='--')
    plt.plot(result['values_mvp'], label='MVP (Markowitz)',  color='#2ca02c', ls='-.')
    plt.title('Backtest: AI vs Buy & Hold vs Minimum Variance Portfolio')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Trading Days (Test set)')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'backtest_three_benchmarks.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Chart saved: {out}")
