import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque


class PortfolioEnv(gym.Env):
    """
    Multi-asset portfolio environment for PPO training.

    Key design decisions
    --------------------
    1. Action space: Box(0, 1)^n_assets — raw allocations, projected to
       the probability simplex.  Cash = 1 - sum(weights).
       An all-zero action is 100% cash (stable, earns 0%).

    2. episode_length / random_start (training mode):
       Training uses short episodes (default 252 days = 1 trading year)
       with a random start drawn uniformly from the available window.
       This lets the agent see ~1 190 different market regimes in 300k
       steps instead of only ~57 full-dataset passes, which is the key
       driver of convergence.
       Evaluation/backtest uses episode_length=None, random_start=False
       to run the full test window uninterrupted.

    3. Reward modes:
       'rolling_sharpe'  (default)  Rolling 60-day Sharpe contribution.
                                    Stationary reward — doesn't accumulate
                                    over the episode, directly optimises
                                    the metric that matters academically.
                                    Based on Moody & Saffell (2001).
       'sharpe_drawdown'            R_t = log_return - 0.5*drawdown
                                    Legacy mode kept for ablation study.
       'log_return'                 R_t = log_return (pure baseline).

    4. Cash model: portfolio value = risky_pnl + cash_pnl.
       Cash earns 0%; avoids the portfolio collapsing to zero if the agent
       outputs near-zero actions.
    """

    def __init__(
        self,
        features_path: str,
        prices_path: str,
        initial_balance: float = 10_000,
        commission: float = 0.001,
        start_idx: int = 0,
        end_idx: int | None = None,
        reward_mode: str = 'rolling_sharpe',
        episode_length: int | None = 252,
        random_start: bool = True,
    ):
        super().__init__()

        df_f = pd.read_csv(features_path, index_col=0).dropna()
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        if end_idx is None:
            end_idx = len(df_f)

        self.df_features = (
            df_f.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        )
        self.df_precios = (
            df_p.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        )

        self.n_assets        = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission      = commission
        self.reward_mode     = reward_mode
        self.episode_length  = episode_length
        self.random_start    = random_start

        # Rolling window size for rolling_sharpe reward
        self._sharpe_window = 60

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.df_features.shape[1],), dtype=np.float32,
        )

        n_avail = len(self.df_features)
        ep_len  = episode_length if episode_length is not None else n_avail
        print(
            f"[PortfolioEnv] {n_avail} steps | assets={self.n_assets} "
            f"| obs={self.df_features.shape[1]} | reward='{reward_mode}' "
            f"| ep_len={ep_len} | random_start={random_start}"
        )

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        n      = len(self.df_features)
        ep_len = self.episode_length if self.episode_length is not None else n

        if self.random_start and ep_len < n:
            # Random start: sample a different 1-year market window each episode.
            # The np_random from Gymnasium is seeded properly via super().reset().
            self._ep_start = int(self.np_random.integers(0, n - ep_len))
        else:
            self._ep_start = 0

        self._ep_end = min(self._ep_start + ep_len, n)
        self.current_step = self._ep_start

        self.portfolio_value     = float(self.initial_balance)
        self.max_portfolio_value = float(self.initial_balance)
        self.weights             = np.zeros(self.n_assets, dtype=np.float32)

        # Buffer for rolling Sharpe reward
        self._rets_buf: deque = deque(maxlen=self._sharpe_window)

        obs = self.df_features.iloc[self._ep_start].values
        return obs.astype(np.float32), {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """
        One trading day.

        P&L mechanic
        ------------
            risky_fraction = clamp(sum(action), 0, 1)
            w_i            = action_i / sum(action) * risky_fraction
            cash           = 1 - risky_fraction
            portfolio_new  = sum(w_i * price_return_i) * pv + cash * pv - costs
        """
        action     = np.clip(action, 0.0, None)
        action_sum = float(np.sum(action))

        if action_sum > 1e-6:
            # risky_fraction in (0, 1]: how much of the portfolio is invested
            risky_fraction = min(action_sum, 1.0)
            risky_weights  = (action / action_sum) * risky_fraction
        else:
            risky_weights  = np.zeros(self.n_assets, dtype=np.float32)

        cash_weight = max(1.0 - float(np.sum(risky_weights)), 0.0)

        # Transaction costs — proportional to daily turnover
        turnover = float(np.sum(np.abs(risky_weights - self.weights)))
        cost     = turnover * self.portfolio_value * self.commission
        self.portfolio_value -= cost
        self.portfolio_value  = max(self.portfolio_value, 1e-8)

        # End-of-episode guard
        done = self.current_step >= self._ep_end - 1
        if done:
            obs = self.df_features.iloc[self.current_step].values
            return obs.astype(np.float32), 0.0, True, False, {
                "value": self.portfolio_value, "drawdown": 0.0,
                "weights": self.weights,
            }

        prev_value        = self.portfolio_value
        self.current_step += 1

        # Apply market returns
        p_prev = self.df_precios.iloc[self.current_step - 1].values
        p_next = self.df_precios.iloc[self.current_step].values

        # Pre-launch assets (price==0 or unchanged) earn 0%
        safe_prev    = np.where(p_prev > 0, p_prev, 1.0)
        asset_rets   = np.where(p_prev > 0, p_next / safe_prev, 1.0)

        risky_pnl            = float(np.dot(risky_weights, asset_rets)) * self.portfolio_value
        cash_pnl             = cash_weight * self.portfolio_value
        self.portfolio_value = max(risky_pnl + cash_pnl, 1e-8)
        self.weights         = risky_weights.astype(np.float32)

        # Log-return (clean formula, no bias)
        log_return = float(np.log(self.portfolio_value / max(prev_value, 1e-8)))
        self._rets_buf.append(log_return)

        # Drawdown from high-water mark
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = float(
            (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        )

        # ------ Reward ------------------------------------------------
        # Daily risk-free rate used in all modes to break the cash equilibrium.
        # Cash earns 0% < rf, so any mode gives a slightly negative reward for
        # doing nothing.  4% annual ÷ 252 trading days ≈ US T-bill baseline.
        _DAILY_RF = 0.04 / 252          # ≈ 0.0159 % per day

        if self.reward_mode == 'rolling_sharpe':
            # Excess Sharpe Ratio reward (Moody & Saffell, 2001).
            # Key fix: subtract daily rf from mu so cash (mu≈0) gives a
            # negative reward instead of the degenerate 0 equilibrium that
            # made the agent prefer cash indefinitely.
            # _MIN_SIG = 0.2% daily vol ≈ BND-level; prevents division by
            # near-zero when returns are flat (cash or very stable assets).
            _MIN_SIG = 0.002             # 0.2 % daily vol floor
            buf  = np.array(self._rets_buf)
            mu   = float(buf.mean())
            sig  = float(buf.std()) if len(buf) > 1 else _MIN_SIG
            sig  = max(sig, _MIN_SIG)
            reward = (mu - _DAILY_RF) / sig
            # Safety net: only penalise extreme drawdowns (> 10%)
            if current_drawdown > 0.10:
                reward -= 0.5 * (current_drawdown - 0.10)

        elif self.reward_mode == 'sharpe_drawdown':
            # Ablation: excess log_return over rf minus drawdown penalty
            reward = (log_return - _DAILY_RF) - 0.5 * current_drawdown

        else:
            # 'log_return' — excess return over risk-free (pure baseline)
            reward = log_return - _DAILY_RF

        done = self.current_step >= self._ep_end - 1

        obs  = self.df_features.iloc[self.current_step].values
        info = {
            "value":    self.portfolio_value,
            "drawdown": current_drawdown,
            "weights":  self.weights,
        }
        return obs.astype(np.float32), float(reward), done, False, info
