import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):

    def __init__(self, features_path, prices_path, initial_balance=10000, commission=0.001,
                 start_idx=0, end_idx=None, reward_mode='sharpe_drawdown'):
        """
        Args:
            features_path:   CSV con features normalizadas (observación del agente)
            prices_path:     CSV con precios de cierre originales (mecánica P&L)
            initial_balance: Capital inicial en $
            commission:      Coste por rebalanceo (fracción del volumen movido)
            start_idx:       Índice inicial del slice (train / test)
            end_idx:         Índice final del slice
            reward_mode:     'sharpe_drawdown' → R = log_return − φ·drawdown  (estrategia principal)
                             'log_return'      → R = log_return                (baseline para ablation)
        """
        super().__init__()

        df_f = pd.read_csv(features_path, index_col=0).dropna()
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        if end_idx is None:
            end_idx = len(df_f)

        self.df_features = df_f.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        self.df_precios  = df_p.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)

        print(f"Entorno creado con {len(self.df_features)} pasos "
              f"(índice {start_idx} → {end_idx}) | reward_mode='{reward_mode}'")

        self.n_assets        = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission      = commission
        self.reward_mode     = reward_mode

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df_features.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step        = 0
        self.balance             = self.initial_balance
        self.portfolio_value     = self.initial_balance
        self.max_portfolio_value = self.initial_balance  # Marca de agua para drawdown
        self.weights             = np.zeros(self.n_assets)

        obs = self.df_features.iloc[self.current_step].values
        return obs.astype(np.float32), {}

    def step(self, action):
        """
        Ejecuta un paso de rebalanceo y devuelve la recompensa según reward_mode:

        'sharpe_drawdown':
            R_t = log_return(t) − φ · drawdown(t)
            El agente maximiza retorno pero paga un coste proporcional a la caída
            desde el máximo histórico. Con las features de riesgo disponibles
            (kurtosis, div_volatility, corr_IBIT_IVV), puede anticipar el drawdown
            antes de que ocurra.

        'log_return':
            R_t = log_return(t)
            Baseline puro sin penalización. Útil para el ablation study:
            comparar si el risk-shaping realmente mejora el Sharpe y reduce el drawdown.
        """
        precios_hoy = self.df_precios.iloc[self.current_step].values
        valor_base  = max(self.portfolio_value, 1e-6)

        # Proyectar acción al simplex (pesos suman 1)
        new_weights = action / (np.sum(action) + 1e-6)

        # Coste de transacción: penaliza rebalanceos erráticos
        diff_weights          = np.abs(new_weights - self.weights)
        cost                  = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost

        # Fin del episodio si no hay más datos
        if self.current_step >= len(self.df_features) - 1:
            obs = self.df_features.iloc[self.current_step].values
            return obs.astype(np.float32), 0.0, True, False, {
                "value": self.portfolio_value, "drawdown": 0.0
            }

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1

        # Aplicar retornos del mercado
        precios_manana   = self.df_precios.iloc[self.current_step].values
        retornos_activos = precios_manana / precios_hoy
        self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        self.portfolio_value = max(self.portfolio_value, 1e-6)

        self.weights = new_weights

        # Retorno logarítmico del paso
        log_return = float(np.log(self.portfolio_value / valor_base + 1e-8))

        # Drawdown desde la marca de agua
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # Recompensa según modo
        if self.reward_mode == 'sharpe_drawdown':
            phi    = 0.5
            reward = log_return - phi * current_drawdown
        else:  # 'log_return'
            reward = log_return

        # Condición de quiebra (pérdida > 90%)
        if self.portfolio_value < (self.initial_balance * 0.1):
            done   = True
            reward = -10.0

        obs  = self.df_features.iloc[self.current_step].values
        info = {
            "value":    self.portfolio_value,
            "drawdown": current_drawdown,
            "weights":  self.weights,
        }
        return obs.astype(np.float32), float(reward), done, False, info

    def step_simplex(self, action):
        """
        Alias de step() con reward_mode='log_return'.
        Se mantiene por compatibilidad con código externo.
        """
        original_mode    = self.reward_mode
        self.reward_mode = 'log_return'
        result           = self.step(action)
        self.reward_mode = original_mode
        return result
