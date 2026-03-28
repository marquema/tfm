import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):

    def __init__(self, features_path, prices_path, initial_balance=10000, commission=0.001, start_idx=0, end_idx=None):
        super().__init__()

        # Cargar datos y eliminar NaNs
        df_f = pd.read_csv(features_path, index_col=0).dropna()

        # Sincronizar precios con las features que han quedado vivas
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        if end_idx is None:
            end_idx = len(df_f)

        # Slices para train / test
        self.df_features = df_f.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        self.df_precios  = df_p.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)

        print(f"Entorno creado con {len(self.df_features)} pasos (índice {start_idx} → {end_idx}).")

        self.n_assets       = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission      = commission

        # Espacios de Gymnasium
        # Acción: pesos continuos [0,1] para cada activo (se normalizan internamente a suma=1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        # Observación: vector de features normalizadas (dimensión dinámica según el CSV cargado)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df_features.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step        = 0
        self.balance             = self.initial_balance
        self.portfolio_value     = self.initial_balance
        self.max_portfolio_value = self.initial_balance   # Marca de agua para el drawdown
        self.weights             = np.zeros(self.n_assets)

        obs = self.df_features.iloc[self.current_step].values
        return obs.astype(np.float32), {}

    def step(self, action):
        """
        Función de recompensa con penalización por drawdown (Sharpe-Drawdown balance):

            R_t = log_return(t) − φ · drawdown(t)

        - log_return: retorno logarítmico del portfolio en el paso t
        - drawdown:   caída relativa desde el máximo histórico (marca de agua)
        - φ = 0.5:    equilibrio agresivo/conservador; el agente puede arriesgar
                      pero paga un coste proporcional a cuánto se aleja del máximo

        Con el nuevo espacio de observación (features técnicas + dividend dynamics),
        el agente dispone de señales anticipatorias de riesgo (kurtosis, div_volatility,
        corr_IBIT_IVV) que le permiten reducir exposición ANTES de que ocurra el drawdown,
        lo que hace que la penalización sea accionable y no solo reactiva.
        """
        # 1. Estado previo
        precios_hoy    = self.df_precios.iloc[self.current_step].values
        valor_base     = max(self.portfolio_value, 1e-6)

        # 2. Normalizar pesos: el agente emite logits, proyectamos al simplex
        new_weights = action / (np.sum(action) + 1e-6)

        # 3. Coste de transacción: penaliza rebalanceos erráticos por ruido
        diff_weights          = np.abs(new_weights - self.weights)
        cost                  = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost

        # 4. Fin del episodio si no hay más datos
        if self.current_step >= len(self.df_features) - 1:
            obs = self.df_features.iloc[self.current_step].values
            return obs.astype(np.float32), 0.0, True, False, {
                "value": self.portfolio_value, "drawdown": 0.0
            }

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1

        # 5. Evolución del mercado: aplicamos los retornos de mañana
        precios_manana   = self.df_precios.iloc[self.current_step].values
        retornos_activos = precios_manana / precios_hoy
        self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        self.portfolio_value = max(self.portfolio_value, 1e-6)

        self.weights = new_weights

        # 6. Retorno logarítmico del paso
        log_return = float(np.log(self.portfolio_value / valor_base + 1e-8))

        # 7. Drawdown desde la marca de agua
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # 8. Recompensa: R_t = log_return − φ · drawdown
        phi    = 0.5
        reward = log_return - phi * current_drawdown

        # 9. Condición de quiebra (pérdida > 90% del capital inicial)
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
        Variante alternativa: solo log-return sin penalización por drawdown.
        Útil para comparar el impacto del risk-shaping en ablation studies.
        """
        new_weights  = action / (np.sum(action) + 1e-6)
        precios_hoy  = self.df_precios.iloc[self.current_step].values

        diff_weights          = np.abs(new_weights - self.weights)
        cost                  = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost
        valor_antes           = self.portfolio_value

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1

        precios_manana   = self.df_precios.iloc[self.current_step].values
        retornos_activos = precios_manana / precios_hoy
        self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        self.weights         = new_weights
        self.portfolio_value = max(self.portfolio_value, 1e-6)
        valor_antes          = max(valor_antes, 1e-6)

        reward = float(np.log(self.portfolio_value / valor_antes + 1e-8))

        if self.portfolio_value < (self.initial_balance * 0.1):
            done   = True
            reward = -10.0

        obs = self.df_features.iloc[self.current_step].values
        return obs.astype(np.float32), reward, done, False, {"value": self.portfolio_value}
