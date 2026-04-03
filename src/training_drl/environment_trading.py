"""
Entorno de trading basado en Gymnasium para gestión de carteras con DRL.

Implementa un entorno compatible con Stable-Baselines3 donde el agente
decide la asignación de pesos entre N activos en cada paso temporal.
La recompensa combina Sharpe rolling, penalización por drawdown y
penalización por turnover para fomentar estrategias estables.

Referencia: OpenAI Gym / Gymnasium — interfaz estándar para RL.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Entorno Gymnasium para asignación dinámica de carteras con Deep RL.

    El agente recibe como observación las features de mercado concatenadas
    con el estado interno de la cartera (pesos actuales + retorno acumulado).
    La acción es un vector de pesos no negativos que se normaliza a suma 1.

    La función de recompensa compuesta es:
        R_t = Sharpe_rolling - phi * MDD(t) - gamma * Turnover(t)

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas (índice = fecha).
    prices_path : str
        Ruta al CSV de precios originales (índice = fecha).
    initial_balance : float
        Capital inicial de la cartera en USD.
    commission : float
        Comisión proporcional por rebalanceo (fracción del volumen operado).
    start_idx : int
        Índice de inicio del subconjunto temporal (para splits train/test).
    end_idx : int or None
        Índice final del subconjunto temporal. None = hasta el final.
    phi : float
        Coeficiente de penalización por drawdown en la función de recompensa.
    gamma : float
        Coeficiente de penalización por turnover en la función de recompensa.
    """

    def __init__(self, features_path, prices_path, initial_balance=10000,
                 commission=0.001, start_idx=0, end_idx=None, phi=0.02,
                 gamma=0.01):
        super().__init__()

        # Cargar datos: primero reemplazar ±inf (dropna NO los elimina) y luego NaNs
        df_f = (pd.read_csv(features_path, index_col=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna())

        # Sincronizar precios con las features que han quedado vivas tras el dropna
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        # Establecer el límite final del subconjunto (train o test)
        if end_idx is None:
            end_idx = len(df_f)

        # Subconjunto temporal: permite crear entornos de train (0..split) y test (split..N)
        # fillna(0) como segunda línea de defensa ante NaNs residuales introducidos en el slice
        self.df_features = (df_f.iloc[start_idx:end_idx]
                               .reset_index(drop=True)
                               .fillna(0.0)
                               .astype(np.float32))
        # Para precios: forward-fill para propagar el último precio válido,
        # luego backward-fill por si hay NaN al inicio, y finalmente 1.0 como último recurso.
        # Nunca usar 0 directamente porque genera retornos infinitos (división por cero).
        self.df_prices = (df_p.iloc[start_idx:end_idx]
                             .reset_index(drop=True)
                             .replace([np.inf, -np.inf], np.nan)
                             .ffill()
                             .bfill()
                             .fillna(1.0)
                             .astype(np.float32))

        # Verificación de integridad: avisar si quedan NaN/inf tras el saneamiento
        n_nan = self.df_features.isnull().values.sum()
        n_inf = np.isinf(self.df_features.values).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"[AVISO] features con NaN={n_nan}, inf={n_inf} — se sustituirán por 0")
            self.df_features = self.df_features.fillna(0.0)
            self.df_features.replace([np.inf, -np.inf], 0.0, inplace=True)

        print(f"Entorno creado con {len(self.df_features)} pasos "
              f"(del índice {start_idx} al {end_idx}).")

        self.n_assets        = len(self.df_prices.columns)
        self.initial_balance = initial_balance
        self.commission      = commission

        # Coeficientes de la función de recompensa compuesta:
        #   R_t = r_p(t) - phi * MDD(t) - gamma * Turnover(t)
        #
        # CALIBRACIÓN: phi debe ser comparable en escala a log_return diario (~0.001).
        # phi=0.5 con MDD=20% -> penalty=0.1 >> log_return=0.001 -> agente aprende a no hacer nada.
        # phi=0.02: un MDD del 20% penaliza 0.004, del mismo orden que un día positivo.
        self.phi   = phi    # penalización drawdown: 0.02
        self.gamma = gamma  # penalización turnover: 0.01 (10x mayor que antes)
                            # Con gamma=0.001, rotar toda la cartera costaba 0.002 en penalty
                            # vs ~0.001 de log_return -> salía "barato" operar constantemente.
                            # Con gamma=0.01, el coste de rotación completa es 0.02, lo que
                            # equivale a ~20 días de retorno positivo -> el agente aprende a mantener.

        # Buffer para Sharpe rolling: ventana de 20 días para estabilizar la señal
        self._ret_buffer    = []
        self._sharpe_window = 20

        # Observación aumentada: features de mercado + estado de cartera del agente.
        # Sin el estado de cartera, el agente no sabe qué tiene ni cuánto ha ganado/perdido,
        # lo que hace imposible aprender estrategias de gestión de riesgo.
        # Estado añadido: [pesos actuales (n_assets), retorno acumulado normalizado]
        n_market_features = self.df_features.shape[1]
        n_portfolio_state = self.n_assets + 1   # pesos + retorno_acumulado
        self.n_market_features = n_market_features

        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_market_features + n_portfolio_state,),
            dtype=np.float32
        )

    def _build_obs(self) -> np.ndarray:
        """
        Construye la observación aumentada del entorno.

        Concatena las features de mercado del paso actual con el estado
        interno de la cartera (pesos actuales + retorno acumulado normalizado).
        El estado de cartera permite al agente aprender comportamientos
        condicionados a su posición actual.

        Returns
        -------
        np.ndarray
            Vector de observación de dimensión (n_market_features + n_assets + 1,).
        """
        market = np.nan_to_num(
            self.df_features.iloc[self.current_step].values,
            nan=0.0, posinf=0.0, neginf=0.0
        )
        # Retorno acumulado normalizado: 0 = sin cambio, +1 = dobló, -1 = perdió todo
        cumulative_return = np.clip(
            (self.portfolio_value - self.initial_balance) / (self.initial_balance + 1e-8),
            -1.0, 5.0
        )
        portfolio_state = np.append(
            self.weights.astype(np.float32), np.float32(cumulative_return)
        )
        return np.concatenate([market, portfolio_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial para un nuevo episodio.

        Parameters
        ----------
        seed : int or None
            Semilla para el generador de números aleatorios.
        options : dict or None
            Opciones adicionales (no utilizadas actualmente).

        Returns
        -------
        tuple[np.ndarray, dict]
            Observación inicial y diccionario de información vacío.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._ret_buffer = []  # buffer para Sharpe rolling

        # Inicializar la marca de agua máxima aquí, no en step(),
        # para garantizar que se resetea correctamente en cada episodio
        self.max_portfolio_value = self.initial_balance

        return self._build_obs(), {}

    def step(self, action):
        """
        Ejecuta un paso temporal: aplica la acción del agente, calcula
        la evolución de la cartera y devuelve la recompensa compuesta.

        Parameters
        ----------
        action : np.ndarray
            Vector de pesos deseados para cada activo (se normaliza internamente).

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict]
            (observación, recompensa, terminado, truncado, info).
            info contiene: value, drawdown, weights, turnover.
        """
        # 1. Precios y estado previo
        prices_today = self.df_prices.iloc[self.current_step].values
        # Referencia de valor para calcular el retorno del paso
        baseline_value = max(self.portfolio_value, 1e-6)

        # 2. Normalizar pesos (sumatoria = 1)
        # Recortar al espacio declarado [0,1] — SB3 usa Gaussian que puede salirse
        action_clipped = np.clip(action, 0.0, 1.0)
        total_weight = np.sum(action_clipped)
        if total_weight < 1e-3:
            new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        else:
            new_weights = action_clipped / total_weight

        # 3. Gestión de costes: coste de mover dinero, lo que evita rebalanceos erráticos
        diff_weights = np.abs(new_weights - self.weights)
        cost = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost

        # 4. Progresión temporal
        if self.current_step >= len(self.df_features) - 1:
            done = True
            return (self._build_obs(), 0.0, done, False,
                    {"value": self.portfolio_value, "drawdown": 0})

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1

        # 5. Evolución del mercado (impacto de los precios de mañana)
        prices_tomorrow = self.df_prices.iloc[self.current_step].values

        # np.fmax ignora NaN (a diferencia de np.maximum que lo propaga)
        # Necesario porque IBIT no existía antes de 2024 y puede quedar NaN residual
        prices_today_safe    = np.fmax(prices_today,    1e-6)
        prices_tomorrow_safe = np.fmax(prices_tomorrow, 1e-6)

        # Recortar retornos diarios al rango [0.5, 2.0] (-50% / +100%)
        # para evitar que un dato erróneo en el CSV destruya el portfolio_value
        asset_returns = np.clip(prices_tomorrow_safe / prices_today_safe, 0.5, 2.0)

        new_value = np.sum((self.portfolio_value * new_weights) * asset_returns)
        # Recortar a un rango finito para evitar overflow -> NaN en operaciones posteriores
        self.portfolio_value = float(np.clip(new_value, 1e-6, 1e9))

        self.weights = new_weights

        # --- REWARD SHAPING: EL CORAZÓN ---
        # A. Retorno logarítmico del paso
        log_return = float(np.log(self.portfolio_value / baseline_value + 1e-8))

        # B. Sharpe rolling como señal principal de calidad.
        # En lugar de optimizar retorno bruto (que incentiva apalancar riesgo),
        # el agente optimiza retorno ajustado por riesgo reciente.
        # Ventana de 20 días aprox 1 mes de trading: suficiente para capturar volatilidad
        # local sin ignorar el régimen actual del mercado.
        self._ret_buffer.append(log_return)
        if len(self._ret_buffer) > self._sharpe_window:
            self._ret_buffer.pop(0)

        if len(self._ret_buffer) >= 5:
            rets = np.array(self._ret_buffer)
            # Sharpe anualizado: media / std * sqrt(252). Se normaliza a [-1,1] dividiendo por 3.
            rolling_sharpe = float(
                np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
            )
            sharpe_norm = float(np.clip(rolling_sharpe / 3.0, -1.0, 1.0))
        else:
            # Warmup: usar log_return directamente hasta tener suficiente historia
            sharpe_norm = float(np.clip(log_return * 100, -1.0, 1.0))

        # C. Maximum Drawdown actual
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = float(np.clip(
            (self.max_portfolio_value - self.portfolio_value)
            / (self.max_portfolio_value + 1e-8),
            0.0, 1.0
        ))

        # D. Función de recompensa compuesta:
        #   R_t = Sharpe_rolling - phi * MDD(t) - gamma * Turnover(t)
        #
        # Sharpe_rolling: optimiza retorno ajustado por riesgo, no retorno bruto.
        # phi=0.02:  MDD del 20% penaliza 0.004, comparable al Sharpe diario típico.
        # gamma=0.01: turnover completo penaliza 0.02 ~ 20 días de retorno positivo.
        #             El agente aprende que solo vale la pena operar si la mejora
        #             de Sharpe compensa el coste de rotación.
        turnover         = float(np.sum(diff_weights))
        risk_penalty     = self.phi   * current_drawdown
        turnover_penalty = self.gamma * turnover

        reward = float(np.clip(
            sharpe_norm - risk_penalty - turnover_penalty, -1.0, 1.0
        ))

        # E. Condición de quiebra: pérdida del 90% del capital
        if self.portfolio_value < (self.initial_balance * 0.1):
            done = True
            reward = -1.0

        info = {
            "value":    self.portfolio_value,
            "drawdown": current_drawdown,
            "weights":  self.weights,
            "turnover": turnover,
        }

        return self._build_obs(), float(reward), done, False, info


# ---------------------------------------------------------------------------
# Compatibilidad hacia atrás: alias del atributo renombrado
# ---------------------------------------------------------------------------
# df_precios fue renombrado a df_prices. El código externo que acceda al
# atributo directamente seguirá funcionando gracias al property.
_original_init = PortfolioEnv.__init__

def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # Alias para compatibilidad hacia atrás
    self.df_precios = self.df_prices

PortfolioEnv.__init__ = _patched_init
