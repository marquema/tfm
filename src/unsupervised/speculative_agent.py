"""
Agente especulativo basado en aprendizaje no supervisado.

Estrategia:
  Combina la detección de régimen (HMM) con el clustering de activos (K-Means)
  para generar pesos de cartera especulativos sin entrenamiento supervisado ni RL.

  Lógica de asignación:
    1. Detectar el régimen actual del mercado (calma, transición, crisis)
    2. Identificar a qué cluster pertenece cada activo (momentum alto/medio/bajo)
    3. Asignar pesos según la tabla régimen × cluster:

       Régimen CALMA:      concentrar en cluster de mejor momentum (especulativo)
       Régimen TRANSICIÓN:  diversificar entre clusters (prudente)
       Régimen CRISIS:      concentrar en cluster defensivo / reducir exposición

  Esta estrategia es puramente reactiva a la estructura detectada en los datos,
  sin optimización de reward ni gradientes — complementa al PPO mostrando qué
  rendimiento se obtiene solo con detección de patrones.

Uso:
  agent = SpeculativeAgent(n_regimes=3, n_clusters=3)
  agent.fit(features_train, prices_train)
  weights = agent.generate_weights(features_test, prices_test)
"""

import numpy as np
import pandas as pd

from src.unsupervised.regime_hmm import RegimeDetector
from src.unsupervised.asset_clustering import AssetClusterer


# Matrices de asignación por régimen × cluster.
# Cada fila es un régimen [calma, transición, crisis].
# Cada columna es un cluster [peor momentum, medio, mejor momentum].
# Los valores representan la fracción del capital asignada a cada cluster.
DEFAULT_ALLOCATION = {
    0: np.array([0.10, 0.20, 0.70]),   # Calma:      agresivo en momentum alto
    1: np.array([0.25, 0.50, 0.25]),   # Transición: diversificado
    2: np.array([0.60, 0.30, 0.10]),   # Crisis:     concentrar en defensivos
}


class SpeculativeAgent:
    """
    Agente especulativo que combina régimen HMM + clustering K-Means.

    Parameters
    ----------
    n_regimes : int
        Estados ocultos del HMM (3 = calma/transición/crisis).
    n_clusters : int
        Grupos de activos por comportamiento (3 = defensivo/neutro/agresivo).
    cluster_window : int
        Días de la ventana rolling para el clustering de activos.
    allocation : dict o None
        Diccionario {régimen: array de pesos por cluster}. Si None, usa DEFAULT_ALLOCATION.
    """

    def __init__(self, n_regimes: int = 3, n_clusters: int = 3,
                 cluster_window: int = 60, allocation: dict = None):
        self.detector   = RegimeDetector(n_regimes=n_regimes)
        self.clusterer  = AssetClusterer(n_clusters=n_clusters, window=cluster_window)
        self.n_clusters = n_clusters
        self.allocation = allocation or DEFAULT_ALLOCATION
        self._fitted    = False

    def fit(self, features_train: pd.DataFrame,
            prices_train: pd.DataFrame) -> 'SpeculativeAgent':
        """
        Ajusta el HMM y el clustering sobre datos de entrenamiento.

        Parameters
        ----------
        features_train : pd.DataFrame
            Features normalizadas del período de train.
        prices_train : pd.DataFrame
            Precios de cierre del período de train.

        Returns
        -------
        SpeculativeAgent
            Referencia a sí mismo para permitir encadenamiento de métodos.
        """
        print("  [1/2] Ajustando HMM para detección de regímenes...")
        self.detector.fit(features_train)

        # Mostrar resumen de regímenes detectados
        desc = self.detector.describe_regimes(features_train)
        print(desc.to_string(index=False))

        print("  [2/2] Clustering de activos listo (se ejecuta rolling en predict).")
        self._fitted = True
        return self

    def generate_weights(self, features: pd.DataFrame,
                         prices: pd.DataFrame) -> pd.DataFrame:
        """
        Genera los pesos de cartera día a día para el período dado.

        Para cada día:
          1. Detecta el régimen actual (HMM)
          2. Obtiene el cluster de cada activo (K-Means rolling)
          3. Asigna peso según la tabla régimen × cluster
          4. Normaliza para que la suma = 1

        Parameters
        ----------
        features : pd.DataFrame
            Features del período a evaluar.
        prices : pd.DataFrame
            Precios de cierre del período a evaluar.

        Returns
        -------
        pd.DataFrame
            DataFrame de forma (n_dias, n_activos) con pesos en [0, 1].
        """
        assert self._fitted, "Ejecuta .fit() primero"

        # Detectar regímenes para todo el período
        regimes = self.detector.predict(features)

        # Clustering rolling de activos
        clusters = self.clusterer.rolling_clustering(prices)

        # Alinear índices
        common_dates = features.dropna().index[:len(regimes)]
        n_assets = len(prices.columns)

        weights = pd.DataFrame(
            index=common_dates,
            columns=prices.columns,
            dtype=np.float64
        )

        for i, date in enumerate(common_dates):
            regime = regimes[i]

            # Pesos base por cluster para este régimen
            cluster_weights = self.allocation.get(
                regime, np.ones(self.n_clusters) / self.n_clusters
            )

            # Asignar a cada activo según su cluster
            w = np.zeros(n_assets)
            if date in clusters.index:
                labels = clusters.loc[date].values.astype(int)
            else:
                labels = np.zeros(n_assets, dtype=int)

            for j in range(n_assets):
                cluster_id = min(labels[j], len(cluster_weights) - 1)
                # Peso base del cluster dividido equitativamente entre activos del mismo cluster
                n_in_cluster = max(1, (labels == labels[j]).sum())
                w[j] = cluster_weights[cluster_id] / n_in_cluster

            # Normalizar
            w_sum = w.sum()
            if w_sum > 1e-8:
                w = w / w_sum

            weights.iloc[i] = w

        return weights

    def backtest(self, features: pd.DataFrame, prices: pd.DataFrame,
                 initial_balance: float = 10000,
                 commission: float = 0.001) -> pd.Series:
        """
        Ejecuta el backtest completo del agente especulativo.

        Parameters
        ----------
        features : pd.DataFrame
            Features del período de test.
        prices : pd.DataFrame
            Precios de cierre del período de test.
        initial_balance : float
            Capital inicial de la cartera (por defecto 10000).
        commission : float
            Coste de transacción como fracción del turnover (por defecto 0.1%).

        Returns
        -------
        pd.Series
            Serie temporal con la evolución del valor de la cartera.
        """
        weights_df = self.generate_weights(features, prices)

        # Retornos diarios de cada activo
        returns = prices.pct_change().fillna(0)
        returns = returns.loc[weights_df.index]

        balance = initial_balance
        equity = [balance]

        previous_weights = np.zeros(len(prices.columns))

        for i in range(len(weights_df)):
            w = weights_df.iloc[i].values.astype(float)

            # Costes de transacción
            turnover = np.abs(w - previous_weights).sum()
            cost = turnover * balance * commission
            balance -= cost

            # Retorno del día
            if i < len(returns):
                daily_return = (returns.iloc[i].values * w).sum()
                balance *= (1 + daily_return)

            balance = max(balance, 1e-6)
            equity.append(balance)
            previous_weights = w

        return pd.Series(equity, name='Especulativo_HMM')

    # --- Alias de retrocompatibilidad ---
    generar_pesos = generate_weights


# Alias de retrocompatibilidad
ASIGNACION_DEFAULT = DEFAULT_ALLOCATION
