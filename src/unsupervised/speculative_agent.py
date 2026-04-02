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
  agente = SpeculativeAgent(n_regimenes=3, n_clusters=3)
  agente.fit(features_train, precios_train)
  pesos  = agente.generar_pesos(features_test, precios_test)
"""

import numpy as np
import pandas as pd

from src.unsupervised.regime_hmm import RegimenDetector
from src.unsupervised.asset_clustering import AssetClusterer


# Matrices de asignación por régimen × cluster.
# Cada fila es un régimen [calma, transición, crisis].
# Cada columna es un cluster [peor momentum, medio, mejor momentum].
# Los valores representan la fracción del capital asignada a cada cluster.
ASIGNACION_DEFAULT = {
    0: np.array([0.10, 0.20, 0.70]),   # Calma:      agresivo en momentum alto
    1: np.array([0.25, 0.50, 0.25]),   # Transición: diversificado
    2: np.array([0.60, 0.30, 0.10]),   # Crisis:     concentrar en defensivos
}


class SpeculativeAgent:
    """
    Agente especulativo que combina régimen HMM + clustering K-Means.

    Parámetros
    ----------
    n_regimenes     : estados ocultos del HMM (3 = calma/transición/crisis)
    n_clusters      : grupos de activos por comportamiento (3 = defensivo/neutro/agresivo)
    ventana_cluster : días de la ventana rolling para el clustering de activos
    asignacion      : dict {régimen: array de pesos por cluster}. Si None, usa default.
    """

    def __init__(self, n_regimenes: int = 3, n_clusters: int = 3,
                 ventana_cluster: int = 60, asignacion: dict = None):
        self.detector  = RegimenDetector(n_regimenes=n_regimenes)
        self.clusterer = AssetClusterer(n_clusters=n_clusters, ventana=ventana_cluster)
        self.n_clusters = n_clusters
        self.asignacion = asignacion or ASIGNACION_DEFAULT
        self._fitted    = False

    def fit(self, features_train: pd.DataFrame,
            precios_train: pd.DataFrame) -> 'SpeculativeAgent':
        """
        Ajusta el HMM y el clustering sobre datos de entrenamiento.

        Parámetros
        ----------
        features_train : features normalizadas del período de train
        precios_train  : precios de cierre del período de train
        """
        print("  [1/2] Ajustando HMM para detección de regímenes...")
        self.detector.fit(features_train)

        # Mostrar resumen de regímenes detectados
        desc = self.detector.descripcion_regimenes(features_train)
        print(desc.to_string(index=False))

        print("  [2/2] Clustering de activos listo (se ejecuta rolling en predict).")
        self._fitted = True
        return self

    def generar_pesos(self, features: pd.DataFrame,
                       precios: pd.DataFrame) -> pd.DataFrame:
        """
        Genera los pesos de cartera día a día para el período dado.

        Para cada día:
          1. Detecta el régimen actual (HMM)
          2. Obtiene el cluster de cada activo (K-Means rolling)
          3. Asigna peso según la tabla régimen × cluster
          4. Normaliza para que la suma = 1

        Retorna DataFrame (n_dias, n_activos) con pesos [0,1].
        """
        assert self._fitted, "Ejecuta .fit() primero"

        # Detectar regímenes para todo el período
        regimenes = self.detector.predict(features)

        # Clustering rolling de activos
        clusters = self.clusterer.clustering_rolling(precios)

        # Alinear índices
        fechas_comunes = features.dropna().index[:len(regimenes)]
        n_activos = len(precios.columns)

        pesos = pd.DataFrame(
            index=fechas_comunes,
            columns=precios.columns,
            dtype=np.float64
        )

        for i, fecha in enumerate(fechas_comunes):
            regimen = regimenes[i]

            # Pesos base por cluster para este régimen
            pesos_cluster = self.asignacion.get(
                regimen, np.ones(self.n_clusters) / self.n_clusters
            )

            # Asignar a cada activo según su cluster
            w = np.zeros(n_activos)
            if fecha in clusters.index:
                labels = clusters.loc[fecha].values.astype(int)
            else:
                labels = np.zeros(n_activos, dtype=int)

            for j in range(n_activos):
                cluster_id = min(labels[j], len(pesos_cluster) - 1)
                # Peso base del cluster dividido equitativamente entre activos del mismo cluster
                n_en_cluster = max(1, (labels == labels[j]).sum())
                w[j] = pesos_cluster[cluster_id] / n_en_cluster

            # Normalizar
            w_sum = w.sum()
            if w_sum > 1e-8:
                w = w / w_sum

            pesos.iloc[i] = w

        return pesos

    def backtest(self, features: pd.DataFrame, precios: pd.DataFrame,
                  initial_balance: float = 10000,
                  commission: float = 0.001) -> pd.Series:
        """
        Ejecuta el backtest completo del agente especulativo.

        Retorna pd.Series con la evolución del valor de la cartera.
        """
        pesos_df = self.generar_pesos(features, precios)

        # Retornos diarios de cada activo
        retornos = precios.pct_change().fillna(0)
        retornos = retornos.loc[pesos_df.index]

        valor = initial_balance
        equity = [valor]

        pesos_anteriores = np.zeros(len(precios.columns))

        for i in range(len(pesos_df)):
            w = pesos_df.iloc[i].values.astype(float)

            # Costes de transacción
            turnover = np.abs(w - pesos_anteriores).sum()
            coste = turnover * valor * commission
            valor -= coste

            # Retorno del día
            if i < len(retornos):
                ret_dia = (retornos.iloc[i].values * w).sum()
                valor *= (1 + ret_dia)

            valor = max(valor, 1e-6)
            equity.append(valor)
            pesos_anteriores = w

        return pd.Series(equity, name='Especulativo_HMM')
