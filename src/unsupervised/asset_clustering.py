"""
Clustering dinámico de activos basado en comportamiento reciente.

Fundamentación:
  La clasificación estática de activos por sector (utilities, salud, cripto...)
  no captura cómo se comportan realmente en distintos regímenes de mercado.
  En una crisis, activos "defensivos" pueden correlacionarse con los de riesgo.

  El clustering dinámico agrupa activos por su COMPORTAMIENTO RECIENTE:
    - Rolling window de retornos → vector de características por activo
    - K-Means agrupa activos que se mueven de forma similar
    - Los clusters cambian con el tiempo: IBIT puede estar en el cluster
      de "activos de riesgo" un trimestre y en "descorrelados" el siguiente

  Esto permite al agente especulativo concentrar posiciones en el cluster
  con mejor momentum y evitar clusters en drawdown.

Uso:
  clusterer = AssetClusterer(n_clusters=3, window=60)
  clusterer.fit(prices_df)
  labels = clusterer.predict(prices_df)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AssetClusterer:
    """
    Agrupa activos por similitud de comportamiento en una ventana rolling.

    Para cada ventana temporal, cada activo se describe por un vector de features:
      [retorno_acumulado, volatilidad, skewness, max_drawdown]
    K-Means agrupa activos con vectores similares.

    Parameters
    ----------
    n_clusters : int
        Número de grupos de activos (2-4 es razonable con 8 activos).
    window : int
        Días de la ventana rolling para calcular features por activo.
    random_state : int
        Semilla para reproducibilidad.
    """

    def __init__(self, n_clusters: int = 3, window: int = 60,
                 random_state: int = 42):
        self.n_clusters = n_clusters
        self.window  = window
        self.random_state = random_state
        self.scaler  = StandardScaler()

    def _features_per_asset(self, returns: pd.DataFrame,
                            end_idx: int) -> np.ndarray:
        """
        Calcula el vector de características de cada activo en la ventana
        [end_idx - window, end_idx].

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de retornos logarítmicos con una columna por activo.
        end_idx : int
            Índice del último día de la ventana (exclusivo).

        Returns
        -------
        np.ndarray
            Array de forma (n_activos, 4) con: retorno acumulado, volatilidad,
            skewness y max drawdown por activo.
        """
        start = max(0, end_idx - self.window)
        window_returns = returns.iloc[start:end_idx]

        features = []
        for col in window_returns.columns:
            r = window_returns[col].values
            cum_return = np.nansum(r)
            vol        = np.nanstd(r)
            skew       = float(pd.Series(r).skew()) if len(r) > 2 else 0.0

            # Max drawdown de la ventana
            cum = np.nancumsum(r)
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            mdd = float(np.nanmin(dd)) if len(dd) > 0 else 0.0

            features.append([cum_return, vol, skew, mdd])

        return np.array(features)

    def cluster_at_date(self, returns: pd.DataFrame,
                        idx: int) -> np.ndarray:
        """
        Asigna cada activo a un cluster para una fecha concreta.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de retornos logarítmicos con una columna por activo.
        idx : int
            Índice de la fecha para la cual calcular los clusters.

        Returns
        -------
        np.ndarray
            Array de forma (n_activos,) con etiquetas de cluster [0, 1, ..., n_clusters-1],
            ordenadas para que cluster 0 = peor momentum, cluster N-1 = mejor momentum.
        """
        X = self._features_per_asset(returns, idx)
        X_scaled = self.scaler.fit_transform(X)

        n_assets = len(returns.columns)
        k = min(self.n_clusters, n_assets)
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels_raw = km.fit_predict(X_scaled)

        # Ordenar clusters por retorno medio: 0=peor, N-1=mejor
        return_per_cluster = {}
        for c in range(k):
            mask = labels_raw == c
            return_per_cluster[c] = X[mask, 0].mean()  # columna 0 = retorno acumulado

        order = sorted(return_per_cluster.keys(), key=lambda c: return_per_cluster[c])
        mapping = {old: new for new, old in enumerate(order)}
        return np.array([mapping[l] for l in labels_raw])

    def rolling_clustering(self, prices: pd.DataFrame,
                           frequency: int = 20) -> pd.DataFrame:
        """
        Ejecuta el clustering cada ``frequency`` días sobre todo el histórico.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame de precios de cierre con una columna por activo.
        frequency : int
            Cada cuántos días se recalcula el clustering (por defecto 20, ~1 mes).

        Returns
        -------
        pd.DataFrame
            DataFrame de forma (n_fechas, n_activos) con la etiqueta de cluster,
            forward-filled entre evaluaciones.
        """
        # Calcular retornos logarítmicos
        returns = np.log(prices / prices.shift(1)).dropna()

        results = pd.DataFrame(index=returns.index, columns=prices.columns)

        for i in range(self.window, len(returns), frequency):
            labels = self.cluster_at_date(returns, i)
            date   = returns.index[i]
            for j, col in enumerate(prices.columns):
                results.loc[date, col] = labels[j]

        # Forward fill para los días entre evaluaciones
        results = results.ffill().bfill().astype(float)

        return results

    # todo: ojo retro compatibilidad
    # --- Alias de retrocompatibilidad ---
    cluster_en_fecha = cluster_at_date
    clustering_rolling = rolling_clustering
