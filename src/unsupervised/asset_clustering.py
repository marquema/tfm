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
  clusterer = AssetClusterer(n_clusters=3, ventana=60)
  clusterer.fit(precios_df)
  labels = clusterer.predict(precios_df)
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

    Parámetros
    ----------
    n_clusters : número de grupos de activos (2-4 es razonable con 8 activos)
    ventana    : días de la ventana rolling para calcular features por activo
    """

    def __init__(self, n_clusters: int = 3, ventana: int = 60,
                 random_state: int = 42):
        self.n_clusters   = n_clusters
        self.ventana      = ventana
        self.random_state = random_state
        self.scaler       = StandardScaler()

    def _features_por_activo(self, retornos: pd.DataFrame,
                              idx_fin: int) -> np.ndarray:
        """
        Calcula el vector de características de cada activo en la ventana
        [idx_fin - ventana, idx_fin].

        Retorna array (n_activos, 4) con: retorno acum, vol, skew, mdd.
        """
        inicio = max(0, idx_fin - self.ventana)
        ventana_rets = retornos.iloc[inicio:idx_fin]

        features = []
        for col in ventana_rets.columns:
            r = ventana_rets[col].values
            ret_acum = np.nansum(r)
            vol      = np.nanstd(r)
            skew     = float(pd.Series(r).skew()) if len(r) > 2 else 0.0

            # Max drawdown de la ventana
            cum = np.nancumsum(r)
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            mdd = float(np.nanmin(dd)) if len(dd) > 0 else 0.0

            features.append([ret_acum, vol, skew, mdd])

        return np.array(features)

    def cluster_en_fecha(self, retornos: pd.DataFrame,
                          idx: int) -> np.ndarray:
        """
        Asigna cada activo a un cluster para una fecha concreta.

        Retorna array (n_activos,) con etiquetas de cluster [0, 1, ..., n_clusters-1],
        ordenadas para que cluster 0 = peor momentum, cluster N-1 = mejor momentum.
        """
        X = self._features_por_activo(retornos, idx)
        X_scaled = self.scaler.fit_transform(X)

        n_activos = len(retornos.columns)
        k = min(self.n_clusters, n_activos)
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels_raw = km.fit_predict(X_scaled)

        # Ordenar clusters por retorno medio: 0=peor, N-1=mejor
        ret_por_cluster = {}
        for c in range(k):
            mask = labels_raw == c
            ret_por_cluster[c] = X[mask, 0].mean()  # columna 0 = retorno acumulado

        orden = sorted(ret_por_cluster.keys(), key=lambda c: ret_por_cluster[c])
        mapa  = {viejo: nuevo for nuevo, viejo in enumerate(orden)}
        return np.array([mapa[l] for l in labels_raw])

    def clustering_rolling(self, precios: pd.DataFrame,
                            frecuencia: int = 20) -> pd.DataFrame:
        """
        Ejecuta el clustering cada `frecuencia` días sobre todo el histórico.

        Retorna DataFrame (n_fechas, n_activos) con la etiqueta de cluster,
        forward-filled entre evaluaciones.
        """
        # Calcular retornos logarítmicos
        retornos = np.log(precios / precios.shift(1)).dropna()

        resultados = pd.DataFrame(index=retornos.index, columns=precios.columns)

        for i in range(self.ventana, len(retornos), frecuencia):
            labels = self.cluster_en_fecha(retornos, i)
            fecha  = retornos.index[i]
            for j, col in enumerate(precios.columns):
                resultados.loc[fecha, col] = labels[j]

        # Forward fill para los días entre evaluaciones
        resultados = resultados.ffill().bfill().astype(float)

        return resultados
