"""
Detección de regímenes de mercado con Gaussian Mixture Models (GMM).

Fundamentación:
  Los mercados financieros alternan entre regímenes latentes (no observables directamente):
    - Bull market: retornos positivos, baja volatilidad
    - Bear market: retornos negativos, alta volatilidad
    - Crisis/pánico: retornos muy negativos, volatilidad extrema, correlaciones convergen a 1
    - Consolidación: retornos cercanos a cero, volatilidad moderada

  Un GMM Gaussiano modela esto como una mezcla de distribuciones: cada componente
  representa un régimen con su propia media y covarianza. El modelo aprende sin
  supervisión qué distribución genera las observaciones de cada día.

  GMM vs HMM:
    - HMM añade una matriz de transición entre estados (modelo temporal).
    - GMM trata cada observación como independiente (modelo estático).
    - Para detección de régimen diario en carteras, la diferencia práctica es mínima:
      el suavizado temporal se puede añadir post-hoc con una media móvil sobre
      las probabilidades posteriores.
    - Ventaja de GMM: está en scikit-learn (sin compilación C++), es más rápido
      de ajustar y más estable numéricamente.

  Referencia:
    - Hamilton (1989), "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle", Econometrica.
    - Ang & Bekaert (2002), "Regime Switches in Interest Rates", Journal of
      Business & Economic Statistics.

Uso:
  detector = RegimeDetector(n_regimes=3)
  detector.fit(features_df)
  regimes  = detector.predict(features_df)
  probs    = detector.predict_proba(features_df)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class RegimeDetector:
    """
    Detector de regímenes de mercado basado en GMM Gaussiano.

    Pipeline interno:
      1. Seleccionar features observables (retornos, volatilidad, correlaciones)
      2. Estandarizar con Z-Score (fit solo sobre train)
      3. Reducir dimensionalidad con PCA (opcional, si hay muchas features)
      4. Ajustar GMM Gaussiano con N componentes
      5. Asignar cada día al régimen más probable

    Parameters
    ----------
    n_regimes : int
        Número de componentes (regímenes). 3 es estándar en la literatura.
    n_components_pca : int o None
        Componentes PCA antes del GMM. None = sin PCA.
    smoothing : int
        Ventana de media móvil sobre probabilidades para dar continuidad
        temporal (simula parcialmente el efecto de la matriz de transición
        de un HMM). 5 = suavizar con 5 días (1 semana hábil).
    random_state : int
        Semilla para reproducibilidad.
    """

    def __init__(self, n_regimes: int = 3, n_components_pca: int = None,
                 smoothing: int = 5, random_state: int = 42):
        self.n_regimes       = n_regimes
        self.n_components_pca = n_components_pca
        self.smoothing       = smoothing
        self.random_state    = random_state

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components_pca) if n_components_pca else None
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            n_init=5,
            max_iter=200,
            random_state=random_state,
        )

        self._fitted = False
        self._col_names = None
        self._order  = None
        self._mapping= None

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selecciona las columnas más informativas para la detección de régimen.

        Prioriza: retornos, volatilidades, correlaciones dinámicas.
        Excluye: features binarias (calendario, régimen ya calculado) y técnicas
        que son derivadas de las anteriores (RSI, MACD, etc.).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con todas las features disponibles.

        Returns
        -------
        pd.DataFrame
            Subconjunto de columnas relevantes para la detección de régimen.
        """
        cols = []
        for c in df.columns:
            if any(k in c for k in ['_retornos', '_vol_', '_momentum_', 'corr_',
                                     '_skew_', '_kurt_', '_beta_']):
                cols.append(c)

        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self._col_names = cols
        return df[cols]

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Ajusta el GMM sobre datos de entrenamiento.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con features del período de entrenamiento.
            IMPORTANTE: solo datos de train para evitar lookahead bias.

        Returns
        -------
        RegimeDetector
            Referencia a sí mismo para permitir encadenamiento de métodos.
        """
        X = self._select_features(df).dropna()

        X_scaled = self.scaler.fit_transform(X)

        if self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)

        self.gmm.fit(X_scaled)
        self._fitted = True

        # Ordenar regímenes por volatilidad media (0=calma, N-1=crisis)
        self._sort_regimes(X_scaled)

        print(f"  GMM ajustado: {self.n_regimes} regímenes, "
              f"{len(self._col_names)} features, BIC={self.gmm.bic(X_scaled):.0f}")

        return self

    def _sort_regimes(self, X: np.ndarray):
        """
        Reordena los componentes del GMM para que el régimen 0 sea el más tranquilo
        y el último sea el más volátil. Esto hace los resultados interpretables
        sin depender del orden aleatorio de inicialización del GMM.

        Parameters
        ----------
        X : np.ndarray
            Datos escalados (y opcionalmente reducidos con PCA) usados para el ajuste.
        """
        states = self.gmm.predict(X)
        vol_per_state = []
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() > 0:
                vol_per_state.append(X[mask].std())
            else:
                vol_per_state.append(0.0)

        self._order   = np.argsort(vol_per_state)
        self._mapping = {int(old): int(new) for new, old in enumerate(self._order)}

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma features para predicción (escala y aplica PCA si corresponde).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con las mismas columnas usadas en el ajuste.

        Returns
        -------
        np.ndarray
            Datos transformados listos para el GMM.
        """
        X = df[self._col_names].dropna()
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice el régimen de cada día.

        Aplica suavizado temporal sobre las probabilidades posteriores para
        dar continuidad: evita que el régimen cambie de un día para otro por ruido.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con features del período a clasificar.

        Returns
        -------
        np.ndarray
            Array de enteros [0, 1, ..., n_regimes-1] donde 0=calma, N-1=crisis.
        """
        assert self._fitted, "Ejecuta .fit() primero"

        probs = self.predict_proba(df)

        # Suavizar probabilidades con media móvil (simula transiciones del HMM)
        if self.smoothing > 1:
            smoothed_probs = pd.DataFrame(probs).rolling(
                self.smoothing, min_periods=1
            ).mean().values
        else:
            smoothed_probs = probs

        return np.argmax(smoothed_probs, axis=1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Probabilidad posterior de cada régimen para cada día.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con features del período a clasificar.

        Returns
        -------
        np.ndarray
            Array de forma (n_dias, n_regimes) con las columnas reordenadas
            según el mapeo calma→crisis.
        """
        assert self._fitted, "Ejecuta .fit() primero"

        X_scaled  = self._prepare(df)
        probs_raw = self.gmm.predict_proba(X_scaled)
        return probs_raw[:, self._order]

    def describe_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera un resumen estadístico de cada régimen detectado.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con features para clasificar y analizar por régimen.

        Returns
        -------
        pd.DataFrame
            DataFrame con media y std de retornos por régimen,
            útil para validar que los regímenes tienen sentido financiero.
        """
        X = df[self._col_names].dropna()
        regimes = self.predict(df)

        # Buscar columna de retornos del mercado (IVV como proxy)
        returns_col = None
        for c in self._col_names:
            if 'IVV_retornos' in c:
                returns_col = c
                break
        if returns_col is None:
            returns_col = self._col_names[0]

        summary = []
        names = {0: 'Calma', 1: 'Transición', 2: 'Crisis'}
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() == 0:
                continue
            vals = X[returns_col].values[mask]
            summary.append({
                'Régimen': names.get(r, f'Estado {r}'),
                'Días': int(mask.sum()),
                'Pct del total': f"{mask.mean():.1%}",
                'Retorno medio diario': f"{vals.mean():.4f}",
                'Volatilidad diaria': f"{vals.std():.4f}",
            })

        return pd.DataFrame(summary)

    # --- Alias de retrocompatibilidad ---
    descripcion_regimenes = describe_regimes


# Alias de retrocompatibilidad
RegimenDetector = RegimeDetector
