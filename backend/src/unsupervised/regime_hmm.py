"""
Detección de regímenes de mercado con Gaussian Mixture Models (GMM).

Qué es este módulo en una frase:
    Es el "termómetro de mercado" del agente especulativo. Mira las
    features financieras (retornos, volatilidades, correlaciones) y
    decide en qué TIPO de mercado estamos hoy: calma, transición o crisis.
    No predice nada — solo CLASIFICA el momento actual.

Por qué hace falta detectar el régimen:
    Los mercados no son uniformes — alternan entre estados con dinámicas
    radicalmente distintas que el ojo humano detecta de forma intuitiva
    pero que un modelo necesita formalizar:
      - Bull market  : retornos positivos, volatilidad baja, todo crece.
      - Bear market: retornos negativos sostenidos, volatilidad media.
      - Crisis/pánico : retornos muy negativos, volatilidad extrema, las
                         correlaciones entre activos colapsan a 1 (todo
                         cae a la vez, la "diversificación" desaparece).
      - Consolidación: retornos planos, volatilidad moderada, lateralización.

    Saber en qué régimen estamos permite al agente especulativo cambiar
    su estrategia (ver DEFAULT_ALLOCATION en speculative_agent.py): en
    calma, perseguir retorno; en crisis, refugiarse en defensivos.

GMM (Gaussian Mixture Model) — qué hace, en plano:
    Imaginemos 1000 días de datos de mercado, cada uno con varios
    números (retorno del IVV, volatilidad, correlación con BND…). Si los
    pintas en un gráfico de 2 dimensiones, verías "nubes de puntos":
    una nube concentrada en el rango "retornos cero, volatilidad baja"
    (calma), otra dispersa en "retornos negativos, volatilidad alta"
    (crisis), etc.

    Un GMM aprende a identificar esas nubes automáticamente. Asume que
    los datos provienen de una MEZCLA de N distribuciones gaussianas
    (campanas) y averigua dónde está cada una, qué forma tiene y con qué
    probabilidad un día concreto pertenece a cada una.

    Es aprendizaje NO supervisado: nadie le dice "esto es calma, esto es
    crisis". El modelo descubre las nubes solo, basándose en la
    estructura estadística de los datos.

GMM vs HMM (modelo de Markov oculto) — el matiz académico:
    Un HMM "puro" añade una matriz de transición ENTRE estados que
    captura cuán probable es saltar de un régimen a otro entre días
    consecutivos. Es elegante en teoría — modela la persistencia
    temporal explícitamente.

    En la práctica, para detección DIARIA de régimen aplicada a gestión
    de carteras, la diferencia es minima. Lo que necesitamos es saber
    "¿hoy es calma o crisis?", no la probabilidad exacta de transición.
    Por eso aquí usamos GMM (estático, días independientes) y añadimos
    DESPUÉS un suavizado temporal con media móvil (parámetro `smoothing`)
    que cumple el mismo rol práctico que la matriz de transición del HMM:
    evitar que el régimen oscile cada día por ruido.

    Ventajas adicionales de GMM frente a HMM en este TFM:
      - Disponible en scikit-learn .
      - Más rápido de ajustar.
      - Más estable numéricamente (evita el problema de "label switching"
        que afecta a hmmlearn en algunas versiones).

    Nota: el módulo conserva el nombre histórico "regime_hmm" por
    retrocompatibilidad con el código que lo importa, aunque
    internamente use GMM. La clase RegimeDetector encapsula esa
    decisión sin que los usuarios tengan que preocuparse.

Pipeline interno (4 pasos en `fit`):
    1. Selección de features relevantes (filtra retornos/vols/correlaciones,
       descarta indicadores técnicos derivados).
    2. Estandarización con z-score (StandardScaler), entrenado SOLO con
       train para evitar lookahead bias.
    3. PCA opcional para reducir dimensionalidad si hay demasiadas features.
    4. Ajuste del GMM con N componentes y reordenación de regímenes
       por volatilidad creciente (0=calma, N-1=crisis), garantizando
       interpretabilidad consistente.

Documentacion Referencias académicas:
    - Hamilton (1989), "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle", Econometrica.
      Trabajo seminal sobre regime-switching en series financieras.
    - Ang & Bekaert (2002), "Regime Switches in Interest Rates", Journal
      of Business & Economic Statistics. Muestra que los regímenes son
      detectables empíricamente y mejoran la asignación de cartera.

Uso:
    detector = RegimeDetector(n_regimes=3)
    detector.fit(features_df_train)         # aprende los regímenes con train
    regimes  = detector.predict(features_df) # clasifica cada día
    probs    = detector.predict_proba(features_df) # probabilidades blandas
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class RegimeDetector:
    """
    Detector de regímenes de mercado basado en Gaussian Mixture Model.

    Encapsula el pipeline completo (selección de features → estandarización
    → PCA opcional → GMM → reordenación por volatilidad) detrás de una
    interfaz sklearn-friendly: fit / predict / predict_proba.

    Pipeline interno (resumen):
      1. SELECCIONAR features observables relevantes para el régimen
         (retornos, volatilidades, correlaciones; descarta indicadores
         derivados como RSI o MACD que añaden ruido sin información nueva). El método 
         filtra y se queda solo con las que de verdad informan del régimen: retornos, 
         volatilidades, correlaciones, asimetría. Lo demás es ruido para esta tarea 
         concreta.
      2. ESTANDARIZAR con z-score: cada feature pasa a media 0 y desv. 1.
         Necesario porque el GMM compara distancias y una feature en escala
         "0.001" sería ignorada frente a otra en "100".
            El problema sin estandarizar: mido dos cosas en cada coche:

            Consumo de gasolina: 5 a 10 litros/100km (rango: 5).
            Kilometraje: 0 a 200 000 km (rango: 200 000).
            Si pedimos al algoritmo "calcula la distancia entre dos coches", la diferencia de kilómetros (digamos 50 000) 
            eclipsa por completo la diferencia de consumo (digamos 2). El algoritmo solo ve 
            los kilómetros, ignora el consumo aunque sea importante.

            El z-score lo arregla: cada feature se transforma para que su media sea 0 y su 
            desviación estándar sea 1. Después de estandarizar:

            Una diferencia de "1 unidad" significa lo mismo en ambos: "una desviación estándar".
            Fórmula: z = (valor - media) / desviación_estándar.

            Resumen: ponemos a todas las features en el mismo "campo de juego" para que el 
            algoritmo las pondere de forma justa. Sin esto, la feature con números más 
            grandes domina las decisiones.

      3. PCA opcional: si hay muchas features (>10), comprime a unas pocas
         componentes principales para evitar la "maldición de la
         dimensionalidad" — los GMM funcionan peor en espacios muy altos.
         PCA — Análisis de Componentes Principales: Lo que hace PCA: detecta esas 
         correlaciones y construye nuevos ejes "compuestos" que capturan la información 
         esencial sin redundancias. Ejemplo: PCA es como hacer un resumen de un libro. 
         Conservas la esencia (95%) ocupando mucho menos espacio. Los algoritmos trabajan 
         mejor con resúmenes que con libros completos.

      4. AJUSTAR el GMM con N componentes. Cada componente es una
         distribución gaussiana multivariante con su media (centro de la
         "nube") y matriz de covarianza (forma de la nube). Es decir: el GMM aprende a dibujar 
         3 nubes (regímenes) en el espacio de tus datos. Cada nube tiene un centro 
         (la media) y un "halo" (la covarianza) que define hasta dónde se considera que 
         sigue siendo esa nube.
      5. REORDENAR los componentes por volatilidad creciente para que el
         régimen 0 sea siempre el más tranquilo y el N-1 el más volátil
         — lo que hace los resultados interpretables y comparables entre
         ejecuciones.

    Parameters
    ----------
    n_regimes : int
        Número de regímenes a detectar (componentes del GMM). 3 es el
        estándar en la literatura de regime-switching: calma / transición
        / crisis. Más estados (4-5) detectan matices pero requieren más
        datos para converger y son más difíciles de interpretar.
    n_components_pca : int, optional
        Componentes principales tras el PCA. None = sin reducción
        dimensional. Útil cuando se pasan >10 features muy correlacionadas:
        PCA las "resume" a un puñado de ejes ortogonales.
    smoothing : int
        Ventana de media móvil aplicada a las probabilidades posteriores
        del GMM para dar continuidad temporal. 5 días ≈ 1 semana hábil.
        Hace que el régimen no oscile cada día por ruido (un día anómalo
        no cambia la clasificación; hace falta que la anomalía persista).
        Es la forma "barata" de simular el efecto de la matriz de
        transición de un HMM verdadero. Resumen: en lugar de gritar "¡crisis!" cada vez que
        un día parece raro, esperamos a que varios días seguidos lo confirmen. 
        Es exactamente lo que un humano haría: una mala mañana no es una crisis económica.
    random_state : int
        Semilla para reproducibilidad. Importante porque el GMM tiene
        inicialización estocástica (k-means++) y diferentes semillas
        pueden converger a óptimos locales distintos. Resumen: es como decirle al algoritmo
        "usa esta secuencia concreta de números aleatorios" para que mañana podamos 
        reproducir exactamente el mismo experimento. Sin esto, no se puede comparar métricas 
        entre ejecuciones: al azar. El número 42 es el preferido entre 
        programadores. El autoestopista galáctivo.
    """

    def __init__(self, n_regimes: int = 3, n_components_pca: int = None,
                 smoothing: int = 5, random_state: int = 42):
        self.n_regimes        = n_regimes
        self.n_components_pca = n_components_pca
        self.smoothing        = smoothing
        self.random_state     = random_state

        # Componentes del pipeline. Inicializados aquí para poder reutilizarlos
        # en fit y predict (los z-scores y los pesos del PCA aprendidos en fit
        # se reutilizan tal cual en predict para evitar lookahead bias).
        self.scaler = StandardScaler()#zscore
        self.pca = PCA(n_components=n_components_pca) if n_components_pca else None
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            # 'full': cada gaussiana tiene su propia matriz de covarianza
            # completa (sin restricciones). Más flexible pero más parámetros
            # a ajustar — para 3 regímenes y datos abundantes, manejable.
            covariance_type='full',
            # n_init=5: probamos 5 inicializaciones distintas y nos quedamos
            # con la que mejor verosimilitud da. Robustece frente a óptimos
            # locales pobres del algoritmo EM.
            n_init=5,
            max_iter=200,# si no puede converger, se para a las 200 iteracciones.
            random_state=random_state,
        )

        # Flags de estado y mapeos calculados en fit().
        self._fitted= False
        self._col_names = None  # qué columnas se seleccionaron del DataFrame
        self._order = None  # mapeo etiqueta_GMM → etiqueta_ordenada por vol
        self._mapping= None  # diccionario inverso del anterior

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selecciona las columnas relevantes para detectar régimen de mercado.

        El dataset que llega tiene decenas de features: precios, retornos,
        indicadores técnicos (RSI, MACD), métricas de calendario (día de la
        semana, mes), etc. No todas son útiles para clasificar régimen:

          - SÍ informan del régimen: retornos (`_retornos`), volatilidades
            (`_vol_`), momentum, correlaciones entre activos (`corr_`),
            asimetría (`_skew_`), kurtosis (`_kurt_`), beta (`_beta_`).
            Son las que cambian sistemáticamente entre calma y crisis.
          - NO informan (o duplican información): RSI, MACD y otros
            indicadores técnicos son DERIVADAS de los retornos.

        Si el filtro no encuentra ninguna columna con esos prefijos
        (puede pasar con datasets de prueba o con nombres no estándar),
        cae al fallback "todas las numéricas". Es un comportamiento
        defensivo para que el módulo siga funcionando con cualquier
        DataFrame, aunque idealmente las features deberían venir
        nombradas siguiendo la convención del pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con todas las features disponibles.

        Returns
        -------
        pd.DataFrame
            Subconjunto de columnas filtrado por su utilidad para detectar
            régimen. La lista exacta queda guardada en self._col_names para
            que predict() use las mismas columnas (consistency entre fit y predict).
        """
        cols = []
        for c in df.columns:
            if any(k in c for k in ['_retornos', '_vol_', '_momentum_', 'corr_',
                                     '_skew_', '_kurt_', '_beta_']):
                cols.append(c)

        # Fallback: si no hay convención de nombres reconocible, usar todas
        # las columnas numéricas como mejor aproximación posible.
        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()

        self._col_names = cols
        return df[cols]

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Ajusta el GMM sobre datos de entrenamiento — APRENDE los regímenes.

        Es el equivalente al "vamos a leer 3 años de mercado y descubrir
        cuántos tipos de mercado hay y cómo es cada uno". Cuando termina,
        el detector tiene en memoria la "huella estadística" de cada
        régimen y puede clasificar cualquier día futuro contra ellas.

        IMPORTANTE — evitar lookahead bias:
            Solo se debe pasar datos de TRAIN. Si pasas el dataset
            completo (train + test), el modelo ve los regímenes del
            futuro y los aprende — luego al "predecir" en test estaría
            haciendo trampa. Coherente con el resto del pipeline TFM:
            scaler, PCA y GMM se ajustan UNA VEZ con train y se reutilizan
            tal cual en predict.

        Pipeline (4 pasos):
          1. Filtrar columnas con _select_features.
          2. Estandarizar con z-score (StandardScaler.fit_transform).
          3. Reducir con PCA si está activado.
          4. Ajustar GMM y reordenar regímenes por volatilidad.

        Métrica de calidad — BIC (Bayesian Information Criterion):
            El BIC mide cuán bien el modelo se ajusta a los datos
            penalizando complejidad. Cuanto MENOR sea, mejor es el
            modelo (menor desajuste y/o menos parámetros). Lo imprimimos
            al final para que el admin pueda compararlo entre ejecuciones
            con distintos n_regimes y decidir si añadir más estados
            mejora realmente el ajuste o solo añade ruido.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con features del periodo de entrenamiento.

        Returns
        -------
        RegimeDetector
            self, para permitir encadenamiento (.fit(...).predict(...)).
        """
        # 1) Seleccionar y limpiar features
        X = self._select_features(df).dropna()

        # 2) Estandarizar: cada feature pasa a media 0, desv 1.
        # fit_transform aprende la media/desv y las aplica al mismo tiempo.
        X_scaled = self.scaler.fit_transform(X)

        # 3) PCA opcional: reduce N features a n_components_pca componentes
        # principales. Útil cuando hay redundancia (features muy correlacionadas).
        if self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)

        # 4) Ajustar el GMM. Internamente usa el algoritmo EM
        # (Expectation-Maximization): alterna entre estimar a qué componente
        # pertenece cada punto y refinar los parámetros de cada componente.
        self.gmm.fit(X_scaled)
        self._fitted = True

        # 5) Reordenar regímenes por volatilidad ascendente.
        # Sin esto, el GMM asigna IDs aleatorios a cada componente — un día
        # el régimen 0 sería "calma", al siguiente arranque sería "crisis".
        # Forzando orden 0=calma, N-1=crisis garantizamos interpretabilidad.
        self._sort_regimes(X_scaled)

        # Log diagnóstico: BIC permite comparar la calidad entre diferentes
        # configuraciones (más bajo = mejor). N_features documenta cuántas
        # columnas pasaron el filtro de _select_features.
        print(f"  GMM ajustado: {self.n_regimes} regímenes, "
              f"{len(self._col_names)} features, BIC={self.gmm.bic(X_scaled):.0f}")

        return self

    def _sort_regimes(self, X: np.ndarray):
        """
        Reordena los regímenes detectados para que sigan un orden interpretable.

        Por qué hace falta:
            El GMM asigna IDs internos a cada componente de forma arbitraria
            (depende de la inicialización aleatoria). Una ejecución puede
            etiquetar "calma" como régimen 0, otra como régimen 2. Eso
            destrozaría la interpretación de DEFAULT_ALLOCATION en
            speculative_agent.py — la matriz dice "régimen 0 = calma" y si
            el GMM nos da el régimen 2 como calma, asignaríamos los pesos
            de crisis a un día tranquilo.

        La solución:
            Calculamos la VOLATILIDAD MEDIA de los días asignados a cada
            régimen y los reordenamos de menor a mayor. Después,
            cualquier predicción del GMM se pasa por self._order para
            obtener el ID consistente: 0=calma (menos volátil), 1=transición,
            2=crisis (más volátil).

        Por qué la volatilidad como criterio (y no el retorno medio):
            La volatilidad es la característica MÁS DISTINTIVA entre
            regímenes financieros. Un día de "crisis" puede tener cualquier
            retorno (a veces positivo por rebote, a veces negativo), pero
            siempre tiene volatilidad alta. La calma es lo opuesto: poco
            movimiento. Ordenar por volatilidad da una jerarquía estable
            entre ejecuciones.

        Parameters
        ----------
        X : np.ndarray
            Datos escalados (y opcionalmente reducidos por PCA) ya usados
            para el ajuste. Los reutilizamos para asignar cada punto a su
            componente y calcular la volatilidad por componente.
        """
        # Predecir a qué componente pertenece cada punto del train.
        states = self.gmm.predict(X)

        # Calcular la desviación estándar dentro de cada régimen.
        # Si un régimen quedó vacío (raro pero posible), le asignamos vol 0
        # para no crashear — luego quedará el primero del orden.
        vol_per_state = []
        for s in range(self.n_regimes):
            mask = states == s
            if mask.sum() > 0:
                vol_per_state.append(X[mask].std())
            else:
                vol_per_state.append(0.0)

        # argsort devuelve los índices que ordenarían el array.
        # Ej: vol = [0.8, 0.3, 1.5] → argsort = [1, 0, 2]
        # Lectura: "la posición 1 es la de menor vol → será el nuevo 0".
        self._order = np.argsort(vol_per_state)
        # Diccionario inverso para conversiones rápidas en otros sitios.
        self._mapping = {int(old): int(new) for new, old in enumerate(self._order)}

    def _prepare(self, df: pd.DataFrame) -> np.ndarray:
        """
        Aplica el pipeline de preprocesado a un DataFrame de predicción.

        CRÍTICO: usa scaler.transform() y pca.transform(), NO fit_transform.
        La diferencia es vital:
          - fit_transform aprende parámetros nuevos a partir de los datos.
          - transform aplica los parámetros YA APRENDIDOS en fit().
        Si en predict usáramos fit_transform, la escala de las features
        cambiaría según los datos de cada predicción → "lookahead bias"
        sutil que invalidaría las métricas. Aquí garantizamos que los
        datos de eval pasan por el MISMO pipeline numérico que train.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con las mismas columnas que se usaron en fit().
            Si faltan columnas, fallaría con KeyError (señal clara de
            inconsistencia entre entornos de train y predict).

        Returns
        -------
        np.ndarray
            Matriz numérica lista para alimentar al GMM.
        """
        X = df[self._col_names].dropna()
        # transform (no fit_transform): usa la media/desv aprendidas en fit.
        X_scaled = self.scaler.transform(X)
        if self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        return X_scaled

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Clasifica cada día del DataFrame en uno de los regímenes aprendidos.

        El régimen elegido para cada día es el de mayor probabilidad
        posterior tras un suavizado temporal. Este suavizado (controlado
        por self.smoothing) es importante: sin él, el GMM podría asignar
        un día concreto a "crisis" solo porque tuvo retorno anómalo aunque
        los días previos y siguientes sean de calma. El suavizado promedia
        las probabilidades en una ventana corta (5 días = 1 semana hábil)
        y solo cambia el régimen si el cambio es sostenido — exactamente
        lo que un humano consideraría un "cambio de régimen real".

        Parameters
        ----------
        df : pd.DataFrame
            Features del periodo a clasificar.

        Returns
        -------
        np.ndarray
            Array de enteros con un régimen por día: 0=calma, N-1=crisis.
        """
        assert self._fitted, "Ejecuta .fit() primero"

        # Probabilidades blandas de cada régimen para cada día.
        probs = self.predict_proba(df)

        # Suavizado temporal: media móvil de las probabilidades.
        # Nota: suavizamos PROBABILIDADES, no etiquetas duras. Promediar
        # etiquetas no tiene sentido (¿qué es la media entre régimen 0 y 2?).
        # Promediar probabilidades sí: 0.4/0.5/0.1 vecinos da 0.4/0.5/0.1
        # → régimen 1, robusto al ruido día a día.
        if self.smoothing > 1:
            smoothed_probs = pd.DataFrame(probs).rolling(
                self.smoothing, min_periods=1
            ).mean().values
        else:
            smoothed_probs = probs

        # argmax: para cada día (fila), el régimen con mayor probabilidad.
        return np.argmax(smoothed_probs, axis=1)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Devuelve las probabilidades posteriores ("blandas") de cada régimen.

        En vez de un único régimen por día (que es la salida de predict),
        este método devuelve una distribución de probabilidad sobre todos
        los regímenes. Útil para:
          - Diagnóstico: ver con qué confianza se clasifica cada día.
            Si las probabilidades son cercanas a [0.33, 0.33, 0.33], el
            día es ambiguo y el GMM no tiene una opinión clara.
          - Estrategias avanzadas: pesar la asignación por probabilidad
            en vez de elegir un solo régimen (no implementado actualmente,
            pero sería una extensión natural).

        El reordenamiento self._order asegura que la columna 0 sea siempre
        "calma" y la N-1 sea "crisis", consistente con predict().

        Parameters
        ----------
        df : pd.DataFrame
            Features del periodo a clasificar.

        Returns
        -------
        np.ndarray
            Forma (n_días, n_regimes). Cada fila suma 1 (probabilidades).
        """
        assert self._fitted, "Ejecuta .fit() primero"

        X_scaled  = self._prepare(df)
        # GMM devuelve probabilidades en el orden interno (aleatorio).
        probs_raw = self.gmm.predict_proba(X_scaled)
        # Reordenamos columnas para que sigan el mapping calma→crisis.
        return probs_raw[:, self._order]

    def describe_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera una tabla descriptiva de cada régimen detectado.

        Para cada régimen, calcula:
          - Cuántos días se asignaron a ese régimen.
          - Qué porcentaje del periodo total representan.
          - Retorno medio diario observado en esos días.
          - Volatilidad diaria observada en esos días.

        Es la herramienta clave para VALIDAR que los regímenes detectados
        son interpretables y tienen sentido financiero. Lo esperable:

          - "Calma" (régimen 0) : retorno medio ≥ 0, volatilidad baja.
          - "Transición" (régimen 1) : retorno medio cercano a 0, vol media.
          - "Crisis" (régimen 2) : retorno medio negativo, vol alta.

        Si la tabla resultante NO sigue ese patrón (los 3 regímenes con
        retornos similares), es señal de que el GMM no encontró estructura
        útil y conviene revisar las features de entrada o reducir n_regimes.

        Por qué se usa IVV como proxy:
            El detector trabaja con muchas features de mercado, pero para
            describir el régimen al humano hace falta UNA serie de retornos
            representativa. IVV (S&P 500) es el proxy estándar del mercado
            estadounidense. Si IVV_retornos no está en las features
            (universos sin IVV), caemos a la primera columna disponible
            como fallback — la descripción seguirá siendo útil aunque
            pierda interpretabilidad económica directa.

        Parameters
        ----------
        df : pd.DataFrame
            Features del periodo a describir. Típicamente train.

        Returns
        -------
        pd.DataFrame
            Tabla resumen con una fila por régimen detectado.
        """
        X = df[self._col_names].dropna()
        regimes = self.predict(df)

        # Buscar columna de retornos del mercado: IVV es nuestro proxy
        # estándar (S&P 500). Si no está, fallback a la primera columna.
        returns_col = None
        for c in self._col_names:
            if 'IVV_retornos' in c:
                returns_col = c
                break
        if returns_col is None:
            returns_col = self._col_names[0]

        summary = []
        # Nombres humanos para los 3 regímenes estándar.
        # Si n_regimes > 3, los adicionales aparecerán como "Estado N".
        names = {0: 'Calma', 1: 'Transición', 2: 'Crisis'}
        for r in range(self.n_regimes):
            mask = regimes == r
            # Saltar regímenes vacíos — pueden ocurrir con datos limitados.
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

    # ─── Aliases de retrocompatibilidad ─────────────────────────────────────
    # Antes de la refactorización a inglés se llamaba `descripcion_regimenes`.
    # Mantenemos el alias para que código antiguo (notebooks, scripts) siga
    # funcionando. Eliminar cuando se confirme que ningún consumidor lo usa.
    descripcion_regimenes = describe_regimes


# Alias de la clase por la misma razón histórica.
RegimenDetector = RegimeDetector

#todo:
#El BIC NO se muestra en ningún reporte. Solo aparece como un print(...) en consola al 
# ajustar el GMM, no se persiste en CSV ni se incluye en el dashboard ni en la tabla final. 
# Es información de diagnóstico interno que solo veremos si miras los logs del backend al lanzar 
# fit_speculative_agent (POST /admin/fase4/ajustar-especulativo).
#Recomendación: si usar el BIC como argumento académico ante el tribunal 
# ("justificamos n_regimes=3 porque el BIC no mejora con n_regimes=4"), hay que:
#Persistir el BIC en BD (tabla TrainedModel.train_metrics cuando entrenas el especulativo).
#O mostrarlo en el log del agente especulativo en el dashboard.
#Si no, mejor no mencionarlo en la memoria, porque diría algo que el código no expone 
# públicamente. simplificar el comentario del docstring o añadir la persistencia del BIC 
# en la BD.