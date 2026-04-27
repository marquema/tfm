"""
Clustering dinámico de activos basado en comportamiento reciente.

Qué es este módulo en una frase:
    Es el "agrupador de activos" del agente especulativo. Dada una lista
    de activos y sus precios, los CLASIFICA en N grupos según cómo se
    han movido recientemente. Los grupos se recalculan cada cierto
    tiempo, así que un activo puede cambiar de grupo a lo largo del
    histórico — exactamente lo que queremos.

Por qué clustering dinámico (y no estático por sector):
    La clasificación tradicional agrupa activos por su NATURALEZA: IVV
    es "renta variable", BND es "renta fija", IBIT es "cripto". Eso
    funciona como mapa mental, pero NO refleja cómo se comportan en el
    mercado real, sobre todo en crisis.

    Ejemplo histórico: en marzo 2020, durante el crash COVID, "activos
    defensivos" como muchas utilities y bonos cayeron casi tanto como
    el S&P 500. La correlación entre clases supuestamente distintas se
    disparó cerca de 1 — todo se movió a la vez. La clasificación
    sectorial daba una falsa sensación de diversificación.

    El clustering DINÁMICO ataca esto agrupando por COMPORTAMIENTO REAL:
      - Rolling window de retornos → para cada activo, un "perfil" de
        cómo se ha movido en los últimos N días.
      - K-Means agrupa activos con perfiles similares (independientemente
        de su sector "oficial").
      - Los clusters CAMBIAN con el tiempo: IBIT puede estar en el cluster
        "agresivo" un trimestre y, si baja la correlación con tech, pasar
        al cluster "descorrelado" al siguiente.

    Lo importante: el algoritmo NO conoce los sectores. Aprende los grupos
    desde los datos sin etiquetas previas (aprendizaje no supervisado).

Cómo se usa en el TFM:
    El agente especulativo (speculative_agent.py) llama a este módulo
    para saber, cada día, en qué cluster está cada activo. Luego cruza
    esa información con el régimen de mercado (regime_hmm.py) y la
    matriz DEFAULT_ALLOCATION para decidir pesos.

    El K-Means se aplica en VENTANA RODANTE de 60 días (~3 meses) — no
    en ventana expansiva — porque queremos REACTIVIDAD: clasificar al
    activo por su comportamiento ACTUAL, no por su historial completo.
    Ver explicación detallada en speculative_agent.py.

Vector de características por activo:
    Para cada ventana de N días, cada activo se describe por 4 números:
      1. Retorno acumulado en la ventana — "¿ha subido o bajado?".
      2. Volatilidad — "¿cuánto ha oscilado?".
      3. Skewness (asimetría) — "¿tiene cola larga a la izquierda
         (tendencia a caídas grandes) o a la derecha (subidas grandes)?".
      4. Max Drawdown en la ventana — "¿cuánto bajó desde su pico
         dentro de esos días?".

    Estos 4 números capturan el "perfil de riesgo-retorno" del activo
    sin necesitar etiquetas de sector. Activos con perfiles similares
    en estas 4 dimensiones acaban en el mismo cluster.

Uso:
    clusterer = AssetClusterer(n_clusters=3, window=60)
    labels_df = clusterer.rolling_clustering(prices_df)
    # labels_df: (n_dias x n_activos) con la etiqueta de cluster cada día
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class AssetClusterer:
    """
    Agrupa activos por similitud de comportamiento usando K-Means en ventana rodante.

    Encapsula el pipeline: cálculo del vector de características por activo
    → estandarización → K-Means → reordenación de clusters por momentum.

    K-Means en plano (para situarnos):
        Algoritmo clásico (Documentacion: MacQueen, 1967) que agrupa puntos en N "racimos"
        (clusters). Cada cluster tiene un centro (centroide) y los puntos
        se asignan al centro más cercano. El algoritmo itera entre dos pasos:
          1. Asignar cada punto al centroide más cercano.
          2. Recolocar cada centroide en la media de los puntos que tiene.
        Hasta que los centroides dejan de moverse. Sin supervisión: descubre
        los grupos solo, partiendo solo de las distancias entre puntos.

    Por qué K-Means y no algo más sofisticado:
        - Rápido: con 9-20 activos, el clustering es instantáneo.
        - Determinista (con semilla fija): mismas entradas → mismas salidas.
        - Suficiente para nuestro caso: queremos 3 grupos diferenciables
          en 4 dimensiones. Algoritmos más sofisticados (DBSCAN, GMM)
          serían overkill y aportarían poca mejora con tan pocos puntos.

    Parameters
    ----------
    n_clusters : int
        Número de grupos a formar. 3 es coherente con n_regimes del HMM
        para que la matriz DEFAULT_ALLOCATION (régimen x cluster) sea
        cuadrada y manejable. Con muchos más activos (50+), valores 4-5
        podrían capturar más matices.
    window : int
        Días de la ventana rodante usada para calcular el vector de
        características de cada activo. 60 ≈ 3 meses de trading: lo
        bastante largo para que el cálculo de volatilidad/skew sea
        estable, lo bastante corto para reaccionar a cambios de
        comportamiento del activo.
    random_state : int
        Semilla para reproducibilidad. K-Means tiene inicialización
        estocástica (k-means++); fijar la semilla garantiza que
        ejecuciones idénticas den resultados idénticos.
    """

    def __init__(self, n_clusters: int = 3, window: int = 60,
                 random_state: int = 42):
        self.n_clusters   = n_clusters
        self.window       = window
        self.random_state = random_state
        # StandardScaler para normalizar las 4 features del vector de cada
        # activo antes del K-Means. Sin esto, la volatilidad (típicamente
        # 0.01-0.05) sería dominada por el retorno acumulado (puede ser
        # 0.1-0.5) en el cálculo de distancias.
        self.scaler = StandardScaler()

    def _features_per_asset(self, returns: pd.DataFrame,
                            end_idx: int) -> np.ndarray:
        """
        Calcula la "huella estadística" de cada activo en la ventana actual.

        Para una ventana de los últimos `window` días terminada en
        `end_idx`, genera un vector de 4 números por cada activo. Esos
        4 números son lo que el K-Means usará para decidir qué activos
        son "parecidos" entre sí.

        Las 4 dimensiones, con su interpretación financiera:
          1. Retorno acumulado (np.nansum):
                Suma de los retornos diarios de la ventana. Mide cuánto
                ha subido o bajado el activo en estos N días en total.
                Es la dimensión más obvia: separa "ganadores" de "perdedores".
          2. Volatilidad (np.nanstd):
                Desviación estándar de los retornos. Mide cuánto oscila
                día a día. Un activo puede subir mucho con baja vol
                (tendencia clara) o subir lo mismo con alta vol (camino
                turbulento) — son perfiles distintos.
          3. Skewness / asimetría (pd.Series.skew):
                Si la distribución de retornos tiene cola larga a la
                izquierda (skew negativo), el activo tiene tendencia a
                caídas grandes inesperadas. Si tiene cola larga a la
                derecha (skew positivo), tiende a subidas grandes.
                Importante porque dos activos pueden tener mismo retorno
                y volatilidad pero asimetría opuesta — comportamiento
                muy distinto en eventos extremos.
          4. Max Drawdown intra-ventana:
                La peor caída desde un máximo dentro de la ventana.
                Captura dolor "psicológico" — un activo que cierra
                positivo pero cayó un -15% en medio se comportó muy
                distinto a uno que subió linealmente.

        Por qué se usan operadores `nan*`:
            np.nansum, np.nanstd ignoran NaN en lugar de propagarlos.
            Esto protege contra activos con datos incompletos en la
            ventana (ej. ETHA antes de su listado en julio 2024 → NaN
            que no debe contaminar el cálculo).

        Cálculo del Max Drawdown (técnica clásica):
            cum = retornos acumulados día a día
            peak = máximo histórico hasta cada día (cumulative max)
            dd = cum - peak (negativo o cero, mide caída desde pico)
            mdd = el mínimo de dd (la peor caída de la ventana)
            Esta forma vectorizada es mucho más rápida que un bucle
            explícito buscando máximos y mínimos.

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de retornos logarítmicos (ya calculados por el caller).
        end_idx : int
            Índice del último día de la ventana (exclusivo). La ventana
            cubre [end_idx - window, end_idx).

        Returns
        -------
        np.ndarray
            Matriz (n_activos × 4) con el vector de características por
            activo. Cada fila representa un activo, cada columna una
            de las 4 dimensiones descritas.
        """
        # max(0, ...) protege contra ventanas que se salen del inicio.
        # Si end_idx < window, simplemente usamos los días disponibles.
        start = max(0, end_idx - self.window)
        window_returns = returns.iloc[start:end_idx]

        features = []
        for col in window_returns.columns:
            r = window_returns[col].values

            # 1. Retorno acumulado (suma de retornos diarios)
            cum_return = np.nansum(r)
            # 2. Volatilidad (desviación estándar de retornos diarios)
            vol = np.nanstd(r)
            # 3. Skewness (asimetría de la distribución).
            # Pandas la implementa con corrección de Fisher (centrada en 0).
            # Si hay menos de 3 puntos, no es calculable → 0.0 como fallback neutro.
            skew = float(pd.Series(r).skew()) if len(r) > 2 else 0.0

            # 4. Max Drawdown intra-ventana (vectorizado, sin bucle)
            cum  = np.nancumsum(r)              # retornos acumulados día a día
            peak = np.maximum.accumulate(cum)   # máximo hasta cada día
            dd   = cum - peak                   # caída desde el pico
            mdd  = float(np.nanmin(dd)) if len(dd) > 0 else 0.0

            features.append([cum_return, vol, skew, mdd])

        return np.array(features)

    def cluster_at_date(self, returns: pd.DataFrame,
                        idx: int) -> np.ndarray:
        """
        Asigna cada activo a un cluster para una fecha concreta.

        Es el método "instantáneo": dado un día específico, calcula los
        vectores de características de cada activo, ejecuta K-Means y
        devuelve las etiquetas. rolling_clustering() lo invoca repetidamente
        para construir la serie temporal completa de clusters.

        Pipeline en este método:
          1. Calcular vector de 4 features por activo en la ventana
             [idx-window, idx) — vía _features_per_asset.
          2. Estandarizar las 4 dimensiones para que ninguna domine la
             distancia del K-Means.
          3. Ajustar K-Means y obtener etiquetas .
          4. Reordenar etiquetas por retorno acumulado del cluster:
             0 = peor momentum, N-1 = mejor momentum. Crítico para que
             la matriz DEFAULT_ALLOCATION se aplique correctamente
             (ver speculative_agent.py).

        Por qué `min(n_clusters, n_assets)`:
            Si tengo 3 clusters configurados pero solo 2 activos pasados
            como entrada, K-Means crashearía. Reducir k al número de
            activos disponibles es un fallback razonable — algunos
            clusters quedan vacíos pero el código sigue funcionando.
            Resumen: Para tontos: es como configurar tu coche para 5 pasajeros pero hoy 
            solo van 2. En lugar de no poder arrancar, simplemente quedan 3 asientos vacíos.
            El coche sigue rodando.¿Cuándo pasaría esto en el TFM? Casi nunca — el screener 
            garantiza al menos 9-15 activos. Pero el código se cubre la espalda por si 
            alguien llama el clusterer con un universo de prueba pequeño.

        Por qué `n_init=10`:
            K-Means tiene resultados sensibles a la inicialización. n_init=10
            ejecuta el algoritmo 10 veces con inicios distintos y se queda
            con el mejor (mínima inercia/dispersión intra-cluster). Coste
            x10 pero con N pequeño (3 clusters, ~9-20 activos) es trivial
            en milisegundos.
            Coste: hacer 10 K-Means en lugar de 1. Pero con 9-20 activos y 4 dimensiones, cada K-Means tarda 
            microsegundos. 10× microsegundos siguen siendo microsegundos. Ni se nota.

            Resumen: hacemos 10 intentos con puntos de partida distintos y nos quedamos con el mejor. Defensa 
            barata contra "mala suerte" en la inicialización aleatoria.


        Por qué reordenamos por retorno acumulado (columna 0):
            Sin reordenar, las etiquetas que devuelve K-Means son
            arbitrarias (depende del azar de inicialización). Hoy podría
            llamar "cluster 0" al de mejor momentum y mañana al peor.
            Eso destrozaría la matriz DEFAULT_ALLOCATION. Forzando la
            convención "0=peor, N-1=mejor" garantizamos que las celdas
            de la matriz [calma → cluster 2 = 0.70] se interpretan
            siempre igual: "70% en los activos con MEJOR momentum".

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame de retornos logarítmicos con una columna por activo.
        idx : int
            Índice de la fecha para la cual calcular los clusters. La
            ventana usada será [idx-window, idx).

        Returns
        -------
        np.ndarray
            Array (n_activos,) con etiquetas en {0, ..., k-1} donde
            k = min(n_clusters, n_assets). Reordenadas: 0=peor momentum,
            N-1=mejor momentum.
        """
        # 1) Vector de características por activo
        X = self._features_per_asset(returns, idx)

        # 2) Estandarizar cada dimensión por separado (z-score).
        # Nota: aquí SÍ usamos fit_transform (no transform), porque cada
        # ventana es independiente — no hay un "modelo persistente" de
        # escalas que mantener entre llamadas, a diferencia del HMM.
        X_scaled = self.scaler.fit_transform(X)

        # 3) Configurar y ejecutar K-Means
        n_assets = len(returns.columns)
        # Fallback defensivo: si hay menos activos que clusters, ajustar k.
        k = min(self.n_clusters, n_assets)
        km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        labels_raw = km.fit_predict(X_scaled)

        # 4) Reordenar clusters por retorno medio (columna 0 del vector
        #    de características es el retorno acumulado). Convertimos las
        #    etiquetas de K-Means en etiquetas con semántica estable.
        return_per_cluster = {}
        for c in range(k):
            mask = labels_raw == c
            # X[mask, 0]: retornos acumulados de los activos en este cluster.
            return_per_cluster[c] = X[mask, 0].mean()

        # Ordenar IDs de cluster de menor a mayor retorno medio.
        order = sorted(return_per_cluster.keys(), key=lambda c: return_per_cluster[c])
        # mapping: {id_kmeans: id_ordenado_por_momentum}
        mapping = {old: new for new, old in enumerate(order)}
        # Aplicar el mapping a cada etiqueta cruda.
        return np.array([mapping[l] for l in labels_raw])

    def rolling_clustering(self, prices: pd.DataFrame,
                           frequency: int = 20) -> pd.DataFrame:
        """
        Construye la serie temporal completa de clusters por activo.

        Es el método "alto nivel" que se llama desde el agente especulativo:
        recibe los precios del periodo y devuelve un DataFrame con la
        etiqueta de cluster de cada activo en cada día.

        Estrategia — re-clustering periódico, no diario:
            En vez de recalcular el clustering CADA día (carísimo y
            ruidoso), lo recalculamos cada `frequency` días (default 20,
            ~1 mes). Para los días intermedios entre re-clusterings,
            propagamos las últimas etiquetas calculadas con forward-fill.

            Por qué esto tiene sentido financiero:
              - El comportamiento de un activo no cambia drásticamente
                de un día para otro. Recalcular diariamente daría
                etiquetas casi idénticas con coste 20× mayor.
              - La frecuencia mensual captura cambios reales de régimen
                de un activo (ej. cuando una acción cambia de "neutro" a
                "agresivo" tras unos resultados trimestrales).
              - Reduce el turnover del agente especulativo: si los
                clusters cambiaran cada día, el agente rotaría cartera
                a diario y pagaría comisiones excesivas.

        Algoritmo paso a paso:
            1. Calcular retornos logarítmicos diarios desde precios.
            2. Recorrer el histórico saltando cada `frequency` días.
            3. Para cada día de "checkpoint", llamar a cluster_at_date
               y guardar las etiquetas en el DataFrame de salida.
            4. Forward-fill: los días entre checkpoints heredan las
               etiquetas del último checkpoint anterior.
            5. Backward-fill: el inicio del periodo (antes del primer
               checkpoint, cuando aún no hay window días de historia)
               se rellena hacia atrás con las primeras etiquetas
               calculadas — es la mejor aproximación posible.

        Por qué retornos LOGARÍTMICOS:
            log(p_t / p_{t-1}) en vez de (p_t - p_{t-1}) / p_{t-1}.
            Los retornos log son aditivos en el tiempo (suman) y simétricos
            (un +10% y un -9.09% logarítmicos suman 0, lo que cuadra con
            que el activo vuelve al precio inicial). Es la convención
            estándar en finanzas cuantitativas para ventanas largas.
            
            Resuemn: los retornos logarítmicos son una "moneda" que SE SUMA en lugar de 
                MULTIPLICARSE. Permite escribir todas las fórmulas de finanzas 
                cuantitativas con sumas (rápidas, lineales) en lugar de productos 
                (más lentos y propensos a errores numéricos). Es la convención estándar en 
                cualquier paper académico de finanzas cuantitativas serio.

        Parameters
        ----------
        prices : pd.DataFrame
            DataFrame de precios de cierre, una columna por activo.
        frequency : int
            Días entre re-clusterings consecutivos. 20 ≈ 1 mes hábil.
            Ventanas más cortas (5) detectan cambios rápido pero
            aumentan turnover y ruido. Ventanas más largas (60) son
            estables pero lentas en detectar cambios reales.

        Returns
        -------
        pd.DataFrame
            (n_fechas × n_activos) con la etiqueta de cluster por
            (día, activo). Las etiquetas son float (no int) porque el
            forward-fill puede introducir NaN antes de propagarlos —
            el caller (speculative_agent.py) las convierte a int al usarlas.
        """
        # 1) Retornos logarítmicos diarios (estándar en finanzas cuantitativas)
        returns = np.log(prices / prices.shift(1)).dropna()

        # DataFrame de salida pre-asignado, una fila por día y una col por activo.
        results = pd.DataFrame(index=returns.index, columns=prices.columns)

        # 2-3) Iterar checkpoints cada `frequency` días, empezando cuando
        # ya tenemos al menos `window` días de historia para el primer cálculo.
        for i in range(self.window, len(returns), frequency):
            labels = self.cluster_at_date(returns, i)
            date = returns.index[i]
            # Asignar la etiqueta de cluster a cada activo para esta fecha.
            for j, col in enumerate(prices.columns):
                results.loc[date, col] = labels[j]

        # 4-5) Rellenar días sin checkpoint:
        #   - ffill: días entre checkpoints heredan del último cálculo previo.
        #   - bfill: días al inicio (sin checkpoint anterior) toman el primero.
        # astype(float) explícito para evitar problemas con NaN→Object dtype.
        results = results.ffill().bfill().astype(float)

        return results

    # ─── Aliases de retrocompatibilidad ─────────────────────────────────────
    # Antes de la refactorización a inglés, los métodos se llamaban
    # `cluster_en_fecha` y `clustering_rolling`. Mantenemos los aliases para
    # que código antiguo (notebooks, scripts del pipeline) siga funcionando
    # sin tocar nada. Eliminar cuando se confirme que ningún consumidor los usa.
    cluster_en_fecha = cluster_at_date
    clustering_rolling = rolling_clustering
