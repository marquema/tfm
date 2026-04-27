"""
Agente especulativo basado en aprendizaje no supervisado (GMM + K-Means).

NOTA TERMINOLÓGICA — "HMM" en este TFM:
    Por razones históricas verás en este módulo, en nombres de columnas
    (ej. 'Especulativo_HMM') y en algún comentario antiguo la palabra
    "HMM". Lo que técnicamente usamos es un GMM (Gaussian Mixture Model)
    con suavizado temporal de las probabilidades posteriores —
    equivalente PRÁCTICO a un HMM verdadero para nuestro caso, pero más
    simple y estable numéricamente. Detalle completo de la decisión en
    el docstring de src/unsupervised/regime_hmm.py.

    El nombre se mantiene por retrocompatibilidad con código que importa
    'Especulativo_HMM' como etiqueta de estrategia en gráficas y tablas.
    En la memoria del TFM conviene presentarlo como "GMM + K-Means" para
    ser técnicamente preciso.

Rol en el TFM:
    Es el "agente contraste" que enfrentamos al PPO institucional. Mientras
    el PPO aprende una política compleja optimizando un reward (Sharpe
    rolling - phi·MDD - gamma·Turnover), el especulativo NO aprende: solo
    DETECTA estructura en los datos y reacciona a ella con reglas fijas.

    Esta diferencia es deliberada y aporta narrativa al tribunal:
        "¿De verdad hace falta DRL? ¿No bastaría con detectar regímenes
         de mercado y aplicar reglas heurísticas?"
    El especulativo da la respuesta empírica a esa pregunta — si su
    rendimiento es comparable al PPO, el TFM tendría poco que defender;
    si el PPO lo bate de forma sistemática, queda demostrado que la
    optimización por reward aporta valor real más allá del "pattern matching".

Arquitectura — dos componentes no supervisados que cooperan:

  1. Detector de RÉGIMEN de mercado (GMM con suavizado temporal).
     Modelo probabilístico que asume que el mercado atraviesa un número
     finito de "estados latentes" caracterizados por distribuciones
     estadísticas distintas. En este TFM usamos 3 estados, interpretables como:
         0 = CALMA      (volatilidad baja, retornos estables y positivos)
         1 = TRANSICIÓN (volatilidad media, dirección incierta)
         2 = CRISIS     (volatilidad alta, retornos predominantemente negativos)
     Se entrena sin supervisión: descubre solo los regímenes a partir de
     la dinámica observada en las features de mercado. Implementación
     concreta: GaussianMixture de scikit-learn (familia de los HMM
     simplificada — sin matriz de transición explícita, sustituida por
     suavizado de probabilidades). Detalle en regime_hmm.py.

  2. K-Means — agrupador de ACTIVOS por comportamiento.
     Clustering clásico (MacQueen, 1967): agrupa los activos en N grupos
     según su trayectoria reciente de retornos. En este TFM usamos 3
     clusters, que típicamente se interpretan como:
         0 = momentum bajo  (defensivos, refugio)
         1 = momentum medio (neutros)
         2 = momentum alto  (agresivos, en tendencia alcista)
     El K-Means se aplica en VENTANA RODANTE (cluster_window días) — los
     activos pueden cambiar de cluster a lo largo del tiempo según evolucionen.

Estrategia de asignación — tabla régimen × cluster:
    El cruce de los dos componentes determina los pesos de cartera. La
    matriz DEFAULT_ALLOCATION (más abajo) codifica la lógica:
        Régimen CALMA      → 70 % en momentum alto, 10 % en momentum bajo.
                             "Si el mercado está tranquilo, busquemos retorno".
        Régimen TRANSICIÓN → 25/50/25 (diversificación).
                             "Ante incertidumbre, diversificar".
        Régimen CRISIS     → 60 % en momentum bajo (defensivos), 10 % en alto.
                             "Cuando todo cae, refugiarse en lo más estable".

    Es una estrategia REACTIVA (responde al régimen detectado), no
    PROACTIVA (no anticipa cambios). Es exactamente la antítesis del PPO,
    que aprende a anticipar consecuencias de sus acciones.

Sobre las ventanas temporales — por qué rolling y no expanding:
    El K-Means de este módulo se aplica en VENTANA RODANTE (rolling) de
    cluster_window días, NO en ventana expansiva (expanding window). La
    distinción es deliberada y depende del propósito de cada componente:

      - K-Means quiere REACTIVIDAD: clasifica al activo por su
        comportamiento ACTUAL, no por su historial completo. Si TSLA fue
        defensiva 2018-2020 y agresiva 2021-2026, queremos que el K-Means
        la vea como "agresiva hoy", no como "agresiva en promedio". Un
        expanding window diluiría el cambio reciente al promediar 8 años
        de comportamiento heterogéneo. Rolling de 60 días "olvida" lo
        viejo y solo mira los últimos 3 meses.
      - El detector de régimen (GMM), en cambio, NO usa ninguna ventana — se entrena una sola
        vez con `fit()`. Razón: detecta TIPOS de régimen (calma/
        transición/crisis), no instantes concretos. Y los tipos son
        estructuralmente estables: una "crisis" en 1929, 2008, 2020 o
        2024 tiene siempre las mismas características (volatilidad alta
        + retornos negativos). Aprendemos esos patrones genéricos una
        vez y los reutilizamos en cualquier momento posterior.
      - El expanding window SÍ se usa, pero en otro sitio del TFM:
        en training_analysis.py para validar la robustez del PPO. Allí
        la pregunta es "¿el modelo mejora cuando le doy más historia?",
        que sí justifica acumular datos.

    Resumen para la memoria del TFM: tres usos distintos, tres elecciones
    distintas. Rolling cuando importa el AHORA; sin ventana cuando se
    aprenden patrones estructurales atemporales; expanding cuando se
    valida un modelo que aprende de cantidad de datos.

Por qué este agente NO sustituye al PPO:
    - No optimiza nada — la matriz de asignación está fijada por el
      diseñador, no aprendida. Cualquier cambio en mercados (ej. nueva
      cripto-crisis con dinámica distinta a 2008) requeriría redefinir
      la matriz a mano.
    - No considera coste de turnover ni penaliza drawdown explícitamente.
      Si el régimen oscila, el agente rota cartera y paga comisiones sin
      que el reward refleje ese coste durante el "aprendizaje".
    - Es interpretable y rápido (no necesita red neuronal) — útil como
      benchmark intermedio entre las baselines clásicas (Equal Weight,
      60/40) y el PPO. Da al tribunal un punto de referencia adicional.

Uso:
    agent = SpeculativeAgent(n_regimes=3, n_clusters=3)
    agent.fit(features_train, prices_train)            # ajusta GMM con train
    equity = agent.backtest(features_test, prices_test) # devuelve curva de equity

Documentacion Referencias:
    Rabiner, L. (1989). "A tutorial on hidden Markov models and selected
        applications in speech recognition." IEEE Proc. 77(2).
    MacQueen, J. (1967). "Some methods for classification and analysis of
        multivariate observations."
"""

import numpy as np
import pandas as pd

from src.unsupervised.regime_hmm import RegimeDetector
from src.unsupervised.asset_clustering import AssetClusterer


# ─────────────────────────────────────────────────────────────────────────────
# Matriz de asignación régimen × cluster
# ─────────────────────────────────────────────────────────────────────────────
# Es la "tabla heurística" que reemplaza al modelo aprendido del PPO. Cada
# fila representa un régimen detectado por el GMM; las columnas son los
# clusters de activos detectados por K-Means; los valores son la fracción
# del capital que se asigna al conjunto de activos pertenecientes a ese
# cluster en ese régimen.
#
# Estructura:
#                       Cluster 0       Cluster 1       Cluster 2
#                       (momentum bajo, (momentum medio,(momentum alto,
#                        defensivos)     neutros)        agresivos)
#   Régimen 0 (CALMA):       0.10           0.20            0.70
#   Régimen 1 (TRANSICIÓN):  0.25           0.50            0.25
#   Régimen 2 (CRISIS):      0.60           0.30            0.10
#
# Cada fila SUMA 1 (toda la cartera invertida — sin liquidez ni apalancamiento,
# coherente con la restricción long-only del entorno PortfolioEnv).
#
# Lógica heurística:
#   - En CALMA (régimen estable, baja volatilidad), perseguir retorno → 70 %
#     en cluster de momentum alto. Sesgo agresivo deliberado.
#   - En TRANSICIÓN (incertidumbre), diversificar uniformemente → 25/50/25.
#     Es el "no sé qué pasará, mantengamos exposición equilibrada".
#   - En CRISIS (alta volatilidad, retornos negativos), refugiarse en
#     defensivos → 60 % en momentum bajo (típicamente bonos, utilities).
#
# Esta matriz es el ÚNICO conocimiento "humano" que entra en el agente. Todo
# lo demás (qué es CALMA, qué activos son agresivos hoy) lo descubren los
# modelos no supervisados a partir de los datos. La heurística es deliberadamente
# simple para que el contraste con el PPO sea claro: si esta tabla bate al
# PPO, el TFM tiene poco que defender; si pierde, queda demostrado que
# aprender por reward aporta más valor que reglas fijas.
DEFAULT_ALLOCATION = {
    0: np.array([0.10, 0.20, 0.70]),   # Calma:      agresivo en momentum alto
    1: np.array([0.25, 0.50, 0.25]),   # Transición: diversificado
    2: np.array([0.60, 0.30, 0.10]),   # Crisis:     concentrar en defensivos
}


class SpeculativeAgent:
    """
    Agente especulativo que combina régimen (GMM con suavizado) + clustering (K-Means)
    para generar pesos de cartera reactivos sin entrenamiento supervisado.

    Coordina dos modelos no supervisados que viven en módulos 
    (regime_hmm.py y asset_clustering.py) y los une mediante la matriz
    DEFAULT_ALLOCATION para producir la decisión final.

    Patrón arquitectonico — composición sobre herencia:
        El agente NO hereda de RegimeDetector ni de AssetClusterer; los
        compone como atributos. Esto permite probar cada componente por
        separado (ej. cambiar el detector de régimen por uno HMM "puro"
        si en el futuro se justifica, sin tocar el agente, o usar otro
        algoritmo de clustering sin afectar al detector).

    Parameters
    ----------
    n_regimes : int
        Número de estados latentes del detector (componentes del GMM).
        3 es el estándar académico
        para series financieras (calma/transición/crisis), apoyado en
        múltiples estudios (documentacion: Hamilton, 1989; Ang & Bekaert, 2002). Más
        estados (4-5) detectan matices, pero son más difíciles de
        interpretar y requieren más datos para converger.
    n_clusters : int
        Número de grupos de activos generados por K-Means. 3 es coherente
        con n_regimes para que la matriz régimen×cluster sea cuadrada y
        manejable. Cambiar este valor obliga a redefinir la matriz
        DEFAULT_ALLOCATION (que asume 3 columnas).
    cluster_window : int
        Días de la ventana rolling sobre la que K-Means recalcula los
        clusters. 60 ≈ 3 meses de trading. Ventanas más cortas (20)
        reaccionan rápido a cambios de comportamiento pero son ruidosas;
        ventanas más largas (120) son estables pero lentas en detectar
        cuando un activo cambia de régimen propio.
    allocation : dict, optional
        Matriz de asignación personalizada {regime_id: pesos_por_cluster}.
        Si None, usa DEFAULT_ALLOCATION. Permite experimentar con otras
        heurísticas sin tocar el código del agente.
    """

    def __init__(self, n_regimes: int = 3, n_clusters: int = 3,
                 cluster_window: int = 60, allocation: dict = None):
        # Composición: instanciamos los dos componentes no supervisados.
        # El agente actúa como "director de orquesta" sobre ellos.
        self.detector   = RegimeDetector(n_regimes=n_regimes)
        self.clusterer  = AssetClusterer(n_clusters=n_clusters, window=cluster_window)
        self.n_clusters = n_clusters
        # `or` permite que el caller pase su propia matriz; si no, defaults.
        self.allocation = allocation or DEFAULT_ALLOCATION
        # Flag de seguridad: predict()/backtest() requieren fit() previo.
        self._fitted    = False

    def fit(self, features_train: pd.DataFrame,
            prices_train: pd.DataFrame) -> 'SpeculativeAgent':
        """
        Ajusta los componentes no supervisados sobre datos de train.

        Diseño asimétrico de los dos componentes (es deliberado):
          - GMM (detector de régimen): SE ENTRENA aquí. Aprende los regímenes a partir del
            histórico de train y memoriza su matriz de transición y las
            distribuciones de cada régimen. Después, en predict, asigna
            cada día a uno de los regímenes aprendidos.
          - K-Means: NO se entrena aquí, se aplica EN VENTANA RODANTE
            durante predict/backtest. Esto significa que cada día los
            clusters se recalculan con los últimos `cluster_window` días.
            Permite que un activo cambie de cluster a lo largo del tiempo
            (ej. una tecnológica que era "agresiva" en 2021 puede pasar a
            "defensiva" en 2023) — es la flexibilidad que hace al agente
            mínimamente reactivo a cambios estructurales.

        Por eso prices_train se acepta pero NO se usa actualmente — es
        un parámetro reservado para futuras extensiones (ej. clustering
        global previo a la ventana rodante). Dejarlo en la firma anticipa
        esa evolución sin romper retrocompatibilidad.

        Parameters
        ----------
        features_train : pd.DataFrame
            Features normalizadas del periodo de train (las que alimentan
            al GMM para detectar regímenes).
        prices_train : pd.DataFrame
            Precios de train. Reservado para extensiones; el K-Means
            actual se basa solo en datos de test (rolling).

        Returns
        -------
        SpeculativeAgent
            self, para permitir encadenamiento (.fit(...).backtest(...)).
        """
        print("  [1/2] Ajustando GMM para detección de regímenes...")
        self.detector.fit(features_train)

        # Mostrar resumen de regímenes detectados (volatilidad, retorno medio,
        # n días por régimen…). Útil para diagnosticar si los regímenes son
        # interpretables — si los 3 son casi idénticos, el GMM no encontró
        # estructura clara y los pesos serán erráticos.
        desc = self.detector.describe_regimes(features_train)
        print(desc.to_string(index=False))

        print("  [2/2] Clustering de activos listo (se ejecuta rolling en predict).")
        self._fitted = True
        return self

    def generate_weights(self, features: pd.DataFrame,
                         prices: pd.DataFrame) -> pd.DataFrame:
        """
        Genera los pesos de cartera día a día reaccionando al estado del mercado.

        Algoritmo (4 pasos por cada día del periodo):
          1. PREGUNTAR AL GMM "¿qué régimen estamos hoy?" → entero {0, 1, 2}.
          2. PREGUNTAR AL K-MEANS "¿a qué cluster pertenece cada activo hoy?"
             → vector de N etiquetas, una por activo.
          3. CONSULTAR LA MATRIZ régimen × cluster → vector de pesos por cluster.
          4. REPARTIR el peso de cada cluster ENTRE LOS ACTIVOS DE ESE CLUSTER:
             si cluster 2 tiene peso 0.7 y hay 3 activos en él, cada uno recibe
             0.7/3 ≈ 0.233. Esto evita concentrar todo en un solo activo.
          5. NORMALIZAR para que la suma = 1 (defensa contra clusters vacíos).
                Régimen CALMA: [0.10, 0.20, 0.70]   ← cluster 0 = 0.10, cluster 1 = 0.20, cluster 2 = 0.70

                Para repartir esos pesos entre activos hace falta saber qué activos hay en cada cluster. La idea es:
                Cluster 0 → su 0.10 se reparte entre los activos del cluster 0.
                Cluster 1 → su 0.20 se reparte entre los activos del cluster 1.
                Cluster 2 → su 0.70 se reparte entre los activos del cluster 2.
                El problema "cluster vacío": ¿qué pasa si un día el K-Means no asigna ningún activo al cluster 1? (Por ejemplo: por azar, todos los activos cumplen criterios de cluster 0 o cluster 2 ese día concreto).


                Activos:  IVV  BND  IBIT  ETHA  AAPL
                Cluster:   0    0    2     2     2     ← nadie en cluster 1
                Si simplemente repartiéramos:

                Cluster 0 (2 activos) → 0.10 / 2 = 0.05 a IVV y BND.
                Cluster 1 (0 activos) → 0.20 / ? ← se pierde, no se asigna a nadie.
                Cluster 2 (3 activos) → 0.70 / 3 = 0.233 a IBIT, ETHA, AAPL.
                Total invertido: 0.05+0.05 + 0+0+0 + 0.233+0.233+0.233 = 0.80, no 1.

                → Nos quedan 0.20 sin invertir, liquidez en cuenta que no estaba prevista. La cartera no está "plenamente invertida" como debería.

                La solución: al final del bucle dividimos todo entre la suma actual:
                w = w / w.sum()   # ahora suma exacta 1
                Resumen: si por mala suerte un cluster queda vacío, parte del dinero quedaría en cash. 
                La normalización final redistribuye ese sobrante entre los clusters que sí 
                tienen activos, garantizando que TODO el capital queda trabajando.


        Decisiones de diseño:
          - El K-Means se ejecuta ANTES del bucle (rolling_clustering devuelve
            todas las etiquetas de una vez). No hacemos una llamada por etiqueta. 
            Es más eficiente que recalcular  el clustering cada día — pandas/sklearn están optimizados para
            operaciones vectorizadas.
          - Si un día no tiene clustering asignado (común al inicio del
            periodo, antes de que la ventana rolling tenga `cluster_window`
            días de historia), usamos labels=0 como fallback. Eso concentra
            el día en el cluster 0 (defensivos), comportamiento conservador
            seguro. Cuando aún no tenemos información para clasificar, no inventamos. Tomamos la 
            postura más conservadora posible (todo en defensivos) hasta que la ventana se 
            llene y podamos clasificar de verdad
          - Si el régimen detectado no aparece en self.allocation (no
            debería ocurrir con DEFAULT_ALLOCATION, pero protege contra
            allocations personalizadas incompletas), caemos al equiponderado
            entre clusters como fallback. Es decir, si nadie nos dijo qué hacer en este régimen, 
            repartimos a partes iguales entre todos los clusters. Un fallback defensivo y 
            razonable que evita errores y mantiene el agente operativo.

        Parameters
        ----------
        features : pd.DataFrame
            Features del periodo a evaluar (input del GMM).
        prices : pd.DataFrame
            Precios de cierre del periodo (input del K-Means rolling).

        Returns
        -------
        pd.DataFrame
            Forma (n_días, n_activos) con pesos ∈ [0, 1] que suman 1 por fila.
            Listo para multiplicar por retornos diarios y obtener PnL.
        """
        # Guard: predict sobre modelo no ajustado daría resultados aleatorios.
        assert self._fitted, "Ejecuta .fit() primero"

        # Una sola llamada al GMM: devuelve un array con el régimen detectado
        # para CADA día del periodo. Vectorizado, mucho más rápido que un
        # predict día por día.
        regimes = self.detector.predict(features)

        # Una sola llamada al K-Means rolling: devuelve un DataFrame
        # (días × activos) con la etiqueta de cluster de cada activo cada día.
        # Internamente recomputa el clustering en una ventana de cluster_window
        # días que se desliza por el periodo.
        clusters = self.clusterer.rolling_clustering(prices)

        # Alineamos índices: solo días con features válidas Y dentro del rango
        # que devolvió el GMM (puede ser más corto si hay NaN al inicio).
        common_dates = features.dropna().index[:len(regimes)]
        n_assets = len(prices.columns)

        # DataFrame de salida pre-asignado para evitar concatenaciones costosas
        # dentro del bucle.
        weights = pd.DataFrame(
            index=common_dates,
            columns=prices.columns,
            dtype=np.float64
        )

        for i, date in enumerate(common_dates):
            regime = regimes[i]

            # ── Paso 3: pesos base por cluster para este régimen 
            # Si el régimen no está en la tabla, fallback a equiponderado:
            # 1/n_clusters en cada cluster (cartera totalmente diversificada). es el equivalente al
            # "divídelo a partes iguales" de cuando no sabes qué decisión tomar. Mejor que 
            # equivocarte a lo grande.
            cluster_weights = self.allocation.get(
                regime, np.ones(self.n_clusters) / self.n_clusters
            )

            # ── Paso 4: traducir pesos por cluster → pesos por activo 
            # Cada activo j recibe (peso_de_su_cluster) / (n_activos_en_su_cluster).
            w = np.zeros(n_assets)
            if date in clusters.index:
                labels = clusters.loc[date].values.astype(int)
            else:
                # Fallback conservador: si no hay clustering para este día
                # (típicamente al inicio antes de que la ventana se llene),
                # asumimos todos los activos en cluster 0 (defensivos).
                
                labels = np.zeros(n_assets, dtype=int)

            for j in range(n_assets):
                # min() defiende contra el caso pathological en que el K-Means
                # devuelva una etiqueta fuera de rango (no debería, pero cuesta
                # 0 protegerse). ¿Cuándo pasaría esto? En condiciones normales, nunca. 
                # El K-Means se configura con n_clusters=3 y no debería devolver etiquetas fuera de [0, 1, 2]. 
                # Pero: algún bug futuro en el código del clusterer. Un edge case raro (ej. NaN convertido a int da números absurdos).
                # Un caller que pase un clusterer mal configurado.
                # Coste de la defensa: una llamada a min() que tarda nanosegundos.
                # Beneficio: el agente nunca crashea por un dato malformado, devuelve resultados degradados pero válidos.
                # Resumen: cinturón de seguridad del paso. En condiciones normales no se nota, pero en un accidente improbable evita que todo el sistema falle. Cuesta nada y previene crashes raros.
                
                cluster_id = min(labels[j], len(cluster_weights) - 1)
                # Activos en el mismo cluster reparten su peso a partes iguales.
                n_in_cluster = max(1, (labels == labels[j]).sum())
                w[j] = cluster_weights[cluster_id] / n_in_cluster

            # ── Paso 5: normalizar para garantizar Σ w = 1 
            # En teoría ya suma 1 si la matriz allocation está bien construida,
            # pero si algún cluster queda vacío (todos los activos en otros
            # clusters), parte del peso "se pierde". Renormalizamos para que
            # la cartera siempre esté plenamente invertida.  hacemos un último ajuste 
            # matemático para garantizar que el 100 % del capital está invertido, sin trozos
            # perdidos por casos raros. Es como repesar la balanza al final por si los 
            # pesos individuales no cuadraron exactamente — ahora seguro que sí.

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

        Equivalente conceptual al backtest del PPO (PortfolioEnv.step en
        bucle), pero más sencillo: no hay reward, no hay observation, no
        hay agente que decida — los pesos ya se calcularon todos en
        generate_weights. Aquí solo APLICAMOS los pesos día a día sobre
        precios reales y vemos cómo evoluciona el capital.

        Algoritmo (idéntico en filosofía al PortfolioEnv del PPO para que
        las métricas sean comparables):
          1. Calcular pesos diarios con generate_weights().
          2. Calcular retornos simples (pct_change) de cada activo.
          3. Para cada día:
             a) Coste por turnover: |w_hoy - w_ayer| x balance x commission.
                Penaliza rotación excesiva igual que en el entorno PPO.
             b) Aplicar retorno: balance x (1 + Σ wᵢ · rᵢ).
             c) Clip a 1e-6 para evitar quiebra absoluta que rompería
                divisiones posteriores.

        Coherencia con el PPO:
          - Mismo capital inicial (10 000).
          - Misma estructura de comisión (0.1 % del turnover).
          - Mismo cálculo de retornos diarios.
          → Las métricas resultantes (Sharpe, Retorno, MDD…) son
            DIRECTAMENTE COMPARABLES con las del PPO en la tabla final.
            Es lo que permite escribir en la memoria "el PPO gana en Sharpe
            al especulativo por 0.5 puntos" con rigor.

        Parameters
        ----------
        features : pd.DataFrame
            Features del periodo de test.
        prices : pd.DataFrame
            Precios de cierre del periodo de test.
        initial_balance : float
            Capital inicial. 10 000 alinea con el resto del pipeline.
        commission : float
            Comisión proporcional por turnover. 0.001 = 0.1 %.

        Returns
        -------
        pd.Series
            Serie temporal con la evolución del valor de la cartera.
            Nombre 'Especulativo_HMM' para que aparezca etiquetada así en
            las gráficas y tablas que comparan estrategias.
        """
        # 1) Calcular pesos diarios (delega en generate_weights).
        weights_df = self.generate_weights(features, prices)

        # 2) Retornos diarios de cada activo. fillna(0) trata el primer día
        #    (que no tiene día anterior) como retorno cero — no penaliza ni
        #    premia el inicio.
        returns = prices.pct_change().fillna(0)
        returns = returns.loc[weights_df.index]

        balance = initial_balance
        equity = [balance]

        # Pesos previos arrancan en 0 (sin posiciones). El primer rebalanceo
        # mueve TODO el capital de cash a la cartera inicial → turnover ≈ 1
        # → coste ≈ initial_balance × commission. Es un coste real que el
        # PPO también paga al inicio, así que mantenerlo es correcto.
        previous_weights = np.zeros(len(prices.columns))

        for i in range(len(weights_df)):
            w = weights_df.iloc[i].values.astype(float)

            # Coste de rebalanceo: cuánto cambian los pesos respecto a ayer.
            # turnover ∈ [0, 2] (0 = sin cambio, 2 = giro completo de cartera).
            turnover = np.abs(w - previous_weights).sum()
            cost = turnover * balance * commission
            balance -= cost

            # Retorno ponderado del día: Σ (peso_i × retorno_i).
            if i < len(returns):
                daily_return = (returns.iloc[i].values * w).sum()
                balance *= (1 + daily_return)

            # dato base numérico: evita balance ≤ 0 que rompería operaciones
            # posteriores (división por cero al calcular retornos relativos).
            balance = max(balance, 1e-6)
            equity.append(balance)
            previous_weights = w

        return pd.Series(equity, name='Especulativo_HMM')

    # ─── Alias de retrocompatibilidad ─────────────────────────────────────
    # Antes de la refactorización a inglés, el método se llamaba
    # `generar_pesos` (en español). El alias permite que código externo
    # antiguo (notebooks, scripts del pipeline) siga funcionando sin tocar
    # nada. Eliminar cuando se confirme que ningún consumidor lo usa.
    generar_pesos = generate_weights


# Alias de la constante por la misma razón histórica que el método de arriba.
ASIGNACION_DEFAULT = DEFAULT_ALLOCATION
