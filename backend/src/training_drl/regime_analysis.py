"""
Análisis post-hoc del rendimiento del agente DRL por régimen de volatilidad.

Reusmen:
    Es la "lupa de evaluación" del TFM. Toma los resultados out-of-sample
    del PPO y de las baselines, los segmenta por régimen de mercado
    (calma/transición/crisis) y calcula métricas separadas en cada uno.
    Sirve para responder al tribunal: "¿el agente bate a las baselines
    en TODOS los regímenes o solo en los favorables?".

Pregunta concreta que responde:
    Sin este módulo, la única afirmación posible es "el PPO da Sharpe 1.78
    en el periodo de test". Pero ese Sharpe puede ser 3.0 en mercado
    alcista y -0.5 en crisis. Con este análisis podemos defender:
        "El PPO supera a Equal Weight en los tres regímenes. En calma el
         margen es modesto (+0.2 Sharpe); en crisis el margen es decisivo
         (+1.1 Sharpe), demostrando que el agente aporta sobre todo en
         momentos turbulentos — exactamente el escenario donde la gestión
         activa más debe justificar su existencia."
         
        Esa cifra esconde una hecho incómoda:
        En mercado alcista, el modelo rinde como un genio. Pero claro: en mercado a
        lcista CASI CUALQUIER modelo rinde bien — incluso comprar y olvidar. No es 
        mérito de mi modelo, es del mercado. En crisis, el modelo pierde dinero. 
        Justo cuando MÁS lo necesitamos, falla.
        
        Es decir: es como decir "mi nota media en bachillerato fue 7". Vale, ¿pero esa 
        media es 7 en todo, o es 9 en filosofía y 4 en matemáticas? La nota media 
        no te dice si serías buen ingeniero. Las notas desglosadas sí.

Diferencia con regime_hmm.py — son módulos COMPLEMENTARIOS, no duplicados:
    - regime_hmm.py:
        Detector EN PRODUCCIÓN. Usa Gaussian Mixture Model (GMM) para que
        el agente especulativo decida pesos de cartera EN EL MOMENTO según
        el régimen detectado. Trabaja sobre features completas (retornos,
        volatilidades, correlaciones, etc.) y aprende los regímenes de
        forma no supervisada con un modelo probabilístico complejo.
    - regime_analysis.py (este módulo):
        Análisis A POSTERIORI. Usa una clasificación simple basada en
        percentiles de volatilidad sobre IVV (proxy de mercado). El
        objetivo NO es decidir nada, sino ETIQUETAR cada día del periodo
        de test para cuantificar el rendimiento del PPO en cada zona.
        La simplicidad es deliberada: el tribunal puede entender el
        criterio en una frase ("alta vol = top 33% de la volatilidad
        histórica"), lo que da credibilidad al análisis.

    En resumen: HMM/GMM CREA decisiones; este módulo LAS EVALÚA después
    de tomarlas. Son piezas distintas con propósitos distintos.

Metodología:
    Clasificación basada en percentiles de volatilidad rolling del activo
    de referencia (IVV, S&P 500 — el proxy estándar del mercado USA).
        - Bajo (0):   vol_20d ≤ percentil 33 del periodo de entrenamiento.
        - Medio (1):  percentil 33 < vol_20d ≤ percentil 67.
        - Alto (2):   vol_20d > percentil 67.
     
    Rolling windown y no expanding:   
        Queremos reactividad al régimen ACTUAL del mercado, no estabilidad histórica.

    Crítico: los umbrales se calculan SOLO con datos de entrenamiento
    (primer 80%). Si los calculáramos con todo el dataset, los percentiles
    incluirían información del periodo de test → lookahead bias sutil que
    invalidaría la separación train/test del TFM.

Salida del módulo:
    - src/reports/regime_analysis.png — gráfica con curvas de equity de
      cada estrategia y zonas sombreadas por régimen (verde/naranja/rojo).
      Lista para insertar en la memoria.
    - src/reports/regime_metrics.csv  — tabla con métricas por estrategia
      y por régimen.

Convención de nombres:
    En este módulo conviven dos vocabularios para los mismos 3 regímenes:
        - "Baja Vol. / Vol. Media / Alta Vol." → técnico (eje del gráfico
          superior, métricas internas).
        - "Calma / Transición / Crisis"        → divulgativo (eje del
          gráfico inferior, terminología que el tribunal entiende mejor).
    Ambos refieren a la misma clasificación 0/1/2; la doble etiqueta es
    deliberada para que la gráfica sea autoexplicativa al insertarla en
    la memoria sin necesidad de leyenda adicional.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla para guardar PNGs en servidor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("src/reports", exist_ok=True)


# ---------------------------------------------------------------------------
# Clasificación de regímenes
# ---------------------------------------------------------------------------

def classify_regimes(df_prices: pd.DataFrame,
                     reference_ticker: str = None,
                     window: int = 20,
                     low_percentile: float = 33,
                     high_percentile: float = 67,
                     train_threshold_pct: float = 0.8) -> pd.Series:
    """
    Etiqueta cada día del periodo en uno de tres regímenes de volatilidad.

    Lógica de clasificación:
      Régimen 0 (BAJA / Calma):    volatilidad ≤ percentil 33 del train.
      Régimen 1 (MEDIA / Trans.):  percentil 33 < volatilidad ≤ percentil 67.
      Régimen 2 (ALTA / Crisis):   volatilidad > percentil 67.

    Por qué la VOLATILIDAD del IVV (no retorno):
        La volatilidad es la característica más estable de cada régimen
        financiero. Una crisis SIEMPRE va con volatilidad alta; un retorno
        negativo no es necesariamente crisis (puede ser corrección suave).
        Usar IVV (S&P 500) como referencia da una clasificación que el
        tribunal entiende inmediatamente: "alta volatilidad en la bolsa
        americana = mercado nervioso = crisis para nuestro propósito".

    Por qué los percentiles del TRAIN (no del dataset completo):
        Si calculamos percentiles con todo el histórico, los umbrales
        incluirían información del periodo de test → lookahead bias sutil.
        En la práctica diría "este día es alta vol" basándose parcialmente
        en lo volátil que será el FUTURO de ese día. Calibrar solo con
        train mantiene la honestidad académica del análisis out-of-sample.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Precios de cierre en orden cronológico ascendente. Una columna por
        activo. Las columnas pueden venir como 'IVV' o 'IVV_Close' (admite
        ambos formatos vía búsqueda case-insensitive).
    reference_ticker : str, optional
        Columna sobre la que calcular volatilidad. Si es None, busca
        automáticamente la columna que contenga 'IVV' (S&P 500 = proxy
        de mercado USA). Si tampoco existe, cae a la primera columna
        disponible como fallback.
    window : int
        Días para el cálculo de volatilidad rolling. 20 ≈ 1 mes de trading
        — captura el régimen actual sin arrastrar mucho ruido del pasado.
    low_percentile, high_percentile : float
        Cortes de percentil que delimitan los tres regímenes (33/67 por
        defecto = división en tercios). Cambiarlos altera la proporción
        relativa de días en cada régimen sin tocar la lógica del módulo.
    train_threshold_pct : float
        Fracción de datos usada como train para calibrar umbrales.
        Por defecto 0.8, alineado con el split global del pipeline TFM.

    Returns
    -------
    pd.Series
        Régimen (0, 1, 2) por día, indexado igual que df_prices.
        Los días iniciales con NaN (warmup de la ventana rolling de 20d)
        se asignan al régimen 1 (transición/neutro) — fallback razonable
        para no perder esos días del análisis.
    """
    # Selección automática del activo de referencia.
    # Usamos IVV (S&P 500) como proxy del mercado, NO IBIT (cripto).
    # IBIT tiene volatilidad estructuralmente mayor que la renta variable
    # (típicamente 60% anualizada vs 15% del S&P 500). Si lo usáramos como
    # referencia, los percentiles vivirían en una escala "cripto" donde
    # incluso una crisis bursátil tradicional parecería "calma" relativa.
    # El proxy correcto es siempre el activo más representativo del
    # mercado tradicional al que el TFM hace referencia.
    if reference_ticker is None:
        ivv_cols = [c for c in df_prices.columns if 'IVV' in c.upper()]
        reference_ticker = ivv_cols[0] if ivv_cols else df_prices.columns[0]

    if reference_ticker not in df_prices.columns:
        raise ValueError(
            f"Ticker de referencia '{reference_ticker}' no encontrado en df_prices."
        )

    # Retornos logarítmicos del activo de referencia
    ref_prices = df_prices[reference_ticker]
    log_returns = np.log(ref_prices / ref_prices.shift(1))
    rolling_vol = log_returns.rolling(window).std()

    # Umbrales calibrados solo sobre el período de entrenamiento
    n_train = int(len(rolling_vol) * train_threshold_pct)
    vol_train = rolling_vol.iloc[:n_train].dropna()

    threshold_low  = np.percentile(vol_train, low_percentile)
    threshold_high = np.percentile(vol_train, high_percentile)

    # Clasificación: 0=Baja, 1=Media, 2=Alta
    regime = pd.Series(1, index=df_prices.index, name='regime', dtype=int)
    regime[rolling_vol <= threshold_low]  = 0
    regime[rolling_vol > threshold_high]  = 2
    regime[rolling_vol.isna()]= 1  # NaN al inicio -> régimen neutro

    return regime


# ---------------------------------------------------------------------------
# Ejecución del agente por subperíodo
# ---------------------------------------------------------------------------
def _run_agent(model, features_path: str, prices_path: str,
               start_idx: int, end_idx: int) -> list:
    """
    Ejecuta el agente PPO en un subrango temporal y devuelve la curva de equity.

    Función auxiliar interna (prefijo _) usada por analyze_regimes() para
    obtener los valores diarios de la cartera del PPO y luego cruzarlos
    con los regímenes etiquetados.

    No reentrenamos nada aquí: el modelo ya viene cargado. Solo recorremos
    el periodo paso a paso con `deterministic=True` (sin sampling), que
    da resultados reproducibles. Por qué importa: si reentrenáramos en el periodo 
    de test, estaríamos haciendo trampa. El modelo aprendería del futuro 
    y luego "predecir" sobre lo aprendido sería trivial. 

    Parameters
    ----------
    model : stable_baselines3.PPO
        Modelo PPO ya cargado/entrenado.
    features_path, prices_path : str
        Rutas a los CSVs del pipeline (ver data_downloader.py).
    start_idx : int
        Índice de inicio del subconjunto temporal (típicamente split_idx
        para empezar en el periodo de test).
    end_idx : int or None
        Índice final exclusivo. None = hasta el final.

    Returns
    -------
    list[float]
        Valores de cartera en cada step, incluido el valor inicial.
    """
    # try/except con el mismo import en ambas ramas: defensa por si en el
    # futuro se cambian rutas. Hoy ambas resuelven igual.
    try:
        from src.training_drl.environment_trading import PortfolioEnv
    except ImportError:
        from src.training_drl.environment_trading import PortfolioEnv

    env = PortfolioEnv(features_path, prices_path,
                       start_idx=start_idx, end_idx=end_idx)
    obs, _ = env.reset()
    done   = False
    values = [env.initial_balance]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        values.append(info['value'])

    return values

# Análisis de métricas por régimen
def metrics_by_regime(value_series: pd.Series,
                      regimes: pd.Series,
                      annual_rf: float = 0.04) -> pd.DataFrame:
    """
    Calcula métricas financieras separadas por régimen de volatilidad.

    Es el corazón analítico del módulo: dada una curva de equity de una
    estrategia (PPO, Equal Weight, etc.) y la etiqueta de régimen por día,
    devuelve cuántos días estuvo cada régimen activo y qué métricas dio
    la estrategia DURANTE esos días — Sharpe, Sortino, retorno, vol, MDD.

    Caveat técnico — segmentar serie por máscara:
        Cuando filtramos los días de "calma" tomamos solo esas filas y las
        concatenamos. Esto introduce DISCONTINUIDADES temporales: pasamos
        del lunes al jueves saltándonos los días que no eran calma. El
        retorno calculado entre esos dos puntos NO es un retorno de un
        día — es la suma de los retornos no consecutivos.

            Día	Régimen	Valor cartera
            L	calma	10000
            M	calma	10100
            X	crisis	9800
            J	calma	9900
            V	calma	10050
            L	crisis	9700
            M	crisis	9500
            X	calma	9600
            J	calma	9650
            V	calma	9700
            El problema al filtrar "solo días de calma"
            Cuando el algoritmo "calcula el Sharpe del PPO durante regímenes 
            de calma", filtra solo esos días y construimos una nueva curva:

            Día (etiqueta original)	Valor
            L	10000
            M	10100
            J	9900
            V	10050
            X	9600
            J	9650
            V	9700
            Ahora calculamos los retornos diarios de esa serie. Pero del viernes (10050) 
            al "siguiente" viernes con valor 9600 hay una caída del -4.5 %. ¿Es un 
            retorno diario? No. Es la suma del lunes (-4 %), martes (-2 %) y miércoles
            (+1 %) que no eran de calma y se filtraron.
            Esa "caída del -4.5 %" se mete en el cálculo de la volatilidad como si 
            fuera un día normal. Eso infla artificialmente la volatilidad de la calma 
            (porque ese movimiento real no era de calma).

            Por qué lo aceptamos:

            No hay alternativa limpia: la única forma rigurosa sería re-simular cada 
            estrategia "como si solo existieran los días de calma" — imposible con datos 
            reales.
            El sesgo es CONSISTENTE entre estrategias: tanto el PPO como Equal Weight y 
            todas las baselines sufren el mismo efecto. Eso significa que cuando 
            comparamos "Sharpe del PPO en calma vs Sharpe de Equal Weight en calma", 
            los dos números están sesgados de la misma forma → la diferencia entre 
            ellos sigue siendo válida, que es lo que importa para la conclusión del TFM.


            PResumen: es comparar dos coches midiendo el consumo solo en 
            autopista. Saltarse los tramos de ciudad introduce un error en el cálculo 
            absoluto, pero si ambos coches se miden con el mismo método imperfecto, la 
            comparación entre ellos es justa. Lo mismo aquí.


        En la práctica esto sesga las métricas levemente, pero es la
        mejor aproximación posible sin modelar transiciones explícitas
        entre regímenes. Lo importante es que el sesgo es CONSISTENTE
        entre estrategias (todas se evalúan igual), por lo que las
        COMPARACIONES siguen siendo válidas — que es lo que de verdad
        importa para la conclusión del TFM.

    Mínimo de 10 días por régimen:
        Con menos de 10 días el Sharpe es estadísticamente ruidoso. En
        lugar de devolver un número engañoso, omitimos el régimen — eso
        deja claro al lector que no había datos suficientes y evita
        afirmaciones débiles en la memoria.

    Parameters
    ----------
    value_series : pd.Series
        Curva de equity diaria de la estrategia, indexada por fecha.
    regimes : pd.Series
        Régimen (0, 1, 2) por día. Mismo índice o intersectable.
    annual_rf : float
        Tasa libre de riesgo anualizada (4% por defecto, alineado con el
        resto del pipeline TFM).

    Returns
    -------
    pd.DataFrame
        Filas = nombres de regímenes (Baja Vol., Vol. Media, Alta Vol.).
        Columnas = N días + métricas (Retorno, Vol, Sharpe, Sortino, MDD).
        Vacío si ningún régimen llegó al mínimo de 10 días.
    """
    labels = {0: 'Baja Volatilidad', 1: 'Vol. Media', 2: 'Alta Volatilidad'}
    rows   = {}

    # Alinear los dos índices por intersección
    common_index = value_series.index.intersection(regimes.index)
    aligned_values  = value_series.loc[common_index]
    aligned_regimes = regimes.loc[common_index]

    for code, name in labels.items():
        mask   = aligned_regimes == code
        n_days = mask.sum()

        if n_days < 10:
            # Insuficientes días para calcular métricas robustas
            continue

        subset = aligned_values[mask].reset_index(drop=True)

        # Reconstruir serie continua para métricas correctas
        returns = subset.pct_change().dropna()

        if len(returns) < 2:
            continue

        total_return = (subset.iloc[-1] / subset.iloc[0]) - 1
        annual_vol = returns.std() * np.sqrt(252)
        daily_rf = annual_rf / 252
        excess = returns - daily_rf
        sharpe = (excess.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

        neg_returns  = returns[returns < daily_rf]
        downside_std = (neg_returns.std() * np.sqrt(252)
                        if len(neg_returns) > 1 else 1e-8)
        sortino = (excess.mean() * 252) / downside_std

        rolling_max  = subset.cummax()
        max_dd = ((subset - rolling_max) / (rolling_max + 1e-8)).min()

        rows[name] = {
            'N días': int(n_days),
            'Retorno Total (%)': round(total_return * 100, 2),
            'Volatilidad Anualizada (%)':  round(annual_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Max Drawdown (%)': round(max_dd * 100, 2),
        }

    return pd.DataFrame(rows).T if rows else pd.DataFrame()



# Análisis completo: agente + baselines por régimen
def analyze_regimes(features_path: str = 'data/normalized_features.csv',
                    prices_path: str= 'data/original_prices.csv',
                    model_path: str  = None,
                    split_pct: float = 0.8,
                    initial_balance: float = 10000) -> dict:
    """
    Punto de entrada del módulo: orquesta todo el análisis por régimen.

    Es la función "una llamada → resultado completo". Se invoca desde el
    dashboard Streamlit (app_dashboard.py) y desde results_viewer.py.

    Pipeline interno (5 pasos):
      1. Cargar features y precios. Clasificar régimen sobre todo el
         dataset usando umbrales del periodo de train.
      2. Ejecutar todas las baselines (Equal Weight, Buy & Hold, 60/40,
         Markowitz) sobre el periodo de test.
         Resumen del valor de cada baseline:

            Baseline	Filosofía	Si PPO la bate, demuestra...
            Equal Weight	Diversificación ingenua	"Mi modelo aprende algo más allá de lo trivial"
            Buy & Hold	Pasividad total	"La gestión activa compensa sus costes"
            60/40	Diversificación clásica	"Mi modelo aporta sobre la receta clásica"
            Markowitz	Optimización teórica	"DRL aporta sobre la teoría académica establecida"
            Batir a las cuatro = el TFM tiene contenido defendible. Si solo bate a una o dos, hay qe 
                matizar la conclusión.

      3. Si hay modelo PPO disponible, ejecutarlo también.
         Si hay agente especulativo (.pkl), ejecutarlo también.
      4. Para cada estrategia, calcular métricas POR RÉGIMEN
         (vía metrics_by_regime).
      5. Generar PNG con curvas de equity y régimen sombreado + CSV con
         las tablas de métricas. Listos para insertar en la memoria.

    Por qué este nivel de granularidad responde una pregunta del tribunal:
        "TU PPO da Sharpe 1.78 en test. ¿Pero qué pasa cuando hay
         crisis? ¿Sigue funcionando o solo gana en calma?"
        Sin este módulo, no se puede contestar con datos. Con él, sí:
        las tablas regimen_metrics.csv responden literalmente.
        
            Sin regime_analysis.py, respuesta sería: "Pues... no lo sé exactamente, no 
            lo medí desglosado por régimen". Mala respuesta: deja  duda flotando.

            Con regime_analysis.py, respuesta es:"Buena pregunta. Lo medimos. Aquí 
            tenéis la tabla regime_metrics.csv. En calma, el PPO da Sharpe 2.1, supera 
            a Equal Weight por 0.3. En transición, da 1.5, supera por 0.4. En crisis, 
            da 0.8, supera por 1.1 — el margen máximo. Es decir: nuestro modelo aporta 
            sobre todo en momentos turbulentos, exactamente cuando la gestión activa 
            debe justificar su existencia frente a estrategias pasivas." La pregunta queda respondida con datos, 
            no con palabras vagas.


    Parameters
    ----------
    features_path, prices_path : str
        Rutas a los CSVs del pipeline.
    model_path : str, optional
        Ruta al modelo PPO entrenado (.zip). Si es None, se omite el
        análisis del PPO y solo se analiza distribución de regímenes +
        baselines + (si existe) agente especulativo. Útil para ejecutar
        el módulo cuando el modelo aún no está entrenado.
    split_pct : float
        Fracción de datos usada como train (define el inicio del test
        y los umbrales de los percentiles). Default 0.8 alineado con
        el resto del pipeline TFM.
    initial_balance : float
        Capital inicial estándar ($10 000), alineado con PortfolioEnv.

    Returns
    -------
    dict
        Estructura con:
          - 'regimenes': pd.Series con la etiqueta por día de test.
          - '{strategy_name}': pd.Series con curva de equity (una por estrategia).
          - 'metricas_{strategy_name}': pd.DataFrame de métricas por régimen.
        Las claves 'metricas_*' son las que después _save_metrics() exporta a CSV.
    """
    # Borrar reportes anteriores para evitar mostrar datos incoherentes
    for old_file in ['src/reports/regime_analysis.png',
                     'src/reports/regime_metrics.csv']:
        if os.path.exists(old_file):
            os.remove(old_file)

    # Carga de datos
    df_features = pd.read_csv(features_path, index_col=0)
    df_prices   = pd.read_csv(prices_path,   index_col=0)
    df_features.index = pd.to_datetime(df_features.index)
    df_prices.index   = pd.to_datetime(df_prices.index)

    split_idx = int(len(df_features) * split_pct)

    # Clasificación de regímenes sobre todo el dataset (umbrales del train)
    full_regimes = classify_regimes(df_prices, train_threshold_pct=split_pct)
    test_regimes = full_regimes.iloc[split_idx:]
    test_prices  = df_prices.iloc[split_idx:]

    print(f"\nDistribución de regímenes en el período de test ({len(test_prices)} días):")
    distribution = test_regimes.value_counts().sort_index()
    for code, n in distribution.items():
        name = {0: 'Baja Vol.', 1: 'Vol. Media', 2: 'Alta Vol.'}[code]
        print(f"  Régimen {code} ({name}): {n} días ({n/len(test_regimes)*100:.1f}%)")

    results = {'regimenes': test_regimes}

    # ─── Todas las estrategias ──────────────────────────────────────────────
    try:
        from src.benchmarking.baselines import run_baselines
    except ImportError:
        from src.benchmarking.baselines import run_baselines

    # Ejecutar las 4 baselines
    baselines = run_baselines(test_prices, initial_balance=initial_balance)
    for name, series in baselines.items():
        if series is not None and len(series) > 1:
            results[name] = series

    # Agente DRL (solo si se proporciona el modelo)
    if model_path and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model     = PPO.load(model_path)
            ia_values = _run_agent(model, features_path, prices_path, split_idx, None)
            series_ia = pd.Series(
                ia_values[:len(test_prices)],
                index=test_prices.index,
                name='IA_PPO'
            )
            results['IA_PPO'] = series_ia
        except Exception as e:
            print(f"  [AVISO] No se pudo ejecutar el agente PPO: {e}")

    # Agente especulativo (si existe)
    spec_path = 'models/speculative_gmm.pkl'
    if os.path.exists(spec_path):
        try:
            import pickle
            with open(spec_path, 'rb') as f:
                spec_agent = pickle.load(f)
            df_f_test = df_features.iloc[split_idx:]
            spec_series = spec_agent.backtest(df_f_test, test_prices,
                                              initial_balance=initial_balance)
            results['Especulativo_HMM'] = spec_series
        except Exception as e:
            print(f"  [AVISO] No se pudo ejecutar el especulativo: {e}")

    # Calcular métricas por régimen para cada estrategia
    strategy_names = [k for k in results.keys() if k != 'regimenes']
    for name in strategy_names:
        series = results[name]
        print(f"\nMétricas por régimen — {name}:")
        df_metrics = metrics_by_regime(series, test_regimes)
        if not df_metrics.empty:
            print(df_metrics.to_string())
            results[f'metricas_{name}'] = df_metrics
    
    # Visualización
    _plot_regimes(results, test_regimes, test_prices)
    _save_metrics(results)

    return results


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

# Paleta de colores tipo "semáforo" para los regímenes (verde/ámbar/rojo).
# Es la convención visual estándar en finanzas y se entiende de un vistazo
# sin necesidad de leer la leyenda — útil al insertar el PNG en la memoria.
_REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
# Nombres técnicos (los del eje del gráfico superior y de las métricas).
# El gráfico inferior usa la versión divulgativa "Calma/Transición/Crisis".
_REGIME_NAMES  = {0: 'Baja Volatilidad', 1: 'Volatilidad Media', 2: 'Alta Volatilidad'}


def _shade_backgrounds(ax, regimes: pd.Series) -> None:
    """
    Sombrea el fondo de un gráfico con bandas de color según el régimen.

    Es lo que convierte la gráfica en autoexplicativa: el lector ve la
    curva del PPO atravesando zonas verdes (calma), naranjas (transición)
    y rojas (crisis), y entiende a primera vista en qué condiciones se
    desempeña mejor cada estrategia.

    Implementación: detectamos bloques continuos del mismo régimen
    (`regimes.ne(regimes.shift()).cumsum()`) y pintamos un axvspan por
    bloque. Más eficiente y limpio que un span por día (miles de spans).
    """
    if regimes.empty:
        return

    # Detectar bloques continuos del mismo régimen
    changes = regimes.ne(regimes.shift()).cumsum()
    for _, block in regimes.groupby(changes):
        if block.empty:
            continue
        code  = int(block.iloc[0])
        start = block.index[0]
        end   = block.index[-1]
        ax.axvspan(start, end,
                   alpha=0.15,
                   color=_REGIME_COLORS.get(code, 'gray'),
                   linewidth=0)


def _plot_regimes(results: dict, test_regimes: pd.Series,
                  test_prices: pd.DataFrame) -> None:
    """
    Genera la gráfica principal del análisis (PNG para la memoria del TFM).

    Layout en dos paneles, alineados temporalmente:
      - Superior (3/4 de altura): curvas de equity de TODAS las estrategias
        + bandas tipo semáforo de régimen al fondo. El lector ve qué hace
        cada estrategia en cada zona de mercado simultáneamente.
      - Inferior (1/4 de altura): indicador "barra de regímenes" con
        etiquetas divulgativas (Calma/Transición/Crisis) — sirve de
        leyenda visual del panel superior.

    Robustez frente a curvas con índice no-DatetimeIndex:
        Algunas estrategias (PPO, especulativo) pueden devolver Series con
        RangeIndex numérico en lugar de fechas. Si las pintásemos así, el
        eje X tendría números (0, 1, 2...) en vez de fechas legibles. Las
        re-indexamos con las fechas del test antes de pintar.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax1, ax2 = axes

    # -- Panel superior: evolución de carteras (todas las estrategias) --
    strategy_styles = {
        'IA_PPO':               ('#00d4ff', '-',  2.5, 'IA PPO (DRL)'),
        'Equal_Weight_Mensual': ('#f0a500', '--', 1.5, 'Equal Weight'),
        'Buy_and_Hold':         ('#7ed957', '--', 1.5, 'Buy & Hold'),
        'Cartera_60_40':        ('#ff6b6b', '--', 1.5, 'Cartera 60/40'),
        'Markowitz_MV':         ('#c77dff', ':',  1.5, 'Markowitz MV'),
        'Especulativo_HMM':     ('#ff9f1c', '-.',  1.5, 'Especulativo (GMM)'),
    }

    for name, (color, style, width, label) in strategy_styles.items():
        if name in results and results[name] is not None:
            series = results[name]
            # Asegurar que el índice es DatetimeIndex — algunas estrategias
            # (Especulativo, PPO) pueden devolver RangeIndex numérico
            if not isinstance(series.index, pd.DatetimeIndex):
                # Alinear con las fechas del test, truncando si es necesario
                series = pd.Series(
                    series.values[:len(test_prices)],
                    index=test_prices.index[:len(series)],
                    name=name,
                )
                results[name] = series  # actualizar para metrics_by_regime
            ax1.plot(series.index, series.values,
                     label=label, color=color, linestyle=style, linewidth=width)

    # Fondo sombreado por régimen
    _shade_backgrounds(ax1, test_regimes)

    # Leyenda combinada: estrategias + regímenes
    regime_patches = [
        mpatches.Patch(color=_REGIME_COLORS[k], alpha=0.4, label=_REGIME_NAMES[k])
        for k in sorted(_REGIME_COLORS.keys())
    ]
    all_handles = ax1.get_legend_handles_labels()[0] + regime_patches
    all_labels  = ax1.get_legend_handles_labels()[1] + [p.get_label() for p in regime_patches]
    ax1.legend(handles=all_handles, labels=all_labels,
               loc='upper left', fontsize=8, ncol=2)

    ax1.set_title(
        'Análisis de Regímenes de Volatilidad — Período Out-of-Sample\n'
        'Referencia: IVV (S&P 500). Zonas coloreadas = régimen de volatilidad del mercado.',
        fontsize=12, fontweight='bold')
    ax1.set_ylabel('Valor de Cartera ($)')
    ax1.grid(True, alpha=0.3)

    # -- Panel inferior: indicador de régimen con explicación --
    regime_bar_colors = [_REGIME_COLORS.get(int(v), 'gray') for v in test_regimes.values]
    ax2.bar(test_regimes.index, test_regimes.values,
            color=regime_bar_colors, width=1, alpha=0.8)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Calma', 'Transición', 'Crisis'], fontsize=9)
    ax2.set_title(
        'Régimen del mercado (basado en volatilidad rolling 20d del IVV vs percentiles del período de train)',
        fontsize=10)
    ax2.set_ylabel('Régimen')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = 'src/reports/regime_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGráfica de regímenes guardada: {output_path}")
    plt.close(fig)


def _save_metrics(results: dict) -> None:
    """
    Persiste las tablas de métricas por régimen en CSV.

    Concatena todas las tablas `metricas_*` (una por estrategia) en un
    único CSV `src/reports/regime_metrics.csv`, añadiendo una columna
    'Estrategia' al inicio para distinguirlas. Listo para abrir en Excel
    o pegar en la memoria del TFM como tabla comparativa.

    encoding='utf-8-sig' añade BOM al inicio para que Excel abra
    correctamente los caracteres acentuados (sin BOM, "Volatilidad"
    aparecería como "VolatilidadÃ" en Windows).
    """
    output_path = 'src/reports/regime_metrics.csv'
    tables = []

    for key, val in results.items():
        if key.startswith('metricas_') and isinstance(val, pd.DataFrame) and not val.empty:
            strategy_name = key.replace('metricas_', '')
            df = val.copy()
            df.insert(0, 'Estrategia', strategy_name)
            tables.append(df)

    if tables:
        pd.concat(tables).to_csv(output_path, encoding='utf-8-sig')
        print(f"Métricas por régimen guardadas: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Aliases de retrocompatibilidad
# ─────────────────────────────────────────────────────────────────────────────
# Antes de la refactorización a inglés, las funciones se llamaban en
# español. results_viewer.py todavía importa `analizar_regimenes`. Los
# aliases mantienen el código antiguo funcionando sin tocar nada externo.
# Eliminar cuando se confirme que ningún consumidor los usa.
clasificar_regimenes = classify_regimes
metricas_por_regimen = metrics_by_regime
analizar_regimenes = analyze_regimes
