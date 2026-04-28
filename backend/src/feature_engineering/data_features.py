"""
Módulo de ingeniería de características financieras.

Es la fase 2 del pipeline TFM: recibe los datos crudos OHLCV de cada activo
(que vienen de data_downloader.py) y genera las features que verá el agente
PPO en cada paso. Cada función recibe DataFrames y devuelve DataFrames
transformados, sin efectos colaterales.

Características generadas por activo:
  - Retornos logarítmicos (garantizan aditividad temporal): ln(precio_hoy / precio_ayer).
        Los retornos logarítmicos suman entre sí en lugar de multiplicarse (los
        simples requieren productos de (1+r)), lo que estabiliza numéricamente
        las redes neuronales y evita los sesgos de las escalas porcentuales.
  - Momentum a multiples horizontes (5d, 20d, 60d): Cuánto ha subido o bajado el precio en los últimos 5, 20 
        y 60 días de trading
        5 días (1 semana): ruido de corto plazo — ¿hay presión compradora/vendedora inmediata?
        20 días (1 mes): señal de medio plazo — ¿el activo está en tendencia alcista o bajista?
        60 días (1 trimestre): tendencia estructural — ¿el mercado ha cambiado de dirección?
        Si los tres son positivos simultáneamente, el activo tiene momentum fuerte en todas las escalas 
        temporales. Si el de 5 días es negativo pero el de 60 es positivo, puede ser un retroceso temporal 
        dentro de una tendencia alcista — oportunidad de compra para el agente.

  - Volatilidad, asimetria y curtosis rolling (captura de colas pesadas y memoria larga):
        Volatilidad: la oscilación, la dispersión del movimiento. Subidas sostenidas del 1% todos los días 
            serán volatilidad 0. Uno que alterna +3% y -3% tiene volatilidad  alta aunque su media sea 0%.
        Asimetría - Skew negativo (ej. -1.5): la mayoría de días son ligeramente positivos, pero de vez en 
            cuando hay una caída grande. Es el patrón típico de las acciones — suben lento y caen rápido 
            (efecto "escalera arriba, ascensor abajo").
        Asimetría - Skew positivo (ej. +2.0): la mayoría de días son ligeramente negativos, pero de vez en
            cuando hay un rally explosivo. Patrón típico de opciones o criptomonedas. El agente puede aprender 
            que un activo con skew negativo creciente está acumulando riesgo de caída brusca.         
        Detector de colas pesadas - Curtosis ≈ 3: distribución normal — las sorpresas extremas son muy raras
        Detector de colas pesadas - Curtosis >> 3 (ej. 10, 20): días de ±5% ocurren con mucha más frecuencia de lo 
        que una distribución normal predice. Es el mundo real de los mercados financieros.
        La "memoria larga" se refiere a que usar ventanas amplias (60 días) captura si este patrón de colas pesadas
        es reciente o lleva meses instalado.
                    
  - Drawdown desde el maximo de la ventana larga (proxy de riesgo reciente): Cuánto ha caído el precio respecto a 
        su máximo de los últimos 60 días. 
        Siempre es negativo o cero:
            0%: el precio está en máximos de la ventana
            -15%: ha caído un 15% desde su pico reciente
        Es un proxy directo de riesgo: un activo que lleva -20% de drawdown puede     seguir cayendo (tendencia bajista) o estar sobrevendido (oportunidad). 
        El agente aprende a distinguir ambos casos combinando el drawdown con las otras features.

  - RSI, MACD diff, ATR, CCI, Bollinger Band Width (indicadores tecnicos clasicos)

        RSI (Relative Strength Index) — oscilador entre 0 y 100. 
            Mide la proporción de días alcistas vs bajistas recientes:
            RSI > 70: "sobrecomprado" — el precio ha subido mucho, hay presión de venta latente
            RSI < 30: "sobrevendido" — el precio ha caído mucho, posible rebote
            Documentación: Es la señal más usada en trading técnico desde los años 70 (Wilder, 1978)
        MACD diff (Moving Average Convergence Divergence) — 
            Diferencia entre dos medias móviles exponenciales (rápida de 12 días, 
            lenta de 26 días). Cuando la rápida cruza por encima de la lenta, es señal 
            alcista; cuando cruza por debajo, bajista. El "diff" es la distancia entre 
            la línea MACD y su propia señal (media de 9 días) — captura la aceleración 
            de la tendencia.
            Para tontos: la EMA es como la opinión sobre el clima. Si lleva 30 días de sol y mañana llueve, 
                no decimos "bah, da igual, la media es soleado". Le das más peso al cambio reciente. Eso es 
                exactamente lo que hace la EMA.
                El MACD — la diferencia entre dos EMAs. El MACD es una sola operación: restar dos EMAs distintas, 
                una más rápida (sensible) y otra más lenta (estable):
                MACD = EMA_12_dias  −  EMA_26_dias
                            ↑              ↑
                        rápida         lenta
                    (reacciona     (reacciona
                        rápido)        despacio)
                    Qué significa el resultado
                    MACD positivo → la EMA rápida está por encima de la lenta → el precio reciente es mayor que el promedio de 
                        largo plazo → tendencia alcista ("el precio está subiendo recientemente más rápido de lo 
                        normal").
                    MACD negativo → la rápida por debajo de la lenta → tendencia bajista.
                    MACD cruzando de negativo a positivo = "señal alcista" (Golden cross). El momentum cambia.
                    MACD cruzando de positivo a negativo = "señal bajista" (Death cross).

                Qué dice el diff
                Si el MACD está por encima de su línea de señal → la tendencia se está acelerando en la dirección 
                    actual. Si era alcista, está cogiendo más fuerza.
                Si el MACD está por debajo de su señal → la tendencia se está desacelerando. Si era alcista, 
                    empieza a perder fuerza (posible reversión).
                Resumen: el MACD te dice "vas hacia arriba o hacia abajo". El diff te dice "estás acelerando o 
                    frenando en esa dirección".

        ATR (Average True Range) — rango medio real. 
            No mide dirección, solo cuánto se mueve el precio en un día típico 
            (incluyendo gaps entre cierre y apertura). Un ATR alto significa que el 
            activo se mueve mucho intradía — útil para calibrar stops y dimensionar 
            posiciones.

        CCI (Commodity Channel Index) — mide cuánto se desvía el precio de su media 
            estadística. Valores extremos (>+100 o <-100) indican que el precio está 
            lejos de lo "normal" para su rango reciente — posible reversión a la media.

        Bollinger Band Width — anchura de las Bandas de Bollinger (media ± 2 
            desviaciones estándar). Mide la compresión/expansión de la volatilidad. 
            Cuando las bandas se estrechan mucho (BBW bajo), históricamente precede a 
            un movimiento brusco en cualquier dirección — es una señal de que "algo va 
            a pasar", aunque no dice en qué dirección.


Caracteristicas transversales (cross-asset):
  - Correlaciones dinamicas entre pares de activos a multiples ventanas
  - Indicadores binarios de regimen de alta volatilidad por activo
"""
import numpy as np
import pandas as pd
from ta import add_all_ta_features


# ─────────────────────────────────────────────
# Constantes por defecto
# ─────────────────────────────────────────────
# Ventanas de calculo: corta (ruido), media (senal), larga (tendencia)
DEFAULT_WINDOWS = [5, 20, 60]

# Pares de correlacion: cripto vs tradicional + correlacion clasica acciones/bonos.
# Tres pares con interpretación financiera explícita — evita ruido de correlacionar
# todos los activos contra todos.
DEFAULT_CORRELATION_PAIRS = [
    ('IBIT', 'IVV'),  # Bitcoin vs Renta Variable — activo de riesgo o refugio?
    ('IBIT', 'BND'),  # Bitcoin vs Bonos — diversificador o activo especulativo?
    ('IVV',  'BND'),  # Renta Variable vs Bonos — correlacion clasica de cartera
]

# TODO (trabajo futuro): añadir ('ETHA', 'IBIT') y ('ETHA', 'IVV') a
# DEFAULT_CORRELATION_PAIRS para que el agente reciba señal de correlación
# dinámica de Ethereum.
#
# Justificación: el alcance del TFM ("Integrando Criptoactivos en la Inversión
# Tradicional") incluye dos cripto-ETFs (IBIT y ETHA) precisamente para
# diferenciarlos como clases de activo distintas, pero ETHA no aparece en
# ningún par actual y el agente no puede aprender:
#   - La diferencia entre "Bitcoin como reserva de valor" y "Ethereum como
#     activo tecnológico/DeFi".
#   - Cuándo ETHA e IBIT divergen (información clave para diversificación
#     intra-cripto).
#   - El comportamiento de ETHA frente a renta variable y bonos.
#
# Implicación: añadir los pares requiere reentrenar el modelo (cambia el
# espacio de observación). Documentado como limitación conocida en la
# memoria; por ahora se mantienen los tres pares originales para no romper
# la reproducibilidad del modelo final entregado.
# Si la correlación cambia, puede estar bien (0.7 en bonanza, 0.95 en crisis), demuestro que diferenciar 
# entre criptos aporta valor real al modelo. Es el argumento que justifica por qué tengo
# 2 cripto-ETFs y no solo 1.
# 
# Importantes. Validar Ruben!
# ('ETHA', 'IVV'),   # Ethereum vs Renta Variable — perfil idéntico al de IBIT-IVV
# Por qué: paraleliza el análisis que hago con IBIT. Sin este par, el modelo "ve" 
# cómo Bitcoin se relaciona con la bolsa pero no Ethereum — inconsistente con TFM.

# Posibles, pero weno, no relevantes
# ('ETHA', 'BND'),   # Ethereum vs Bonos
# Por qué dudo: misma debilidad que IBIT-BND — la dinámica histórica cripto-bonos es 
# ruido en gran medida?? Lo añadiría x simetría con el análisis de IBIT.
#
# Más features?? no siempre es mejor:

#DEFAULT_CORRELATION_PAIRS = [
#    # Bloque "cripto vs tradicional" — ¿son las criptos activos de riesgo o refugio?
#    ('IBIT', 'IVV'),  # Bitcoin vs Renta Variable
#    ('ETHA', 'IVV'),  # Ethereum vs Renta Variable
#    # Bloque "cripto vs renta fija" — ¿descorrelacionan con bonos?
#    ('IBIT', 'BND'),
#    # Bloque "intra-cripto" — ¿BTC y ETH son la misma cosa o se diferencian?
#    ('ETHA', 'IBIT'),  # CLAVE para defender la elección de 2 cripto-ETFs
#    # Cartera clásica
#    ('IVV',  'BND'),  # 60/40 — ¿sigue funcionando?
#]

#Justificación para la memoria del TFM: "Se seleccionaron 5 pares de correlación dinámica 
# que cubren tres ejes analíticos: cripto-vs-tradicional, intra-cripto, y la cartera 
# clásica 60/40. Esta selección permite al agente detectar cambios de régimen en cada uno
# de los pilares de la tesis (criptoactivos como diversificadores, diferenciación entre 
# Bitcoin y Ethereum, robustez de la 60/40 ante mercados modernos)."

#Riesgo a vigilar. RUben help
# ETHA solo existe desde julio 2024. Para backtests sobre 2018-2024, las correlaciones 
# que involucren ETHA serán NaN la mayor parte del periodo. El pipeline ya contempla eso 
# (limpieza columnar en generate_dataset), pero conviene saber que:
# Si entreno con dataset largo (ej. 2018-2026), las features de ETHA tendrán datos 
# válidos solo en los últimos 2 años.
# Si la ventana rolling de 60d empieza antes de julio 2024, generará NaN durante los 
# primeros 60d posteriores al listado.
# mencionarlo en la memoria como "limitación de datos" cuando explique esas correlaciones.



# ─────────────────────────────────────────────
# Features por activo individual
# ─────────────────────────────────────────────

def compute_price_features(df_ohlcv: pd.DataFrame, ticker: str,
                           windows: list = None) -> pd.DataFrame:
    """
    Calcula el conjunto completo de features estadisticas y tecnicas para un activo.
    Esta función son los "ojos" de mi modelo.

    Genera retornos logaritmicos, momentum a multiples horizontes, estadisticos
    de orden superior (volatilidad, asimetria, curtosis), drawdown, indicadores
    tecnicos clasicos (RSI, MACD, ATR, CCI, Bollinger, MFI), precio relativo
    a medias moviles, volumen relativo e iliquidez de Amihud.

    Parameters
    ----------
    df_ohlcv : pd.DataFrame
        DataFrame con columnas Open, High, Low, Close, Volume.
    ticker : str
        Nombre del activo; se usa como prefijo en las columnas generadas.
    windows : list, optional
        Lista de horizontes temporales (en dias) para calculos rolling.
        Por defecto DEFAULT_WINDOWS = [5, 20, 60].

    Returns
    -------
    pd.DataFrame
        DataFrame con todas las features del activo, mismo indice que df_ohlcv.        
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    df_feat = pd.DataFrame(index=df_ohlcv.index)

    # Retornos logaritmicos 
    # ln(P_t / P_{t-1}): garantizan aditividad temporal y manejan asimetrias
    # evitamos sesgos de escalas porcentuales
    returns = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1))
    df_feat[f'{ticker}_retornos'] = returns

    # Momentum a multiples horizontes 
    # Retorno acumulado en cada ventana: captura tendencias de corto, medio y largo plazo
    for w in windows:
        df_feat[f'{ticker}_momentum_{w}d'] = df_ohlcv['Close'].pct_change(w)

    # Estadisticos de orden superior (memoria larga y colas pesadas) 
    # Volatilidad: mide la dispersion del riesgo en cada ventana
    # Asimetria: indica si los retornos tienden mas a caidas que a subidas
    # Curtosis: captura la probabilidad de eventos extremos (cisnes negros)
    for w in windows:
        df_feat[f'{ticker}_vol_{w}d']  = returns.rolling(w).std()
        df_feat[f'{ticker}_skew_{w}d'] = returns.rolling(w).skew()
        df_feat[f'{ticker}_kurt_{w}d'] = returns.rolling(w).kurt()

    # ── Drawdown desde maximo de la ventana larga ──────────────────────────────
    # identificamos riesgo reciente: que tan lejos esta el precio de su maximo local
    long_window = max(windows)
    rolling_max = df_ohlcv['Close'].rolling(long_window).max()
    df_feat[f'{ticker}_drawdown_{long_window}d'] = (
        (df_ohlcv['Close'] - rolling_max) / (rolling_max + 1e-8)
    )

    # ── Indicadores tecnicos via libreria 'ta' ─────────────────────────────────
    df_ta = add_all_ta_features(
        df_ohlcv.copy(),
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )

    # Mapa: {nombre_destino -> nombre_en_ta}
    ta_indicators = {
        f'{ticker}_rsi': 'momentum_rsi',    # Oscilador sobrecompra/sobreventa
        f'{ticker}_macd_diff': 'trend_macd_diff', # Senal de cambio de tendencia
        f'{ticker}_atr': 'volatility_atr',  # Volatilidad basada en rangos (True Range)
        f'{ticker}_cci': 'trend_cci',        # Ciclos de precio vs media
        f'{ticker}_bb_width': 'volatility_bbw',  # Anchura de Bandas de Bollinger
        f'{ticker}_mfi': 'volume_mfi',      # Money Flow Index: RSI ponderado por volumen
    }
    for col_dest, col_orig in ta_indicators.items():
        if col_orig in df_ta.columns:
            df_feat[col_dest] = df_ta[col_orig].values

    # Precio relativo a medias moviles 
    # Senales de tendencia: precio por encima de MA -> momentum alcista
    # MA50 captura tendencia de medio plazo; MA200 captura tendencia estructural
    # La ventana se adapta al tamano del dataset para evitar columnas enteramente NaN
    # (con < 200 dias de datos, rolling(200) seria todo NaN -> dropna vaciaria el CSV)
    n_rows = len(df_ohlcv)
    for ma in [50, 200]:
        ma_eff = min(ma, max(2, n_rows // 2))
        ma_val = df_ohlcv['Close'].rolling(ma_eff).mean()
        df_feat[f'{ticker}_precio_vs_ma{ma}'] = df_ohlcv['Close'] / (ma_val + 1e-8) - 1

    # Volumen relativo (volume surge) 
    # Ratio entre el volumen actual y la media de 20 dias.
    # Valores > 1.5 indican interes institucional inusual y confirman movimientos de precio.
    vol_mean_20 = df_ohlcv['Volume'].rolling(20).mean()
    df_feat[f'{ticker}_vol_relativo'] = df_ohlcv['Volume'] / (vol_mean_20 + 1e-8)

    # ── Iliquidez de Amihud ────────────────────────────────────────────────────
    # |retorno diario| / volumen en dólares. Mide el impacto sobre el precio
    # por cada dólar negociado (Amihud, 2002). Alta iliquidez → el precio
    # "salta" con poco dinero → mayor slippage al ejecutar órdenes grandes.
    # Aviso para el agente: entrar/salir con mucho capital en este activo
    # erosionará el alpha esperado por costes de ejecución.
    dollar_volume = df_ohlcv['Volume'] * df_ohlcv['Close']
    df_feat[f'{ticker}_amihud'] = returns.abs() / (dollar_volume + 1e-8)

    return df_feat


# ─────────────────────────────────────────────
# Features transversales (cross-asset)
# ─────────────────────────────────────────────

def compute_dynamic_correlations(dataset: pd.DataFrame,
                                 pairs: list = None,
                                 windows: list = None) -> pd.DataFrame:
    """
    Calcula correlaciones moviles entre pares de activos sobre retornos logaritmicos.
    Cerebro estratégico. 

    Captura la sinergia dinamica entre activos:
      - Cuando corr(IBIT, IVV) sube, Bitcoin se comporta como activo de riesgo.
      - Cuando corr(IBIT, IVV) baja, Bitcoin actua como diversificador.
    El agente puede aprender a reducir exposicion a IBIT cuando la correlacion sube.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame que contiene columnas {ticker}_retornos para cada activo.
    pairs : list of tuple, optional
        Lista de tuplas (ticker_a, ticker_b) con los pares a correlacionar.
        Por defecto DEFAULT_CORRELATION_PAIRS.
    windows : list of int, optional
        Horizontes (en dias) para el calculo de correlacion rolling.
        Por defecto [20, 60].

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas corr_{A}_{B}_{v}d para cada par y ventana.
    """
    if pairs is None:
        pairs = DEFAULT_CORRELATION_PAIRS
    if windows is None:
        windows = [20, 60]

    df_corr = pd.DataFrame(index=dataset.index)

    for ticker_a, ticker_b in pairs:
        col_a = f'{ticker_a}_retornos'
        col_b = f'{ticker_b}_retornos'
        # Solo calcular si ambos activos estan presentes en el universo
        if col_a not in dataset.columns or col_b not in dataset.columns:
            continue
        n_rows = len(dataset)
        for w in windows:
            w_eff = min(w, max(5, n_rows // 3))
            col_name = f'corr_{ticker_a}_{ticker_b}_{w}d'
            df_corr[col_name] = dataset[col_a].rolling(w_eff).corr(dataset[col_b])

    return df_corr


def compute_rolling_beta(dataset: pd.DataFrame, tickers: list,
                         market_ticker: str = 'IVV',
                         window: int = 60) -> pd.DataFrame:
    """
    Calcula la beta rolling de cada activo respecto al mercado (IVV por defecto).
    Es el espejo del mercado, ¿cómo se va moviendo?
    Cerebro estratégico. 

    Beta = Cov(r_activo, r_mercado) / Var(r_mercado).
    Una beta > 1 indica que el activo amplifica los movimientos del mercado;
    beta < 0 indica cobertura natural (relevante para BND en crisis).
    El agente puede usar este indicador para ajustar la exposicion ciclica.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame con columnas {ticker}_retornos para cada activo.
    tickers : list of str
        Activos para los que calcular la beta.
    market_ticker : str, optional
        Activo de referencia que actua como proxy del mercado. Por defecto 'IVV'.
    window : int, optional
        Horizonte de calculo rolling en dias. Por defecto 60.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas {ticker}_beta_{window}d para cada activo
        (excluyendo el propio mercado).
    """
    df_beta = pd.DataFrame(index=dataset.index)
    market_col = f'{market_ticker}_retornos'

    if market_col not in dataset.columns:
        return df_beta

    # Ventana adaptativa: nunca superar 1/3 del dataset para evitar columnas enteramente NaN
    effective_window = min(window, max(10, len(dataset) // 3))

    market_returns = dataset[market_col]
    market_var = market_returns.rolling(effective_window).var()

    for ticker in tickers:
        if ticker == market_ticker:
            continue
        ret_col = f'{ticker}_retornos'
        if ret_col not in dataset.columns:
            continue
        cov = dataset[ret_col].rolling(effective_window).cov(market_returns)
        df_beta[f'{ticker}_beta_{window}d'] = cov / (market_var + 1e-8)

    return df_beta


def compute_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Genera features de calendario derivadas del indice temporal.
    Cerebro estratégico. 

    Capturan anomalias estacionales documentadas en finanzas:
      - Efecto lunes: retornos historicamente negativos los lunes.
      - Efecto enero: mayor rentabilidad en el primer mes del ano.
      - Efecto fin de trimestre: "window dressing" — los gestores ajustan
        carteras para que el reporte trimestral muestre las posiciones
        que querrían que sus clientes vieran (maquillaje contable).
      - Efecto fin de mes: flujos de rebalanceo sistematico.

    Al ser features deterministas (calculadas desde la fecha del calendario,
    no desde estadísticos de mercado), no introducen data leakage.
        Data leakage: "fuga de datos". El modelo ve durante el entrenamiento
            información que no debería conocer (ej. estadísticos calculados
            con datos del futuro). Infla métricas in-sample y hunde el
            rendimiento real en producción.
        Lookahead bias: caso particular de data leakage donde la fuga viene
            del eje temporal — el modelo "ve hacia el futuro".

    Parameters
    ----------
    index : pd.DatetimeIndex
        Indice temporal del dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame con features de calendario normalizadas al rango [0, 1].
    """
    df_cal = pd.DataFrame(index=index)

    # Dia de la semana normalizado: lunes=0, viernes=1
    df_cal['cal_dia_semana'] = index.dayofweek / 4.0

    # Mes del ano normalizado: enero=0, diciembre=1
    df_cal['cal_mes'] = (index.month - 1) / 11.0

    # Indicador binario de fin de mes (ultimos 3 dias habiles del mes)
    # Los fondos rebalancean sistematicamente en este periodo
    df_cal['cal_fin_mes'] = (index.day >= 28).astype(np.float32)

    # Indicador binario de fin de trimestre (marzo, junio, septiembre, diciembre)
    df_cal['cal_fin_trimestre'] = index.month.isin([3, 6, 9, 12]).astype(np.float32)

    return df_cal


def compute_volatility_regime(dataset: pd.DataFrame,
                              tickers: list,
                              short_window: int = 20,
                              long_window: int = 252,
                              threshold: float = 1.5) -> pd.DataFrame:
    """
    Indicador binario de regimen de alta volatilidad por activo.
    Cerebro estratégico. Detector de humo.

    Criterio: vol_rolling_{corta}d > umbral * vol_rolling_{larga}d
      - 1: regimen de alta volatilidad (correccion de mercado, crisis).
      - 0: regimen de calma o volatilidad normal.

    Este indicador permite al agente PPO identificar de forma proactiva los cambios
    de regimen y ajustar la exposicion a activos volatiles como IBIT antes de
    que se produzcan caidas bruscas, en lugar de reaccionar a posteriori.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame con columnas {ticker}_retornos y {ticker}_vol_{v}d.
    tickers : list of str
        Lista de activos para los que calcular el indicador.
    short_window : int, optional
        Ventana de volatilidad de corto plazo en dias. Por defecto 20.
    long_window : int, optional
        Ventana de volatilidad de largo plazo (aprox 1 ano). Por defecto 252.
    threshold : float, optional
        Factor multiplicativo; 1.5 significa 50%% por encima de la media historica.
        Por defecto 1.5.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas {ticker}_regimen_alta_vol (float32: 0.0 o 1.0).
    """
    df_reg = pd.DataFrame(index=dataset.index)

    short_vol_key = f'_vol_{short_window}d'

    # Ventana larga adaptativa: con datasets cortos, rolling(252) produce NaN en todo el periodo.
    # La comparacion NaN > X devuelve False en pandas -> regimen artificialmente siempre en 0.
    # Usamos como minimo short_window+1 y como maximo 1/3 del dataset.
    effective_long_window = min(long_window, max(short_window + 1, len(dataset) // 3))

    for ticker in tickers:
        ret_col     = f'{ticker}_retornos'
        short_vol_col = f'{ticker}{short_vol_key}'

        if ret_col not in dataset.columns:
            continue

        # Volatilidad de largo plazo calculada directamente sobre retornos
        long_vol = dataset[ret_col].rolling(effective_long_window).std()

        # Usar la volatilidad corta ya calculada si existe; si no, recalcular
        if short_vol_col in dataset.columns:
            short_vol = dataset[short_vol_col]
        else:
            short_vol = dataset[ret_col].rolling(short_window).std()

        df_reg[f'{ticker}_regimen_alta_vol'] = (
            short_vol > threshold * long_vol
        ).astype(np.float32)

    return df_reg


# ─────────────────────────────────────────────
# Normalizacion
# ─────────────────────────────────────────────

def normalize_zscore(df: pd.DataFrame,
                     split_pct: float = 0.8) -> pd.DataFrame:
    """
    Normalizacion Z-Score sin lookahead bias.
    Filtro de seguridad.

    Calcula mu y sigma SOLO sobre el conjunto de entrenamiento (primeros split_pct%%
    de filas) y aplica esas mismas estadisticas al conjunto de test.

    Por que importa:
      Si se normalizara sobre el dataset completo, el test contaminaria el train:
      el agente "veria" implicitamente la media y varianza de precios futuros.
      Esto infla las metricas in-sample y hunde el rendimiento out-of-sample,
      exactamente el patron observado cuando PPO es peor que las baselines.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo (train + test concatenados).
    split_pct : float, optional
        Fraccion de datos que pertenece a train. Por defecto 0.8.

    Returns
    -------
    pd.DataFrame
        DataFrame normalizado con media 0 y desviacion estandar 1
        (calculadas sobre la porcion de entrenamiento).
    """
    split_idx = int(len(df) * split_pct)
    df_train  = df.iloc[:split_idx]

    mu    = df_train.mean()
    sigma = df_train.std() + 1e-8

    return (df - mu) / sigma


# ─────────────────────────────────────────────
# Alias retrocompatibles (backward-compatible)
# ─────────────────────────────────────────────
# El pipeline existente importa los nombres originales en espanol.
# Estos alias garantizan que no se rompa ninguna importacion.

VENTANAS_DEFAULT = DEFAULT_WINDOWS
PARES_CORRELACION_DEFAULT = DEFAULT_CORRELATION_PAIRS

calcular_features_precio = compute_price_features
calcular_correlaciones_dinamicas = compute_dynamic_correlations
calcular_regimen_volatilidad = compute_volatility_regime
calcular_beta_rolling = compute_rolling_beta
calcular_features_calendario = compute_calendar_features
normalizar_zscore = normalize_zscore
