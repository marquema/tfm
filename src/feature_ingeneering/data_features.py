"""
Módulo de ingeniería de características financieras.

Contiene funciones puras: sin I/O, sin descarga, sin efectos secundarios.
Reciben DataFrames y devuelven DataFrames transformados.

Características generadas por activo:
  - Retornos logarítmicos (garantizan aditividad temporal)
  - Momentum a múltiples horizontes (5d, 20d, 60d)
  - Volatilidad, asimetría y curtosis rolling (captura de colas pesadas y memoria larga)
  - Drawdown desde el máximo de la ventana larga (proxy de riesgo reciente)
  - RSI, MACD diff, ATR, CCI, Bollinger Band Width (indicadores técnicos clásicos)

Características transversales (cross-asset):
  - Correlaciones dinámicas entre pares de activos a múltiples ventanas
  - Indicadores binarios de régimen de alta volatilidad por activo
"""
import numpy as np
import pandas as pd
from ta import add_all_ta_features


# ─────────────────────────────────────────────
# Constantes por defecto
# ─────────────────────────────────────────────

# Ventanas de cálculo: corta (ruido), media (señal), larga (tendencia)
VENTANAS_DEFAULT = [5, 20, 60]

# Pares de correlación: cripto vs tradicional + correlación clásica acciones/bonos
PARES_CORRELACION_DEFAULT = [
    ('IBIT', 'IVV'),  # Bitcoin vs Renta Variable — ¿activo de riesgo o refugio?
    ('IBIT', 'BND'),  # Bitcoin vs Bonos — ¿diversificador o activo especulativo?
    ('IVV',  'BND'),  # Renta Variable vs Bonos — correlación clásica de cartera
]


# ─────────────────────────────────────────────
# Features por activo individual
# ─────────────────────────────────────────────

def calcular_features_precio(df_ohlcv: pd.DataFrame, ticker: str,
                              ventanas: list = None) -> pd.DataFrame:
    """
    Calcula el conjunto completo de features estadísticas y técnicas para un activo.

    Parámetros
    ----------
    df_ohlcv : DataFrame con columnas Open, High, Low, Close, Volume
    ticker   : nombre del activo (prefijo para las columnas generadas)
    ventanas : lista de horizontes temporales para cálculos rolling

    Retorna
    -------
    DataFrame con todas las features del activo, mismo índice que df_ohlcv
    """
    if ventanas is None:
        ventanas = VENTANAS_DEFAULT

    df_feat = pd.DataFrame(index=df_ohlcv.index)

    # ── Retornos logarítmicos ──────────────────────────────────────────────────
    # ln(P_t / P_{t-1}): garantizan aditividad temporal y manejan asimetrías
    retornos = np.log(df_ohlcv['Close'] / df_ohlcv['Close'].shift(1))
    df_feat[f'{ticker}_retornos'] = retornos

    # ── Momentum a múltiples horizontes ───────────────────────────────────────
    # Retorno acumulado en cada ventana: captura tendencias de corto, medio y largo plazo
    for v in ventanas:
        df_feat[f'{ticker}_momentum_{v}d'] = df_ohlcv['Close'].pct_change(v)

    # ── Estadísticos de orden superior (memoria larga y colas pesadas) ─────────
    # Volatilidad: mide la dispersión del riesgo en cada ventana
    # Asimetría:   indica si los retornos tienden más a caídas que a subidas
    # Curtosis:    captura la probabilidad de eventos extremos (cisnes negros)
    for v in ventanas:
        df_feat[f'{ticker}_vol_{v}d']  = retornos.rolling(v).std()
        df_feat[f'{ticker}_skew_{v}d'] = retornos.rolling(v).skew()
        df_feat[f'{ticker}_kurt_{v}d'] = retornos.rolling(v).kurt()

    # ── Drawdown desde máximo de la ventana larga ──────────────────────────────
    # Proxy de riesgo reciente: qué tan lejos está el precio de su máximo local
    v_larga = max(ventanas)
    rolling_max = df_ohlcv['Close'].rolling(v_larga).max()
    df_feat[f'{ticker}_drawdown_{v_larga}d'] = (
        (df_ohlcv['Close'] - rolling_max) / (rolling_max + 1e-8)
    )

    # ── Indicadores técnicos vía librería 'ta' ─────────────────────────────────
    df_ta = add_all_ta_features(
        df_ohlcv.copy(),
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True
    )

    # Mapa: {nombre_destino → nombre_en_ta}
    indicadores_ta = {
        f'{ticker}_rsi':       'momentum_rsi',    # Oscilador sobrecompra/sobreventa
        f'{ticker}_macd_diff': 'trend_macd_diff', # Señal de cambio de tendencia
        f'{ticker}_atr':       'volatility_atr',  # Volatilidad basada en rangos (True Range)
        f'{ticker}_cci':       'trend_cci',        # Ciclos de precio vs media
        f'{ticker}_bb_width':  'volatility_bbw',  # Anchura de Bandas de Bollinger
        f'{ticker}_mfi':       'volume_mfi',      # Money Flow Index: RSI ponderado por volumen
    }
    for col_dest, col_orig in indicadores_ta.items():
        if col_orig in df_ta.columns:
            df_feat[col_dest] = df_ta[col_orig].values

    # ── Precio relativo a medias móviles ──────────────────────────────────────
    # Señales de tendencia: precio por encima de MA → momentum alcista
    # MA50 captura tendencia de medio plazo; MA200 captura tendencia estructural
    # La ventana se adapta al tamaño del dataset para evitar columnas enteramente NaN
    # (con < 200 días de datos, rolling(200) sería todo NaN → dropna vaciaría el CSV)
    n_filas = len(df_ohlcv)
    for ma in [50, 200]:
        ma_eff = min(ma, max(2, n_filas // 2))
        ma_val = df_ohlcv['Close'].rolling(ma_eff).mean()
        df_feat[f'{ticker}_precio_vs_ma{ma}'] = df_ohlcv['Close'] / (ma_val + 1e-8) - 1

    # ── Volumen relativo (volume surge) ───────────────────────────────────────
    # Ratio entre el volumen actual y la media de 20 días.
    # Valores > 1.5 indican interés institucional inusual y confirman movimientos de precio.
    vol_media_20 = df_ohlcv['Volume'].rolling(20).mean()
    df_feat[f'{ticker}_vol_relativo'] = df_ohlcv['Volume'] / (vol_media_20 + 1e-8)

    # ── Iliquidez de Amihud ────────────────────────────────────────────────────
    # |retorno diario| / volumen en dólares. Mide el impacto de precio por unidad de volumen.
    # Alta iliquidez → mayor slippage → mayor riesgo de ejecución.
    volumen_dolares = df_ohlcv['Volume'] * df_ohlcv['Close']
    df_feat[f'{ticker}_amihud'] = retornos.abs() / (volumen_dolares + 1e-8)

    return df_feat


# ─────────────────────────────────────────────
# Features transversales (cross-asset)
# ─────────────────────────────────────────────

def calcular_correlaciones_dinamicas(dataset: pd.DataFrame,
                                     pares: list = None,
                                     ventanas: list = None) -> pd.DataFrame:
    """
    Calcula correlaciones móviles entre pares de activos (sobre retornos logarítmicos).

    Captura la sinergia dinámica entre activos:
      - Cuando corr(IBIT, IVV) sube → Bitcoin se comporta como activo de riesgo
      - Cuando corr(IBIT, IVV) baja → Bitcoin actúa como diversificador
    El agente puede aprender a reducir exposición a IBIT cuando la correlación sube.

    Parámetros
    ----------
    dataset  : DataFrame que contiene columnas {ticker}_retornos
    pares    : lista de tuplas (ticker_a, ticker_b)
    ventanas : horizontes para el cálculo de correlación rolling

    Retorna
    -------
    DataFrame con columnas corr_{A}_{B}_{v}d para cada par y ventana
    """
    if pares is None:
        pares = PARES_CORRELACION_DEFAULT
    if ventanas is None:
        ventanas = [20, 60]

    df_corr = pd.DataFrame(index=dataset.index)

    for ticker_a, ticker_b in pares:
        col_a = f'{ticker_a}_retornos'
        col_b = f'{ticker_b}_retornos'
        # Solo calcular si ambos activos están presentes en el universo
        if col_a not in dataset.columns or col_b not in dataset.columns:
            continue
        n_filas = len(dataset)
        for v in ventanas:
            v_eff = min(v, max(5, n_filas // 3))
            nombre = f'corr_{ticker_a}_{ticker_b}_{v}d'
            df_corr[nombre] = dataset[col_a].rolling(v_eff).corr(dataset[col_b])

    return df_corr


def calcular_beta_rolling(dataset: pd.DataFrame, tickers: list,
                          ticker_mercado: str = 'IVV',
                          ventana: int = 60) -> pd.DataFrame:
    """
    Calcula la beta rolling de cada activo respecto al mercado (IVV por defecto).

    Beta = Cov(r_activo, r_mercado) / Var(r_mercado)

    Una beta > 1 indica que el activo amplifica los movimientos del mercado;
    beta < 0 indica cobertura natural (relevante para BND en crisis).
    El agente puede usar este indicador para ajustar la exposición cíclica.

    Parámetros
    ----------
    dataset       : DataFrame con columnas {ticker}_retornos
    tickers       : activos para los que calcular la beta
    ticker_mercado: activo de referencia (proxy del mercado)
    ventana       : horizonte de cálculo rolling (días)
    """
    df_beta = pd.DataFrame(index=dataset.index)
    col_mercado = f'{ticker_mercado}_retornos'

    if col_mercado not in dataset.columns:
        return df_beta

    # Ventana adaptativa: nunca superar 1/3 del dataset para evitar columnas enteramente NaN
    ventana_eff = min(ventana, max(10, len(dataset) // 3))

    r_mercado = dataset[col_mercado]
    var_mercado = r_mercado.rolling(ventana_eff).var()

    for ticker in tickers:
        if ticker == ticker_mercado:
            continue
        col_ret = f'{ticker}_retornos'
        if col_ret not in dataset.columns:
            continue
        cov = dataset[col_ret].rolling(ventana_eff).cov(r_mercado)
        df_beta[f'{ticker}_beta_{ventana}d'] = cov / (var_mercado + 1e-8)

    return df_beta


def calcular_features_calendario(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Genera features de calendario derivadas del índice temporal.

    Capturan anomalías estacionales bien documentadas en finanzas:
      - Efecto lunes: retornos históricamente negativos los lunes
      - Efecto enero: mayor rentabilidad en el primer mes del año
      - Efecto fin de trimestre: window dressing institucional en Q4
      - Efecto fin de mes: flujos de rebalanceo sistemático

    Al ser features deterministas (sin lookahead), no introducen data leakage.

    Parámetros
    ----------
    index : índice temporal del dataset (DatetimeIndex)

    Retorna
    -------
    DataFrame con features de calendario normalizadas al rango [0, 1]
    """
    df_cal = pd.DataFrame(index=index)

    # Día de la semana normalizado: lunes=0, viernes=1
    df_cal['cal_dia_semana'] = index.dayofweek / 4.0

    # Mes del año normalizado: enero=0, diciembre=1
    df_cal['cal_mes'] = (index.month - 1) / 11.0

    # Indicador binario de fin de mes (últimos 3 días hábiles del mes)
    # Los fondos rebalancean sistemáticamente en este período
    df_cal['cal_fin_mes'] = (index.day >= 28).astype(np.float32)

    # Indicador binario de fin de trimestre (marzo, junio, septiembre, diciembre)
    df_cal['cal_fin_trimestre'] = index.month.isin([3, 6, 9, 12]).astype(np.float32)

    return df_cal


def calcular_regimen_volatilidad(dataset: pd.DataFrame,
                                 tickers: list,
                                 ventana_corta: int = 20,
                                 ventana_larga: int = 252,
                                 umbral: float = 1.5) -> pd.DataFrame:
    """
    Indicador binario de régimen de alta volatilidad por activo.

    Criterio: vol_rolling_{corta}d > umbral × vol_rolling_{larga}d
      → 1: régimen de alta volatilidad (corrección de mercado, crisis)
      → 0: régimen de calma o volatilidad normal

    Este indicador permite al agente PPO identificar de forma proactiva los cambios
    de régimen y ajustar la exposición a activos volátiles como el IBIT antes de
    que se produzcan caídas bruscas, en lugar de reaccionar a posteriori.

    Parámetros
    ----------
    dataset       : DataFrame con columnas {ticker}_retornos y {ticker}_vol_{v}d
    tickers       : lista de activos para los que calcular el indicador
    ventana_corta : ventana de volatilidad de corto plazo (días)
    ventana_larga : ventana de volatilidad de largo plazo (~1 año)
    umbral        : factor multiplicativo; 1.5 = 50% por encima de la media histórica

    Retorna
    -------
    DataFrame con columnas {ticker}_regimen_alta_vol (float32: 0.0 o 1.0)
    """
    df_reg = pd.DataFrame(index=dataset.index)

    col_vol_corta_key = f'_vol_{ventana_corta}d'

    # Ventana larga adaptativa: con datasets cortos, rolling(252) produce NaN en todo el período.
    # La comparación NaN > X devuelve False en pandas → régimen artificialmente siempre en 0.
    # Usamos como mínimo ventana_corta+1 y como máximo 1/3 del dataset.
    ventana_larga_eff = min(ventana_larga, max(ventana_corta + 1, len(dataset) // 3))

    for ticker in tickers:
        col_ret      = f'{ticker}_retornos'
        col_vol_c    = f'{ticker}{col_vol_corta_key}'

        if col_ret not in dataset.columns:
            continue

        # Volatilidad de largo plazo calculada directamente sobre retornos
        vol_larga = dataset[col_ret].rolling(ventana_larga_eff).std()

        # Usar la volatilidad corta ya calculada si existe; si no, recalcular
        if col_vol_c in dataset.columns:
            vol_corta = dataset[col_vol_c]
        else:
            vol_corta = dataset[col_ret].rolling(ventana_corta).std()

        df_reg[f'{ticker}_regimen_alta_vol'] = (
            vol_corta > umbral * vol_larga
        ).astype(np.float32)

    return df_reg


# ─────────────────────────────────────────────
# Normalización
# ─────────────────────────────────────────────

def normalizar_zscore(df: pd.DataFrame,
                      split_pct: float = 0.8) -> pd.DataFrame:
    """
    Normalización Z-Score sin lookahead bias.

    Calcula μ y σ SOLO sobre el conjunto de entrenamiento (primeros split_pct% de filas)
    y aplica esas mismas estadísticas al conjunto de test.

    Por qué importa:
      Si se normalizara sobre el dataset completo, el test contaminaría el train:
      el agente "vería" implícitamente la media y varianza de precios futuros.
      Esto infla las métricas in-sample y hunde el rendimiento out-of-sample,
      exactamente el patrón observado cuando PPO es peor que las baselines.

    Parámetros
    ----------
    df        : DataFrame completo (train + test concatenados)
    split_pct : fracción de datos que pertenece a train (por defecto 0.8)
    """
    split_idx = int(len(df) * split_pct)
    df_train  = df.iloc[:split_idx]

    mu    = df_train.mean()
    sigma = df_train.std() + 1e-8

    return (df - mu) / sigma
