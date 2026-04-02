"""
Pipeline principal de descarga y preprocesamiento de datos financieros.

Orquesta la ingesta desde Yahoo Finance, la ingeniería de características
y el guardado de los CSVs resultantes en el directorio data/.

Fases del pipeline:
  1. Descarga de datos OHLCV por ticker (con sesión resiliente para proxies SSL)
  2. Cálculo de features estadísticas y técnicas por activo
  3. Correlaciones dinámicas entre pares de activos
  4. Indicadores de régimen de volatilidad
  4b. Beta rolling respecto al mercado (sensibilidad sistémica)
  4c. Features de calendario (anomalías estacionales)
  5. Features de dividendos (calidad del pago: crecimiento, estabilidad, riesgo de cola)
  6. Limpieza, normalización Z-Score (solo estadísticas de train) y guardado

Los CSVs se guardan en orden ASCENDENTE (fecha más antigua primero).
Esto garantiza que el split temporal 80/20 en el entorno de entrenamiento
funcione correctamente: el 80% inicial corresponde al pasado y el 20% final
al período de test out-of-sample.

Robusto para cualquier rango de fechas: datasets cortos (< 1 año), medianos
(1-5 años) o largos (5-25 años). Las ventanas rolling se adaptan y la limpieza
columnar previene CSVs vacíos independientemente del tamaño del dataset.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi.requests import Session as CurlSession
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False

try:
    from feature_ingeneering.data_features import (
        calcular_features_precio,
        calcular_correlaciones_dinamicas,
        calcular_regimen_volatilidad,
        calcular_beta_rolling,
        calcular_features_calendario,
        normalizar_zscore,
        VENTANAS_DEFAULT,
        PARES_CORRELACION_DEFAULT,
    )
except ImportError:
    from src.feature_ingeneering.data_features import (
        calcular_features_precio,
        calcular_correlaciones_dinamicas,
        calcular_regimen_volatilidad,
        calcular_beta_rolling,
        calcular_features_calendario,
        normalizar_zscore,
        VENTANAS_DEFAULT,
        PARES_CORRELACION_DEFAULT,
    )


# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Sesión HTTP resiliente (proxies corporativos)
# ─────────────────────────────────────────────

def _create_session():
    """
    Crea una sesión HTTP que sortea proxies SSL corporativos.

    Usa curl_cffi con impersonación de navegador Chrome para evitar
    bloqueos de firewalls corporativos. Si curl_cffi no está instalado,
    retorna None y yfinance usará su sesión interna por defecto.

    Returns
    -------
    CurlSession o None
    """
    if _CURL_AVAILABLE:
        session = CurlSession(impersonate="chrome")
        session.verify = False
        return session
    return None


_global_session = _create_session()


# ─────────────────────────────────────────────
# Descarga de precios
# ─────────────────────────────────────────────

def download_prices(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Descarga datos OHLCV de Yahoo Finance para cada ticker.

    Maneja errores de forma individual: si un ticker falla, el pipeline
    continúa con los restantes. Solo retorna tickers con datos válidos.

    Parameters
    ----------
    tickers    : lista de símbolos (ej. ['IVV', 'BND', 'IBIT'])
    start_date : fecha de inicio en formato 'YYYY-MM-DD'
    end_date   : fecha de fin en formato 'YYYY-MM-DD'

    Returns
    -------
    dict {ticker: DataFrame OHLCV} para los tickers descargados correctamente
    """
    data = {}
    for ticker in tickers:
        print(f"  Descargando {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Aplanar MultiIndex generado por algunas versiones de yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print(f"  [AVISO] Sin datos para {ticker} en el rango {start_date} – {end_date}.")
                continue
            df = df.sort_index(ascending=True)
            data[ticker] = df
        except Exception as e:
            print(f"  [ERROR] No se pudo descargar {ticker}: {e}")

    print(f"  Tickers válidos descargados: {list(data.keys())}")
    return data


# ─────────────────────────────────────────────
# Features de dividendos
# ─────────────────────────────────────────────

def download_dividends(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga y calcula features de calidad del dividendo para cada ticker.

    La dinámica del dividendo aporta información más allá del importe pagado:
      - div_growth:     cambio porcentual entre pagos consecutivos (tendencia)
      - div_volatility: desviación estándar rolling (4 pagos ≈ 1 año) → estabilidad
      - div_kurtosis:   curtosis rolling (8 pagos ≈ 2 años) → riesgo de cola
                        Alta curtosis = recortes raros pero extremos
      - div_skewness:   asimetría rolling (8 pagos) → sesgo en los cambios

    Las series trimestrales se expanden a frecuencia diaria con forward fill
    para que el agente tenga la información en cada paso de trading.

    Mínimo 4 pagos requeridos por ticker: con menos, las estadísticas rolling
    serían casi enteramente NaN y añadirían ruido en lugar de señal.

    Parameters
    ----------
    tickers    : lista de símbolos
    start_date : fecha de inicio 'YYYY-MM-DD'
    end_date   : fecha de fin 'YYYY-MM-DD'

    Returns
    -------
    DataFrame con features diarias de dividendos, o DataFrame vacío si no hay datos.
    También guarda en data/dividend_features.csv.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    result   = pd.DataFrame()

    for ticker in tickers:
        try:
            print(f"  Procesando dividendos de {ticker}...")
            tk = yf.Ticker(ticker, session=_global_session) if _global_session else yf.Ticker(ticker)
            divs = tk.dividends

            if divs.empty:
                print(f"  {ticker}: sin dividendos (cripto ETF o activo de crecimiento puro)")
                continue

            # Eliminar timezone para homogeneidad con el resto del dataset
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)

            divs = divs[(divs.index >= start_dt) & (divs.index <= end_dt)].to_frame()
            divs.columns = ['div_amount']

            # Mínimo 4 pagos trimestrales para estadísticas rolling con sentido
            if len(divs) < 4:
                print(f"  {ticker}: insuficientes dividendos ({len(divs)} pagos, necesita ≥4)")
                continue

            # Dinámica del dividendo: tendencia, estabilidad y riesgo de cola
            divs['div_growth']     = divs['div_amount'].pct_change()
            divs['div_volatility'] = divs['div_growth'].rolling(4).std()
            divs['div_kurtosis']   = divs['div_growth'].rolling(8).kurt()
            divs['div_skewness']   = divs['div_growth'].rolling(8).skew()

            # Expandir a frecuencia diaria (el agente opera diario, dividendos son trimestrales)
            daily = divs.resample('D').ffill()
            daily.columns = [f'{ticker}_{c}' for c in daily.columns]

            result = daily if result.empty else pd.concat([result, daily], axis=1)

        except Exception as e:
            print(f"  [ERROR] Dividendos {ticker}: {e}")

    if not result.empty:
        result = result.bfill().ffill()
        path = os.path.join(DATA_DIR, "dividend_features.csv")
        result.sort_index(ascending=True).to_csv(path, encoding="utf-8-sig")
        print(f"  Features de dividendos guardadas: {path}")

    return result


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def generate_dataset(tickers: list,
                     start_date: str,
                     end_date: str,
                     windows: list = None,
                     correlation_pairs: list = None,
                     include_dividends: bool = True,
                     source=None) -> tuple:
    """
    Pipeline completo: descarga → features → normalización → guardado.

    Robusto para cualquier rango de fechas:
      - Datasets cortos (< 1 año): ventanas rolling se adaptan al tamaño disponible,
        columnas enteramente NaN se eliminan antes del dropna por filas.
      - Datasets medianos (1-5 años): funcionamiento estándar.
      - Datasets largos (5-25 años): algunos tickers pueden no existir en todo el rango
        (ej. IBIT solo desde 2024). Se usa reindex + bfill/ffill en lugar de .loc[]
        para evitar KeyError por fechas faltantes.

    Parameters
    ----------
    tickers           : lista de símbolos del universo de inversión
    start_date        : fecha de inicio 'YYYY-MM-DD'
    end_date          : fecha de fin 'YYYY-MM-DD'
    windows           : tamaños de ventana rolling para features (por defecto [5, 20, 60])
    correlation_pairs : pares (ticker_a, ticker_b) para correlaciones dinámicas
    include_dividends : si True, añade features de calidad del dividendo
    source            : instancia de DataSource. None → HistoricalSource (Yahoo Finance EOD).
                        Pasar LiveSource() para datos en tiempo real.

    Returns
    -------
    (df_features_norm, df_original_prices)
      - df_features_norm   : features normalizadas con Z-Score (estadísticas solo de train)
      - df_original_prices : precios de cierre sin normalizar (para calcular retornos reales)
    """
    if windows is None:
        windows = VENTANAS_DEFAULT
    if correlation_pairs is None:
        correlation_pairs = PARES_CORRELACION_DEFAULT

    # Fuente de datos: por defecto Yahoo Finance EOD
    if source is None:
        try:
            from pipeline_getdata.data_source import HistoricalSource
        except ImportError:
            from src.pipeline_getdata.data_source import HistoricalSource
        source = HistoricalSource()

    # ── Fase 1: Descarga de datos OHLCV ──────────────────────────────────────
    print(f"\n=== FASE 1: Descarga de datos de mercado [{source.nombre()}] ===")
    raw_data = {}
    for ticker in tickers:
        print(f"  Descargando {ticker}...")
        try:
            df = source.obtener_ohlcv(ticker, start_date, end_date)
            if df.empty:
                print(f"  [AVISO] Sin datos para {ticker} en {start_date} – {end_date}.")
                continue
            raw_data[ticker] = df
        except Exception as e:
            print(f"  [ERROR] No se pudo obtener {ticker}: {e}")

    valid_tickers = list(raw_data.keys())
    print(f"  Tickers válidos: {valid_tickers}")

    if not valid_tickers:
        raise ValueError("No se pudo descargar ningún ticker. Verifica la conexión y los símbolos.")

    # ── Fase 2: Features estadísticas y técnicas por activo ──────────────────
    print(f"\n=== FASE 2: Ingeniería de características ({len(valid_tickers)} activos) ===")
    feature_list = []
    close_prices = pd.DataFrame()

    for ticker in valid_tickers:
        print(f"  Calculando features para {ticker}...")
        df_ohlcv = raw_data[ticker]
        feats    = calcular_features_precio(df_ohlcv, ticker, windows)
        feature_list.append(feats)
        close_prices[f'{ticker}_Close'] = df_ohlcv['Close']

    dataset = pd.concat(feature_list, axis=1)

    # ── Fase 3: Correlaciones dinámicas entre activos ────────────────────────
    # Correlaciones rolling entre pares capturan la diversificación cambiante.
    # Cuando corr(IBIT, IVV) sube → Bitcoin se comporta como activo de riesgo.
    # Cuando baja → actúa como diversificador. El agente aprende a reducir
    # exposición a IBIT cuando la correlación con renta variable aumenta.
    print(f"\n=== FASE 3: Correlaciones dinámicas entre activos ===")
    df_corr = calcular_correlaciones_dinamicas(dataset, correlation_pairs, ventanas=[20, 60])
    dataset = pd.concat([dataset, df_corr], axis=1)
    print(f"  {len(df_corr.columns)} columnas de correlación añadidas.")

    # ── Fase 4: Indicadores de régimen de volatilidad ────────────────────────
    # Indicador binario: ¿la vol de corto plazo supera 1.5× la de largo plazo?
    # Permite al agente reducir exposición a activos volátiles ANTES de un crash.
    print(f"\n=== FASE 4: Indicadores de régimen de volatilidad ===")
    df_reg  = calcular_regimen_volatilidad(dataset, valid_tickers)
    dataset = pd.concat([dataset, df_reg], axis=1)
    print(f"  {len(df_reg.columns)} indicadores de régimen añadidos.")

    # ── Fase 4b: Beta rolling respecto al mercado (IVV como proxy) ───────────
    # Beta = Cov(r_activo, r_mercado) / Var(r_mercado)
    # Beta alta → el activo amplifica las caídas del mercado.
    # Beta negativa → cobertura natural (relevante para BND en crisis).
    market_ticker = 'IVV' if 'IVV' in valid_tickers else valid_tickers[0]
    df_beta = calcular_beta_rolling(dataset, valid_tickers, ticker_mercado=market_ticker)
    if not df_beta.empty:
        dataset = pd.concat([dataset, df_beta], axis=1)
        print(f"  {len(df_beta.columns)} features de beta rolling añadidas.")

    # ── Fase 4c: Features de calendario ──────────────────────────────────────
    # Capturan anomalías estacionales documentadas: efecto lunes, efecto enero,
    # rebalanceos de fin de mes y window dressing de fin de trimestre.
    # Son deterministas → no introducen data leakage.
    df_cal = calcular_features_calendario(dataset.index)
    dataset = pd.concat([dataset, df_cal], axis=1)
    print(f"  {len(df_cal.columns)} features de calendario añadidas.")

    # ── Fase 5: Features de dividendos (opcional) ────────────────────────────
    if include_dividends:
        print(f"\n=== FASE 5: Features de dividendos ===")
        df_divs = download_dividends(valid_tickers, start_date, end_date)
        if not df_divs.empty:
            df_divs_aligned = df_divs.reindex(dataset.index, method='ffill')
            dataset = pd.concat([dataset, df_divs_aligned], axis=1)
            print(f"  {len(df_divs_aligned.columns)} features de dividendos incorporadas.")

    # ── Fase 6: Limpieza, normalización y guardado ───────────────────────────
    print(f"\n=== FASE 6: Limpieza, normalización y guardado ===")

    # Paso 1: Eliminar columnas enteramente NaN.
    # Ocurre cuando el dataset es demasiado corto para alguna ventana rolling
    # (ej. rolling(200) con 150 días → columna todo NaN).
    # Sin esto, el dropna() por filas del paso 2 eliminaría TODAS las filas → CSV vacío.
    cols_before = len(dataset.columns)
    dataset = dataset.dropna(axis=1, how='all')
    cols_dropped = cols_before - len(dataset.columns)
    if cols_dropped > 0:
        print(f"  [AVISO] {cols_dropped} columnas enteramente NaN eliminadas "
              f"(dataset demasiado corto para algunas ventanas rolling).")

    # Paso 2: Rellenar NaN residuales del warmup de ventanas rolling.
    # bfill → propaga el primer valor válido hacia atrás (activos con historia corta)
    # ffill → cierra huecos al final (festivos, mercados cerrados)
    # dropna → elimina filas que no pudieron rellenarse (debería haber muy pocas)
    dataset = dataset.bfill().ffill().dropna()

    if len(dataset) == 0:
        raise ValueError(
            "El dataset quedó vacío tras la limpieza. El rango de fechas es demasiado "
            "corto para calcular ninguna feature. Prueba con un rango más amplio."
        )

    # Paso 3: Sincronizar precios con el índice limpio de features.
    # Se usa reindex() en lugar de .loc[] porque el índice de features puede tener
    # fechas que no existen en close_prices (ej. IVV existe desde 2000 pero IBIT
    # solo desde 2024). .loc[] lanzaría KeyError; reindex() inserta NaN que
    # bfill/ffill resuelve correctamente.
    close_prices = close_prices.reindex(dataset.index).bfill().ffill()

    # Verificar que no quedan NaN en precios tras el relleno
    n_nan_prices = close_prices.isnull().values.sum()
    if n_nan_prices > 0:
        print(f"  [AVISO] {n_nan_prices} NaN residuales en precios. "
              "Rellenando con 1.0 como último recurso.")
        close_prices = close_prices.fillna(1.0)

    # Paso 4: Normalización Z-Score usando SOLO estadísticas de train (sin lookahead).
    # Media y std se calculan sobre el primer 80% de filas y se aplican a todas.
    dataset_norm = normalizar_zscore(dataset)

    # Paso 5: Guardar CSVs en orden ascendente (fecha más antigua primero)
    features_path = os.path.join(DATA_DIR, "normalized_features.csv")
    prices_path   = os.path.join(DATA_DIR, "original_prices.csv")

    dataset_norm.sort_index(ascending=True).to_csv(features_path, encoding="utf-8-sig")
    close_prices.sort_index(ascending=True).to_csv(prices_path, encoding="utf-8-sig")

    n_feat = len(dataset_norm.columns)
    n_days = len(dataset_norm)
    print(f"  Dataset generado: {n_feat} features × {n_days} días de trading")
    print(f"  -> {features_path}")
    print(f"  -> {prices_path}")

    return dataset_norm, close_prices


# ─────────────────────────────────────────────
# Aliases retrocompatibles (usados por main.py)
# ─────────────────────────────────────────────

descargar_precios    = download_prices
descargar_dividendos = download_dividends
generar_dataset      = generate_dataset


# ─────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────

if __name__ == "__main__":
    UNIVERSE   = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
    START_DATE = "2014-01-01"
    END_DATE   = "2026-03-01"

    df_features, df_prices = generate_dataset(UNIVERSE, START_DATE, END_DATE)

    print(f"\nFeatures normalizadas ({len(df_features.columns)} columnas):")
    print(df_features.head())
    print(f"\nPrecios originales ({len(df_prices.columns)} activos):")
    print(df_prices.head())
