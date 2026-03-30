"""
Pipeline principal de descarga y preprocesamiento de datos financieros.

Orquesta la ingesta desde Yahoo Finance, la ingeniería de características
y el guardado de los CSVs resultantes en el directorio data/.

Fases del pipeline:
  1. Descarga de datos OHLCV por ticker (con sesión resiliente para proxies SSL)
  2. Cálculo de features estadísticas y técnicas por activo
  3. Correlaciones dinámicas entre pares de activos
  4. Indicadores de régimen de volatilidad
  5. Features de dividendos (dinámica de pagos: crecimiento, volatilidad, curtosis)
  6. Normalización Z-Score y guardado en data/

Corrección importante frente a versiones anteriores:
  Los CSVs se guardan en orden ASCENDENTE (fecha más antigua primero).
  Esto garantiza que el split temporal 80/20 en el entorno de entrenamiento
  funcione correctamente: el 80% inicial corresponde al pasado y el 20% final
  al período de test out-of-sample.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from curl_cffi.requests import Session as CurlSession
    _curl_disponible = True
except ImportError:
    _curl_disponible = False

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
# Configuración de directorios
# ─────────────────────────────────────────────
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Sesión resiliente para entornos corporativos
# ─────────────────────────────────────────────

def _crear_sesion():
    """
    Crea una sesión HTTP resiliente para entornos con proxies SSL corporativos.

    Usa curl_cffi con impersonación de navegador Chrome para evitar bloqueos
    por firewalls corporativos. Si curl_cffi no está instalado, retorna None
    y yfinance usará su sesión interna por defecto.
    """
    if _curl_disponible:
        sesion = CurlSession(impersonate="chrome")
        sesion.verify = False
        return sesion
    return None


_sesion_global = _crear_sesion()


# ─────────────────────────────────────────────
# Descarga de datos de mercado
# ─────────────────────────────────────────────

def descargar_precios(tickers: list, start_date: str, end_date: str) -> dict:
    """
    Descarga datos OHLCV de Yahoo Finance para cada ticker.

    Maneja errores de forma individual: si un ticker falla, el pipeline
    continúa con los tickers restantes. Retorna solo los tickers con datos válidos.

    Retorna
    -------
    dict {ticker: DataFrame OHLCV} para los tickers descargados correctamente
    """
    datos = {}
    for ticker in tickers:
        print(f"  Descargando {ticker}...")
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            # Normalizar MultiIndex generado por algunas versiones de yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print(f"  [AVISO] Sin datos para {ticker} en el rango {start_date} – {end_date}.")
                continue
            # Garantizar orden ascendente (fecha más antigua primero)
            df = df.sort_index(ascending=True)
            datos[ticker] = df
        except Exception as e:
            print(f"  [ERROR] No se pudo descargar {ticker}: {e}")

    print(f"  Tickers válidos descargados: {list(datos.keys())}")
    return datos


def descargar_dividendos(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Descarga y procesa la dinámica de dividendos para los tickers del universo.

    Calcula cuatro indicadores de calidad del dividendo:
      - div_amount:    importe del dividendo pagado
      - div_growth:    tasa de crecimiento entre pagos (tendencia a largo plazo)
      - div_volatility: volatilidad del crecimiento (ventana 4 pagos) — estabilidad del pago
      - div_kurtosis:  curtosis del crecimiento (ventana 8 pagos) — riesgo de recorte brusco
      - div_skewness:  asimetría del crecimiento (ventana 8 pagos) — sesgo en los cambios

    Las series trimestrales se expanden a frecuencia diaria con forward fill para que
    el agente disponga de la información de calidad del dividendo en cada paso diario.

    Guarda el resultado en data/dividend_features.csv.
    Retorna DataFrame con las features, o DataFrame vacío si no hay datos.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    df_final = pd.DataFrame()

    for ticker in tickers:
        try:
            print(f"  Procesando dividendos de {ticker}...")
            # Usar sesión resiliente para el objeto Ticker
            tk = yf.Ticker(ticker, session=_sesion_global) if _sesion_global else yf.Ticker(ticker)
            divs = tk.dividends

            if divs.empty:
                print(f"  {ticker}: sin dividendos (cripto ETF o activo de crecimiento puro)")
                continue

            # Eliminar timezone para homogeneidad con el resto del dataset
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)

            divs = divs[(divs.index >= start_dt) & (divs.index <= end_dt)].to_frame()
            divs.columns = ['div_amount']

            # Mínimo de 4 pagos para calcular rolling stats con sentido
            if len(divs) < 4:
                print(f"  {ticker}: insuficientes dividendos ({len(divs)} pagos, necesita ≥4)")
                continue

            # Dinámica del dividendo: memoria larga y riesgo de cola
            divs['div_growth']     = divs['div_amount'].pct_change()
            divs['div_volatility'] = divs['div_growth'].rolling(4).std()
            divs['div_kurtosis']   = divs['div_growth'].rolling(8).kurt()
            divs['div_skewness']   = divs['div_growth'].rolling(8).skew()

            # Expansión a frecuencia diaria: el agente opera día a día,
            # pero los dividendos son trimestrales → forward fill
            df_diario = divs.resample('D').ffill()
            df_diario.columns = [f'{ticker}_{c}' for c in df_diario.columns]

            df_final = df_diario if df_final.empty else pd.concat([df_final, df_diario], axis=1)

        except Exception as e:
            print(f"  [ERROR] Dividendos {ticker}: {e}")

    if not df_final.empty:
        # Relleno de NaNs iniciales producidos por los cálculos rolling
        df_final = df_final.bfill().ffill()
        ruta = os.path.join(DATA_DIR, "dividend_features.csv")
        df_final.sort_index(ascending=True).to_csv(ruta, encoding="utf-8-sig")
        print(f"  Features de dividendos guardadas: {ruta}")

    return df_final


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def generar_dataset(tickers: list,
                    start_date: str,
                    end_date: str,
                    ventanas: list = None,
                    pares_correlacion: list = None,
                    incluir_dividendos: bool = True,
                    source=None) -> tuple:
    """
    Pipeline completo de generación del dataset de features normalizado.

    Parámetros
    ----------
    tickers             : lista de símbolos del universo de inversión
    start_date          : fecha de inicio (formato 'YYYY-MM-DD')
    end_date            : fecha de fin
    ventanas            : horizontes temporales para cálculos rolling
    pares_correlacion   : pares (ticker_a, ticker_b) para correlaciones dinámicas
    incluir_dividendos  : si True, añade features de calidad del dividendo
    source              : instancia de DataSource. Si None, usa HistoricalSource()
                          (Yahoo Finance EOD). Para tiempo real, pasar LiveSource().

    Retorna
    -------
    (df_features_norm, df_precios_originales)
      - df_features_norm      : todas las features normalizadas con Z-Score
      - df_precios_originales : solo los precios de cierre sin normalizar,
                                usados por el entorno de trading para calcular retornos reales
    """
    if ventanas is None:
        ventanas = VENTANAS_DEFAULT
    if pares_correlacion is None:
        pares_correlacion = PARES_CORRELACION_DEFAULT

    # Fuente de datos: por defecto Yahoo Finance EOD; se puede sustituir por LiveSource
    if source is None:
        try:
            from pipeline_getdata.data_source import HistoricalSource
        except ImportError:
            from src.pipeline_getdata.data_source import HistoricalSource
        source = HistoricalSource()

    # ── Fase 1: Descarga de datos OHLCV ───────────────────────────────────────
    print(f"\n=== FASE 1: Descarga de datos de mercado [{source.nombre()}] ===")
    datos_raw = {}
    for ticker in tickers:
        print(f"  Descargando {ticker}...")
        try:
            df = source.obtener_ohlcv(ticker, start_date, end_date)
            if df.empty:
                print(f"  [AVISO] Sin datos para {ticker} en {start_date} – {end_date}.")
                continue
            datos_raw[ticker] = df
        except Exception as e:
            print(f"  [ERROR] No se pudo obtener {ticker}: {e}")

    tickers_validos = list(datos_raw.keys())
    print(f"  Tickers válidos: {tickers_validos}")

    if not tickers_validos:
        raise ValueError("No se pudo descargar ningún ticker. Verifica la conexión y los símbolos.")

    # ── Fase 2: Features estadísticas y técnicas por activo ────────────────────
    print(f"\n=== FASE 2: Ingeniería de características ({len(tickers_validos)} activos) ===")
    lista_features = []
    precios_close  = pd.DataFrame()

    for ticker in tickers_validos:
        print(f"  Calculando features para {ticker}...")
        df_ohlcv = datos_raw[ticker]
        feats    = calcular_features_precio(df_ohlcv, ticker, ventanas)
        lista_features.append(feats)
        precios_close[f'{ticker}_Close'] = df_ohlcv['Close']

    dataset = pd.concat(lista_features, axis=1)

    # ── Fase 3: Correlaciones dinámicas entre activos ──────────────────────────
    print("\n=== FASE 3: Correlaciones dinámicas entre activos ===")
    df_corr = calcular_correlaciones_dinamicas(dataset, pares_correlacion, ventanas=[20, 60])
    dataset = pd.concat([dataset, df_corr], axis=1)
    print(f"  {len(df_corr.columns)} columnas de correlación añadidas.")

    # ── Fase 4: Indicadores de régimen de volatilidad ──────────────────────────
    print("\n=== FASE 4: Indicadores de régimen de volatilidad ===")
    df_reg  = calcular_regimen_volatilidad(dataset, tickers_validos)
    dataset = pd.concat([dataset, df_reg], axis=1)
    print(f"  {len(df_reg.columns)} indicadores de régimen añadidos.")

    # ── Fase 4b: Beta rolling respecto al mercado (IVV como proxy) ───────────
    # Mide la sensibilidad sistémica de cada activo: un activo con beta alta
    # amplifica las caídas del mercado; uno con beta negativa actúa como cobertura.
    ticker_mercado = 'IVV' if 'IVV' in tickers_validos else tickers_validos[0]
    df_beta = calcular_beta_rolling(dataset, tickers_validos, ticker_mercado=ticker_mercado)
    if not df_beta.empty:
        dataset = pd.concat([dataset, df_beta], axis=1)
        print(f"  {len(df_beta.columns)} features de beta rolling añadidas.")

    # ── Fase 4c: Features de calendario ──────────────────────────────────────
    # Capturan anomalías estacionales documentadas: efecto lunes, efecto enero,
    # rebalanceos de fin de mes y window dressing de fin de trimestre.
    # Son deterministas: no introducen data leakage.
    df_cal = calcular_features_calendario(dataset.index)
    dataset = pd.concat([dataset, df_cal], axis=1)
    print(f"  {len(df_cal.columns)} features de calendario añadidas.")

    # ── Fase 5: Features de dividendos (opcional) ─────────────────────────────
    if incluir_dividendos:
        print("\n=== FASE 5: Features de dividendos ===")
        df_divs = descargar_dividendos(tickers_validos, start_date, end_date)
        if not df_divs.empty:
            # Alinear el índice de dividendos con el dataset principal
            df_divs_al = df_divs.reindex(dataset.index, method='ffill')
            dataset    = pd.concat([dataset, df_divs_al], axis=1)
            print(f"  {len(df_divs_al.columns)} features de dividendos incorporadas.")

    # ── Fase 6: Limpieza, normalización y guardado ────────────────────────────
    print("\n=== FASE 6: Limpieza, normalización y guardado ===")

    # Rellenar huecos temporales:
    #   - Backfill: para activos con historia más corta (IBIT cotiza desde enero 2024)
    #     se propaga hacia atrás el primer valor válido
    #   - Forward fill: para huecos al final de la serie (mercados cerrados, festivos)
    #   - dropna: eliminar filas con NaN residuales (inicio de la serie con rolling)
    dataset = dataset.bfill().ffill().dropna()

    # Sincronizar precios originales con el índice limpio del dataset
    # y aplicar el mismo relleno que se hizo en las features:
    # activos con historia corta (ej. IBIT desde 2024) quedan NaN en fechas anteriores.
    # bfill propaga hacia atrás el primer precio válido; ffill cierra huecos al final.
    precios_close = precios_close.loc[dataset.index].bfill().ffill()

    # Normalización Z-Score solo sobre las features (no sobre los precios originales)
    dataset_norm = normalizar_zscore(dataset)

    # Guardar en data/ en orden ASCENDENTE (fecha más antigua primero)
    # IMPORTANTE: versiones anteriores guardaban en orden descendente,
    # lo que causaba que el split 80/20 entrenara sobre datos recientes
    # y testeara sobre datos históricos (error temporal invertido).
    ruta_features = os.path.join(DATA_DIR, "normalized_features.csv")
    ruta_precios  = os.path.join(DATA_DIR, "original_prices.csv")

    dataset_norm.sort_index(ascending=True).to_csv(ruta_features, encoding="utf-8-sig")
    precios_close.sort_index(ascending=True).to_csv(ruta_precios,  encoding="utf-8-sig")

    n_feat = len(dataset_norm.columns)
    n_dias = len(dataset_norm)
    print(f"  Dataset generado: {n_feat} features × {n_dias} días de trading")
    print(f"  -> {ruta_features}")
    print(f"  -> {ruta_precios}")

    return dataset_norm, precios_close


# ─────────────────────────────────────────────
# Ejecución directa del pipeline
# ─────────────────────────────────────────────

if __name__ == "__main__":
    UNIVERSO    = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
    START_DATE  = "2014-01-01"
    END_DATE    = "2026-03-01"

    df_features, df_precios = generar_dataset(UNIVERSO, START_DATE, END_DATE)

    print(f"\nPrimeras filas del dataset normalizado ({len(df_features.columns)} columnas):")
    print(df_features.head())
    print(f"\nPrecios originales ({len(df_precios.columns)} activos):")
    print(df_precios.head())
