import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
#import requests
from curl_cffi.requests import Session

#sesión que ignore el SSL
session = Session(impersonate="chrome")
session.verify = False 

def get_table_div(tickers, start, end_date):    
    """
        1. Tasa de Crecimiento del Dividendo (Inter-trimestral): tendencia de crecimiento a largo plazo
        2. Volatilidad del Dividendo (Rolling Std):  Indica si el pago es estable o errático
        3. Curtosis (Colas Pesadas / Riesgo de Recorte): Una curtosis alta en el crecimiento indica que el activo tiene 
            "sorpresas" (recortes o subidas bruscas) fuera de lo normal.
        4. Asimetría (Skewness):  Indica si los cambios en el dividendo tienden a ser a la baja (negativo).
    Args:
        tickers (list): _description_
        start (date): _description_
        end_date (date): _description_

    Returns:
        _type_: _description_
    """
    
    df_final_features = pd.DataFrame()    
    start = pd.to_datetime(start)
    end_date = pd.to_datetime(end_date)

    for ticker in tickers:
        try:
            print(f"Analizando dinámica de dividendos de {ticker}...")
            tk = yf.Ticker(ticker, session=session)
            divs = tk.dividends

            if divs.empty:
                print(f"{ticker}: sin dividendos (posible activo de crecimiento o IBIT)")
                continue

            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)

            # Filtrar por rango del TFM
            divs = divs[(divs.index >= start) & (divs.index <= end_date)].to_frame()
            divs.columns = ['div_amount']

            if divs.empty:
                continue

            # --- CÁLCULO DE MEMORIA LARGA Y RIESGO DE COLA (Dividend Dynamics) ---                        
            divs['div_growth'] = divs['div_amount'].pct_change()
            divs['div_volatility'] = divs['div_growth'].rolling(window=4).std()
            divs['div_kurtosis'] = divs['div_growth'].rolling(window=8).kurt()
            divs['div_skewness'] = divs['div_growth'].rolling(window=8).skew()         
            #Como el agente opera a diario y los dividendos son trimestrales, expandimos los datos y rellenamos (forward fill) para que el agente
            #"sepa" en todo momento cuál es la calidad del dividendo del activo.                       
            df_ticker_divs = divs.resample('D').ffill()
            
            #columnas para identificar el ticker
            df_ticker_divs.columns = [f"{ticker}_{c}" for c in df_ticker_divs.columns]
            if df_final_features.empty:
                df_final_features = df_ticker_divs
            else:
                df_final_features = pd.concat([df_final_features, df_ticker_divs], axis=1)

        except Exception as e:
            print(f"Error con {ticker}: {e}")

    if not df_final_features.empty:
        #rellenar NaNs iniciales por el cálculo de rolling
        df_final_features = df_final_features.bfill().ffill()
        
        # Guardar CSV con indicadores de calidad de dividendos
        df_final_features.sort_index(ascending=False).to_csv(
            "tutor_resultados_dividendos.csv", 
            index=True, 
            encoding="utf-8-sig"
        )
        print("CSV de dividendos avanzado guardado.")
        return df_final_features
    else:
        print("No hay datos de dividendos para procesar.")
        return None

def get_table_div_classic(tickers, start, end_date):


    
    df = pd.DataFrame()    
    start = pd.to_datetime(start)
    end_date = pd.to_datetime(end_date)

    for ticker in tickers:
        try:
            print(f"Buscando dividendos de {ticker}...")
            tk = yf.Ticker(ticker, session=session)
            divs = tk.dividends

            if divs.empty:
                print(f"{ticker}: sin dividendos disponibles")
                continue

            # Quitar timezone del índice
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)

            # Filtrar por fechas
            divs = divs[(divs.index >= start) & (divs.index <= end_date)]

            if divs.empty:
                print(f"{ticker}: sin dividendos en ese rango")
                continue

            df2 = pd.DataFrame(
                data=divs.values,
                index=divs.index.strftime("%Y-%m-%d"),
                columns=[ticker]
            )

            df = pd.concat([df, df2], axis=1)

        except Exception as e:
            print(f"Error con {ticker}: {e}")

    if not df.empty:
        df = df.sort_index(ascending=False)
        df.to_csv("tutor_resultados_dividendos.csv", index=True, encoding="utf-8-sig")
        print("CSV guardado: resultados_dividendos.csv")
        print(df.head())
    else:
        print("No hay datos para guardar")


def generar_dataset_hibrido(tickers, start_date, end_date):
    """ 
        1. Retornos Logarítmicos
        2. Asimetría (Skewness): Captura si hay más riesgo de caídas que de subidas
        3. Curtosis (Kurtosis): Captura la probabilidad de eventos extremos (Colas Pesadas)
        4. Volatilidad (Memoria Larga): Desviación estándar móvil

    Args:
        tickers (_type_): _description_
        start_date (_type_): _description_
        end_date (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    datos_lista = []
    tickers_validos = []

    for ticker in tickers:
        print(f"Descargando {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"¡ERROR! No hay datos para {ticker} en esas fechas.")
            continue

        
        df['retornos'] = np.log(df['Close'] / df['Close'].shift(1))

        # --- SECCIÓN NUEVA: CAPTURA DE MEMORIA LARGA Y COLAS PESADAS ---
        # Ventana de 20 días para capturar la dinámica reciente
        window = 20                
        df['skewness'] = df['retornos'].rolling(window=window).skew()                
        df['kurtosis'] = df['retornos'].rolling(window=window).kurt()                
        df['volatilidad'] = df['retornos'].rolling(window=window).std()
        # --------------------------------------------------------------

        # Indicadores Técnicos Clásicos
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        features_interes = [
            'momentum_rsi',
            'trend_macd_diff',
            'volatility_atr',
            'trend_cci',
            'retornos',   # Añadimos los retornos como feature
            'skewness',   # Añadimos momentos de orden superior
            'kurtosis',
            'volatilidad'
        ]

        df_cols = df[['Close'] + features_interes].copy()
        df_cols.columns = [f"{ticker}_{c}" for c in df_cols.columns]

        datos_lista.append(df_cols)
        tickers_validos.append(ticker)

    if not datos_lista:
        raise ValueError("No se pudo descargar ningún ticker correctamente.")

    # Unir todos los datos
    dataset = pd.concat(datos_lista, axis=1)

    # --- SECCIÓN NUEVA: CORRELACIONES DINÁMICAS (SINERGIA) ---
    # Si tenemos IBIT y IVV (como proxy de mercado), calculamos su correlación móvil
    # Esto le enseña al agente cuándo el Bitcoin se comporta como "refugio" o como "riesgo"
    if 'IBIT_retornos' in dataset.columns and 'IVV_retornos' in dataset.columns:
        dataset['corr_IBIT_IVV'] = dataset['IBIT_retornos'].rolling(window=20).corr(dataset['IVV_retornos'])
    # ---------------------------------------------------------
    
    dataset = dataset.bfill().ffill().dropna()

    # Normalización (Z-Score)
    # Importante: Solo normalizamos las features, no los precios de cierre originales
    dataset_norm = (dataset - dataset.mean()) / dataset.std()
    
    # Precios originales para el entorno de trading
    precios_originales = dataset[[f"{t}_Close" for t in tickers_validos]]

    # Guardar resultados (manteniendo tu lógica de orden descendente)
    precios_originales.sort_index(ascending=False).to_csv("precios_originales.csv", encoding="utf-8-sig")
    dataset_norm.sort_index(ascending=False).to_csv("features_normalizadas.csv", encoding="utf-8-sig")

    print(f"Dataset generado con {len(dataset_norm.columns)} características.")
    return dataset_norm, precios_originales

def generar_dataset_completo(tickers, start_date, end_date):
    """
    Pipeline completo de generación de features:
    1. Features técnicas + momentos estadísticos (SIN columnas Close → evita data leakage)
    2. Correlación dinámica IBIT-IVV (régimen cripto: refugio vs riesgo)
    3. Dividend dynamics integradas (crecimiento, volatilidad, kurtosis, skewness del dividendo)

    Genera 2 CSVs:
      - features_normalizadas.csv: observaciones del agente (Z-Score, sin precios absolutos)
      - precios_originales.csv: precios de cierre para calcular P&L en el entorno
    """
    datos_lista = []
    precios_lista = []
    tickers_validos = []

    for ticker in tickers:
        print(f"Descargando {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"¡ERROR! No hay datos para {ticker} en esas fechas.")
            continue

        # Precios de cierre → fichero separado para la mecánica del entorno
        precios_cols = df[['Close']].copy()
        precios_cols.columns = [f"{ticker}_Close"]
        precios_lista.append(precios_cols)

        # Retornos logarítmicos y momentos estadísticos (memoria larga y colas pesadas)
        df['retornos'] = np.log(df['Close'] / df['Close'].shift(1))
        window = 20
        df['skewness']   = df['retornos'].rolling(window=window).skew()
        df['kurtosis']   = df['retornos'].rolling(window=window).kurt()
        df['volatilidad'] = df['retornos'].rolling(window=window).std()

        # Indicadores técnicos clásicos
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )

        features_interes = [
            'momentum_rsi',    # Sobrecompra / sobreventa
            'trend_macd_diff', # Momentum de tendencia
            'volatility_atr',  # Volatilidad de precio (rango verdadero)
            'trend_cci',       # Desviación cíclica del precio
            'retornos',        # Log-retorno diario
            'skewness',        # Asimetría (¿más riesgo de caída que de subida?)
            'kurtosis',        # Colas pesadas (¿alta probabilidad de eventos extremos?)
            'volatilidad',     # Desviación estándar móvil (memoria larga de la volatilidad)
        ]

        df_cols = df[features_interes].copy()
        df_cols.columns = [f"{ticker}_{c}" for c in features_interes]

        datos_lista.append(df_cols)
        tickers_validos.append(ticker)

    if not datos_lista:
        raise ValueError("No se pudo descargar ningún ticker correctamente.")

    # Dataset base (features técnicas)
    dataset = pd.concat(datos_lista, axis=1)
    precios_originales = pd.concat(precios_lista, axis=1)

    # Correlación dinámica IBIT-IVV: enseña al agente cuándo el Bitcoin actúa como
    # "activo de refugio" (correlación baja) o "activo de riesgo" (correlación alta con IVV)
    if 'IBIT_retornos' in dataset.columns and 'IVV_retornos' in dataset.columns:
        dataset['corr_IBIT_IVV'] = (
            dataset['IBIT_retornos'].rolling(window=20).corr(dataset['IVV_retornos'])
        )

    # Limpiar y alinear
    dataset = dataset.bfill().ffill().dropna()
    precios_originales = precios_originales.loc[dataset.index].bfill().ffill()

    # Integrar dividend dynamics (solo métricas estadísticas, no importe bruto)
    df_divs = get_table_div(tickers_validos, start_date, end_date)
    if df_divs is not None:
        div_cols = [c for c in df_divs.columns if any(
            m in c for m in ['div_growth', 'div_volatility', 'div_kurtosis', 'div_skewness']
        )]
        df_divs_stats = df_divs[div_cols].copy()
        df_divs_stats.index = pd.to_datetime(df_divs_stats.index)
        dataset.index = pd.to_datetime(dataset.index)
        # Left join: mantenemos todos los días de trading; dividendos ya tienen forward-fill diario
        dataset = dataset.join(df_divs_stats, how='left').bfill().ffill()
        print(f"Dividend dynamics integradas: {len(div_cols)} columnas adicionales.")

    # Normalización Z-Score global (sobre features completas, sin precios absolutos)
    dataset_norm = (dataset - dataset.mean()) / dataset.std()

    # Guardar (orden descendente para legibilidad humana)
    precios_originales.sort_index(ascending=False).to_csv("precios_originales.csv", encoding="utf-8-sig")
    dataset_norm.sort_index(ascending=False).to_csv("features_normalizadas.csv", encoding="utf-8-sig")

    n_tech   = len(tickers_validos) * 8 + (1 if 'corr_IBIT_IVV' in dataset.columns else 0)
    n_divs   = len(div_cols) if df_divs is not None else 0
    print(f"Dataset completo: {len(dataset_norm.columns)} features "
          f"({n_tech} técnicas + {n_divs} dividend dynamics), {len(dataset_norm)} días.")
    return dataset_norm, precios_originales


def generar_dataset_hibrido_classic(tickers, start_date, end_date):
    datos_lista = []
    tickers_validos = []

    for ticker in tickers:
        print(f"Descargando {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"¡ERROR! No hay datos para {ticker} en esas fechas.")
            continue

        df[f'retorno_{ticker}'] = np.log(df['Close'] / df['Close'].shift(1))

        df = add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True
        )

        features_interes = [
            'momentum_rsi',
            'trend_macd_diff',
            'volatility_atr',
            'trend_cci'
        ]


        df_cols = df[['Close'] + features_interes].copy()
        df_cols.columns = [f"{ticker}_{c}" for c in df_cols.columns]

        datos_lista.append(df_cols)
        tickers_validos.append(ticker)

    if not datos_lista:
        raise ValueError("No se pudo descargar ningún ticker correctamente.")

    # 1. Unimos todos los datos (IBIT tendrá NaNs desde 2014 hasta 2024)
    dataset = pd.concat(datos_lista, axis=1)

    # 2. Rellenamos hacia atrás (Backfill) para los activos que empezaron tarde
    # Esto copia el primer precio/valor válido de 2024 hacia todos los años anteriores
    dataset = dataset.bfill()

    # 3. Rellenamos hacia adelante (Forward fill) por si algún activo dejó de cotizar
    # o faltan datos al final de la serie
    dataset = dataset.ffill()

    # 4. Finalmente, si queda alguna fila con NaNs (que no debería), la quitamos
    dataset = dataset.dropna()


    #dataset = pd.concat(datos_lista, axis=1).dropna()
    dataset_norm = (dataset - dataset.mean()) / dataset.std()
    precios_originales = dataset[[f"{t}_Close" for t in tickers_validos]]

    if not precios_originales.empty:
        precios_originales = precios_originales.sort_index(ascending=False)
        precios_originales.to_csv("precios_originales.csv", index=True, encoding="utf-8-sig")
        print("CSV guardado: precios_originales.csv")
        print(precios_originales.head())
    else:
        print("No hay datos para guardar")

    if not dataset_norm.empty:
        dataset_norm = dataset_norm.sort_index(ascending=False)
        dataset_norm.to_csv("features_normalizadas.csv", index=True, encoding="utf-8-sig")
        print("CSV guardado: features_normalizadas.csv")
        print(dataset_norm.head())
    else:
        print("No hay datos para guardar")        

    return dataset_norm, precios_originales
