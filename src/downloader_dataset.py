import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
#import requests
from curl_cffi.requests import Session

def get_table_div(tickers, start, end_date):
    # Crear una sesión que ignore el SSL
    session = Session(impersonate="chrome")
    session.verify = False 
    
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

    return dataset_norm, precios_originales


#universo = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']

#get_table_div(universo, "2024-02-01", "2026-01-01")

#df_features, df_precios = get_dataset_hybrid(universo, "2024-02-01", "2026-01-01")

#df_features.to_csv("features_normalizadas.csv", index=True, encoding="utf-8-sig")
#df_precios.to_csv("precios_originales.csv", index=True, encoding="utf-8-sig")

#print(df_features.head())