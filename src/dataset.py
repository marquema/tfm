import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import ssl
import requests
import urllib3
from yahoo_fin import stock_info as si
from datetime import datetime, timedelta, date



#1. Desactiva advertencias de certifs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


session = requests.Session()
session.verify = False

def tabla_div(tickers, start, end_date):
    df = pd.DataFrame()
    cont = 0
    for ticker in tickers:
        try:
            aux_div = si.get_dividends(ticker, start_date=start, end_date=end_date)

            df2 = pd.DataFrame(index=aux_div.index.astype(str).to_list(),
                               data=aux_div['dividend'].tolist(),
                               columns=[ticker])
            df = pd.concat([df, df2], axis=1)
            df = df.sort_index(ascending=False)
            cont += 1

        except Exception as error:
            print(str(error))
            print(ticker)
            continue

    if not df.empty:
        df.to_csv('resultados_dividendos.csv', index=True)
    else:
        print("No hay datos para guardar")

def generar_dataset_hibrido(tickers, start_date, end_date):
    """
    Descarga y preprocesa datos de ETFs para un entorno de RL.
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

    dataset = pd.concat(datos_lista, axis=1).dropna()

    dataset_norm = (dataset - dataset.mean()) / dataset.std()

    precios_originales = dataset[[f"{t}_Close" for t in tickers_validos]]

    return dataset_norm, precios_originales



universo = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
tabla_div(universo, "2024-02-01", "2026-01-01")
df_features, df_precios = generar_dataset_hibrido(universo, "2024-02-01", "2026-01-01")
print(df_features.head())