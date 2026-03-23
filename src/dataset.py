import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features

def tabla_div(tickers, start, end_date):
    df = pd.DataFrame()

    start = pd.to_datetime(start)
    end_date = pd.to_datetime(end_date)

    for ticker in tickers:
        try:
            print(f"Buscando dividendos de {ticker}...")
            tk = yf.Ticker(ticker)
            divs = tk.dividends

            if divs.empty:
                print(f"{ticker}: sin dividendos disponibles")
                continue

            # Quitar timezone del índice
            if getattr(divs.index, "tz", None) is not None:
                divs.index = divs.index.tz_localize(None)            
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
            print(f"No hay datos para {ticker} en esas fechas.")
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
    
    dataset = pd.concat(datos_lista, axis=1)

    #2. Rellenamos hacia atrás (Backfill) para los activos que empezaron tarde
    dataset = dataset.bfill()
    #Rellenamos hacia adelante (Forward fill) si algún activo dejó de cotizar
    #o faltan datos al final de la serie
    dataset = dataset.ffill()    
    dataset = dataset.dropna()


    #dataset = pd.concat(datos_lista, axis=1).dropna()
    dataset_norm = (dataset - dataset.mean()) / dataset.std()
    precios_originales = dataset[[f"{t}_Close" for t in tickers_validos]]

    return dataset_norm, precios_originales


universo = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']

tabla_div(universo, "2014-01-01", "2026-03-01")

df_features, df_precios = generar_dataset_hibrido(universo, "2014-01-01", "2026-03-01")

df_features.to_csv("features_normalizadas.csv", index=True, encoding="utf-8-sig")
df_precios.to_csv("precios_originales.csv", index=True, encoding="utf-8-sig")

print(df_features.head())