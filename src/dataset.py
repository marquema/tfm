import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import ssl
import requests
import urllib3


# --- SOLUCIÓN PARA ENTORNOS CORPORATIVOS ---
# 1. Desactiva advertencias de certificados inseguros
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Crea un contexto SSL que no verifique certificados
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


session = requests.Session()
session.verify = False

def generar_dataset_hibrido(tickers, start_date, end_date):
    """
    Descarga y preprocesa datos de ETFs para un entorno de RL.
    Tickers sugeridos: ['IVV', 'BND', 'IBIT'] (S&P500, Bonos, Bitcoin ETF)
    """
    datos_lista = []
    
    for ticker in tickers:
        print(f"Descargando {ticker}...")
        # Descargamos OHLCV (Open, High, Low, Close, Volume)
        df = yf.download(ticker, start=start_date, end=end_date)
        

        # --- CORRECCIÓN PARA MULTIINDEX ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Validar si el DataFrame está vacío
        if df.empty:
            print(f"¡ERROR! No hay datos para {ticker} en esas fechas.")
            continue

        # 1. Calculamos retornos logarítmicos (más estables para RL)
        df[f'retorno_{ticker}'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Añadimos Indicadores Técnicos (RSI, MACD, Bandas Bollinger)
        # Usamos la librería 'ta' para automatizarlo
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        
        # 3. Selección de features clave para el 'State'
        # Elegimos indicadores que den pistas de tendencia y volatilidad
        features_interes = [
            'momentum_rsi', 
            'trend_macd_diff', 
            'volatility_atr', 
            'trend_cci'
        ]
        
        # Filtramos y renombramos para identificar el activo
        df_cols = df[['Close'] + features_interes].copy()
        df_cols.columns = [f"{ticker}_{c}" for c in df_cols.columns]
        
        datos_lista.append(df_cols)

    # Concatenamos todos los activos por fecha
    dataset = pd.concat(datos_lista, axis=1).dropna()
    
    # 4. Normalización Z-Score (Media 0, Desviación 1)
    # Importante: El agente aprende mejor si los datos están en la misma escala
    dataset_norm = (dataset - dataset.mean()) / dataset.std()
    
    # Mantenemos los precios originales aparte para calcular los retornos reales en el env
    precios_originales = dataset[[f"{t}_Close" for t in tickers]]
    
    return dataset_norm, precios_originales

# Ejemplo de uso:
universo = ['IVV', 'BND', 'IBIT']
df_features, df_precios = generar_dataset_hibrido(universo, "2024-02-01", "2026-01-01")
print(df_features.head())