"""
Abstracción de fuentes de datos para el pipeline de features.

Qué es este módulo en una frase:
    Es la "capa de aislamiento" entre el origen físico de los datos
    (Yahoo Finance, ficheros CSV, una API en tiempo real) y el resto del
    pipeline TFM. Implementa el patrón Strategy: cualquier componente
    posterior trabaja contra la misma interfaz (`get_ohlcv`), sin
    importar de dónde vengan los datos en realidad.

Por qué existe esta abstracción:
    Sin ella, `data_downloader.py` llamaría directamente a `yfinance` y
    cambiar de fuente exigiría reescribir el pipeline. Con ella, basta
    con instanciar otra clase que cumpla el contrato. Tiene tres ventajas
    académicas defendibles ante el tribunal:

      1. EXTENSIBILIDAD: añadir un proveedor en tiempo real solo requiere
         crear una subclase de LiveSource. El entrenamiento, los baselines
         y la simulación funcionan sin cambios.
      2. TESTABILIDAD: los tests pueden inyectar CsvSource con datos
         fijos para reproducibilidad (sin depender de la red ni de Yahoo).
      3. ROBUSTEZ ARQUITECTÓNICA: si Yahoo Finance cierra su API gratuita
         o cambia su contrato (algo que ya ha pasado), el impacto se
         contiene en una sola clase.

Fuentes disponibles:
  - HistoricalSource : Yahoo Finance (yfinance) — datos EOD. Fuente actual
                       del pipeline. Usa curl_cffi para sortear proxies SSL
                       corporativos cuando está disponible.
  - LiveSource: placeholder para datos en tiempo real (WebSocket o
                       REST). Documentado pero no implementado — queda como
                       "trabajo futuro" en la memoria del TFM.
  - CsvSource: carga desde CSVs locales. No se usa hoy en producción,
                       pero es el camino estándar para tests unitarios o
                       trabajar offline con datos previamente descargados.

Interfaz común (todas las clases implementan el mismo método):

    get_ohlcv(ticker, start_date, end_date) -> pd.DataFrame
        Retorna DataFrame con columnas: Open, High, Low, Close, Volume
        Índice: DatetimeIndex en orden ascendente.

Cómo cambiar de fuente en el pipeline:
    generate_dataset(tickers, start, end, source=MiLiveSource(...))
    El resto del pipeline (features, entorno, baselines) no necesita
    modificarse — esa es exactamente la ventaja del patrón Strategy.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


# Interfaz base
class DataSource(ABC):
    """
    Contrato abstracto que toda fuente de datos del TFM debe cumplir.

    Heredando de ABC (Abstract Base Class) y marcando los métodos con
    @abstractmethod, garantizamos que ninguna subclase pueda instanciarse
    sin implementar `get_ohlcv` y `name`. Si alguien crea una clase mal
    formada e intenta usarla, Python lanza TypeError EN TIEMPO DE
    INSTANCIACIÓN, no en runtime cuando ya es demasiado tarde.

    Es la forma en Python de definir interfaces (a diferencia
    de Java, donde existe la palabra clave 'interface'; en Python se
    expresa con ABC).
    """

    @abstractmethod
    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retorna datos OHLCV para un activo en el rango solicitado.

        Parameters
        ----------
        ticker : símbolo del activo (ej. 'IVV')
        start_date : fecha de inicio en formato 'YYYY-MM-DD'
        end_date: fecha de fin en formato 'YYYY-MM-DD'

        Returns
        -------
        DataFrame con columnas [Open, High, Low, Close, Volume],
        índice DatetimeIndex ascendente (fecha más antigua primero).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Identificador legible de la fuente, usado en logs del pipeline."""
        ...

    # Alias retrocompatibles (el pipeline actual usa estos nombres)
    def obtener_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Alias de get_ohlcv() para retrocompatibilidad con el pipeline."""
        return self.get_ohlcv(ticker, start_date, end_date)

    def nombre(self) -> str:
        """Alias de name() para retrocompatibilidad con el pipeline."""
        return self.name()


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Fuente 1: Datos históricos EOD (Yahoo Finance)
# # # # # # # # # # # # # # # # # # # # # # # # # 

class HistoricalSource(DataSource):
    """
    Fuente de datos históricos End-Of-Day desde Yahoo Finance.

    Es la fuente activa por defecto en el pipeline TFM. EOD significa
    "end of day": un único registro por activo y día con los precios de
    apertura, máximo, mínimo y cierre + el volumen total. Suficiente para
    un entorno DRL que decide a frecuencia diaria.

    Estrategia HTTP — sesión resiliente:
        En entornos corporativos los proxies SSL bloquean librerías típicas
        (requests, urllib). curl_cffi se hace pasar por Chrome (incluyendo
        TLS fingerprinting) y sortea ese bloqueo. Si la librería no está
        instalada, caemos a la sesión estándar de yfinance — el pipeline
        sigue funcionando, solo pierde la robustez extra ante proxies.

    Notas técnicas relevantes:
      - `auto_adjust=True` (en get_ohlcv): yfinance ajusta automáticamente
        los precios por splits y dividendos. Sin esto, un split 2-por-1
        haría que el precio cayera a la mitad de un día para otro y el
        agente lo interpretaría como un crash falso.
      - Cualquier rango de fechas es aceptable: si un ticker no existía
        en parte del rango (ej. IBIT antes de enero 2024), retorna solo
        los datos disponibles y el pipeline rellena huecos con bfill/ffill.
    """

    def __init__(self):
        try:
            from curl_cffi.requests import Session as CurlSession
            self._session = CurlSession(impersonate="chrome")
            self._curl = True
        except ImportError:
            self._session = None
            self._curl = False

    def name(self) -> str:
        return "Yahoo Finance EOD (curl_cffi)" if self._curl else "Yahoo Finance EOD"

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga datos OHLCV de Yahoo Finance para un ticker.

        Parameters
        ----------
        ticker : símbolo del activo
        start_date : fecha de inicio 'YYYY-MM-DD'
        end_date  : fecha de fin 'YYYY-MM-DD'

        Returns
        -------
        DataFrame con columnas [Open, High, Low, Close, Volume] en orden ascendente.
        DataFrame vacío si no hay datos disponibles.
        """
        import yfinance as yf

        kwargs = {"start": start_date, "end": end_date, "progress": False, "auto_adjust": True}
        if self._curl:
            kwargs["session"] = self._session

        df = yf.download(ticker, **kwargs)
        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Aplanar MultiIndex si yfinance lo devuelve con columnas multinivel
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index(ascending=True)
        return df


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Fuente 2: Datos en tiempo real (placeholder)
# # # # # # # # # # # # # # # # # # # # # # # # # 

class LiveSource(DataSource):
    """
    Fuente de datos en tiempo real (intraday o tick-by-tick).

    Placeholder: la interfaz está definida pero la implementación concreta
    depende del proveedor (Alpaca, Interactive Brokers, Polygon.io, etc.).

    Para activar tiempo real:
      1. Subclasificar LiveSource e implementar get_ohlcv().
      2. Implementar get_latest_bar() para inferencia en tiempo real.
      3. Pasar source=MiLiveSource(api_key=...) a generate_dataset().

    La separación de responsabilidades garantiza que el entorno de entrenamiento
    y los baselines no necesitan modificarse para soportar tiempo real.

    Ejemplo de integración con Alpaca:
    -----------------------------------
    from alpaca_trade_api.rest import REST

    class AlpacaSource(LiveSource):
        def __init__(self, api_key, secret_key, base_url):
            super().__init__(proveedor="Alpaca")
            self._api = REST(api_key, secret_key, base_url)

        def get_ohlcv(self, ticker, start_date, end_date):
            bars = self._api.get_bars(ticker, '1Day', start_date, end_date).df
            bars = bars.rename(columns={'open':'Open','high':'High',
                                        'low':'Low','close':'Close','volume':'Volume'})
            return bars[['Open','High','Low','Close','Volume']]

        def get_latest_bar(self, ticker):
            bar = self._api.get_latest_bar(ticker)
            return pd.Series({'Open': bar.o, 'High': bar.h,
                               'Low': bar.l, 'Close': bar.c, 'Volume': bar.v})
    """
    # todo: implementar conexión a datos en RealTime

    def __init__(self, proveedor: str = "pendiente", **kwargs):
        self._proveedor = proveedor
        self._config = kwargs

    def name(self) -> str:
        return f"LiveSource [{self._proveedor}] — pendiente de implementación"

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Lanza error hasta que se implemente en una subclase concreta."""
        raise NotImplementedError(
            f"LiveSource '{self._proveedor}' no está implementada todavía. "
            "Subclasifica LiveSource e implementa get_ohlcv()."
        )

    def get_latest_bar(self, ticker: str) -> pd.Series:
        """
        Retorna el último bar disponible para un activo.

        Útil para inferencia en producción: alimentar el modelo con datos frescos
        sin tener que descargar todo el histórico.
        """
        raise NotImplementedError(
            "Implementar en la subclase concreta del proveedor."
        )


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Fuente 3: CSV local (tests y reproducibilidad)
# # # # # # # # # # # # # # # # # # # # # # # # # 

class CsvSource(DataSource):
    """
    Carga datos OHLCV desde CSVs locales.

    Útil para:
      - Tests unitarios con datos fijos (reproducibilidad garantizada)
      - Entornos sin acceso a internet
      - Datos propietarios ya descargados

    Espera un CSV por activo con nombre '{ticker}.csv' en el directorio base,
    con columnas Open, High, Low, Close, Volume e índice de fechas.

    Parameters
    ----------
    directory : ruta al directorio con los CSVs (por defecto 'data/raw')
    """

    def __init__(self, directory: str = "data/raw"):
        self._dir = directory

    def name(self) -> str:
        return f"CSV local ({self._dir})"

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Lee datos OHLCV de un CSV local y filtra por rango de fechas.

        Parameters
        ----------
        ticker     : símbolo del activo (busca {ticker}.csv en el directorio)
        start_date : fecha de inicio 'YYYY-MM-DD'
        end_date   : fecha de fin 'YYYY-MM-DD'

        Returns
        -------
        DataFrame filtrado con columnas [Open, High, Low, Close, Volume]
        """
        import os
        path = os.path.join(self._dir, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró {path}. "
                "Asegúrate de que el CSV tiene columnas Open, High, Low, Close, Volume."
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)

        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        return df.loc[mask, ["Open", "High", "Low", "Close", "Volume"]]
