"""
Abstracción de fuentes de datos para el pipeline de features.

Desacopla la lógica de features del origen concreto de los datos,
permitiendo cambiar de datos históricos EOD a datos en tiempo real
sin modificar ni el entorno de entrenamiento ni los baselines.

Fuentes disponibles:
  - HistoricalSource : Yahoo Finance (yfinance) — datos EOD. Modo actual.
  - LiveSource       : placeholder para datos en tiempo real (WebSocket / REST).
  - CsvSource        : carga desde CSVs locales (tests y reproducibilidad).

Interfaz común (todas las clases implementan el mismo método):

    get_ohlcv(ticker, start_date, end_date) -> pd.DataFrame
        Retorna DataFrame con columnas: Open, High, Low, Close, Volume
        Índice: DatetimeIndex en orden ascendente

Para activar una fuente en el pipeline, pasar source=LiveSource(...) a
generate_dataset(). El resto del pipeline (features, entorno, baselines)
no necesita modificarse.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# Interfaz base
# ─────────────────────────────────────────────────────────────────────────────
class DataSource(ABC):
    """Contrato que toda fuente de datos debe cumplir."""

    @abstractmethod
    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retorna datos OHLCV para un activo en el rango solicitado.

        Parameters
        ----------
        ticker     : símbolo del activo (ej. 'IVV')
        start_date : fecha de inicio en formato 'YYYY-MM-DD'
        end_date   : fecha de fin en formato 'YYYY-MM-DD'

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

    # Aliases retrocompatibles (el pipeline actual usa estos nombres)
    def obtener_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Alias de get_ohlcv() para retrocompatibilidad con el pipeline."""
        return self.get_ohlcv(ticker, start_date, end_date)

    def nombre(self) -> str:
        """Alias de name() para retrocompatibilidad con el pipeline."""
        return self.name()


# ─────────────────────────────────────────────────────────────────────────────
# Fuente 1: Datos históricos EOD (Yahoo Finance)
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalSource(DataSource):
    """
    Fuente de datos históricos End-Of-Day desde Yahoo Finance.

    Usa curl_cffi si está disponible (entornos corporativos con proxy SSL);
    si no, usa la sesión estándar de yfinance.

    Esta es la fuente por defecto del pipeline. Soporta cualquier rango de
    fechas: si un ticker no tiene datos en parte del rango (ej. IBIT antes
    de 2024), retorna solo los datos disponibles y el pipeline se encarga
    de rellenar los huecos con bfill/ffill.
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
        ticker     : símbolo del activo
        start_date : fecha de inicio 'YYYY-MM-DD'
        end_date   : fecha de fin 'YYYY-MM-DD'

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


# ─────────────────────────────────────────────────────────────────────────────
# Fuente 2: Datos en tiempo real (placeholder)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Fuente 3: CSV local (tests y reproducibilidad)
# ─────────────────────────────────────────────────────────────────────────────

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
