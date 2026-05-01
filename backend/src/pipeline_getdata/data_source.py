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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Proxies de cripto-ETFs por sus subyacentes
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
# Justificacion academica:
#   IBIT (iShares Bitcoin Trust, BlackRock) cotiza desde el 11-ene-2024.
#   ETHA (iShares Ethereum Trust) cotiza desde el 23-jul-2024.
#   Ambos son ETFs spot regulados que replican el precio del activo subyacente
#   con un tracking error muy reducido (fees ~0.25 % anuales, retraso de minutos
#   en la cotizacion). El comportamiento estadistico relevante para el
#   entrenamiento de un agente DRL —volatilidad, correlaciones, ciclos,
#   regimenes— es practicamente identico al del subyacente.
#
#   Para garantizar un historico de entrenamiento suficiente (multiples
#   regimenes de mercado, incluidos crashes 2018, COVID 2020, bear cripto 2022),
#   este modulo sustituye IBIT y ETHA por sus subyacentes BTC-USD y ETH-USD
#   en el periodo anterior a la fecha de lanzamiento del ETF correspondiente.
#   La sustitucion se acompana de:
#       (1) Un escalado multiplicativo (factor k) que hace coincidir el ultimo
#           precio del proxy con el primer precio real del ETF, eliminando
#           saltos artificiales en el dia del splice.
#       (2) Forward-fill del primer volumen real del ETF hacia atras durante
#           el periodo de proxy. El volumen del subyacente esta en una unidad
#           distinta (BTC se mueve en miles de millones de USD agregados de
#           todos los exchanges, no comparable con volumen del ETF en NYSE).
#           Mantener el volumen real del ETF como constante hacia atras es la
#           opcion mas honesta: no inventa volumen historico inexistente, no
#           rompe el calculo de iliquidez de Amihud, y refleja el orden de
#           magnitud realista del activo final.
#       (3) Logging explicito del splice point y del factor de escala para
#           trazabilidad y reproducibilidad.
#
# Si se desea desactivar el proxy (por ejemplo, para un experimento limitado al
# periodo post-lanzamiento), basta con eliminar la entrada correspondiente del
# CRYPTO_PROXY_MAP. La logica de splice solo se activa cuando el ticker tiene
# entrada en este mapa.

CRYPTO_PROXY_MAP = {
    'IBIT': 'BTC-USD',
    'ETHA': 'ETH-USD',
}

ETF_LAUNCH_DATES = {
    'IBIT': '2024-01-11',  # iShares Bitcoin Trust — primer dia de cotizacion
    'ETHA': '2024-07-23',  # iShares Ethereum Trust — primer dia de cotizacion
}


def _splice_etf_with_proxy(etf_df: pd.DataFrame,
                           proxy_df: pd.DataFrame,
                           launch_date: pd.Timestamp,
                           ticker: str,
                           proxy_ticker: str) -> pd.DataFrame:
    """
    Fusiona la serie real de un ETF cripto con la de su subyacente como proxy.

    Estrategia (justificada al inicio del modulo):
      1. Tomar el primer precio real del ETF en o despues de launch_date.
      2. Tomar el ultimo precio del proxy estrictamente antes de launch_date.
      3. Calcular factor de escala k = first_real_price / last_proxy_price.
      4. Multiplicar OHLC del proxy por k para alinearlo con el ETF.
      5. Forward-fill el primer volumen real del ETF hacia atras durante el
         periodo de proxy (Opcion B documentada en el modulo).
      6. Concatenar las dos series y devolver.

    Si alguno de los DataFrames esta vacio, devuelve el otro sin transformacion.
    """
    if etf_df.empty and proxy_df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    if etf_df.empty:
        # Solo tenemos proxy: lo devolvemos sin escalar (no hay ancla a ETF real)
        print(f"  [PROXY] {ticker}: sin datos reales del ETF, usando "
              f"{proxy_ticker} sin escalar.")
        return proxy_df.copy()
    if proxy_df.empty:
        # Solo tenemos ETF real: no hace falta proxy
        return etf_df.copy()

    # Quedarnos con la parte del proxy estrictamente anterior al lanzamiento
    proxy_pre = proxy_df[proxy_df.index < launch_date].copy()
    if proxy_pre.empty:
        # No hay periodo previo que rellenar
        return etf_df.copy()

    # Anclas para el escalado
    last_proxy_price = float(proxy_pre['Close'].iloc[-1])
    first_real_price = float(etf_df['Close'].iloc[0])
    if last_proxy_price <= 0:
        print(f"  [PROXY] {ticker}: ultimo precio del proxy <= 0; "
              f"sin escalado (factor k = 1.0).")
        k = 1.0
    else:
        k = first_real_price / last_proxy_price

    # Escalar OHLC del proxy. Volume se trata aparte (Opcion B: forward-fill).
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in proxy_pre.columns]
    proxy_scaled = proxy_pre.copy()
    proxy_scaled[price_cols] = proxy_scaled[price_cols] * k

    # Forward-fill (hacia atras) del primer volumen real del ETF en el proxy.
    # Asi mantenemos un orden de magnitud realista del activo final, en lugar
    # de usar el volumen agregado de BTC/ETH spot (que esta en otra unidad).
    if 'Volume' in etf_df.columns and not etf_df['Volume'].empty:
        first_real_volume = float(etf_df['Volume'].iloc[0])
        proxy_scaled['Volume'] = first_real_volume
    else:
        proxy_scaled['Volume'] = np.nan

    spliced = pd.concat([proxy_scaled, etf_df]).sort_index(ascending=True)
    # Si por casualidad hay duplicados en el indice, conservar el real (etf_df)
    spliced = spliced[~spliced.index.duplicated(keep='last')]

    n_proxy = len(proxy_scaled)
    n_real = len(etf_df)
    print(f"  [PROXY] {ticker}: splice aplicado.")
    print(f"      Proxy ({proxy_ticker}): {proxy_scaled.index[0].date()} -> "
          f"{proxy_scaled.index[-1].date()} ({n_proxy} dias)")
    print(f"      Real ({ticker}): {etf_df.index[0].date()} -> "
          f"{etf_df.index[-1].date()} ({n_real} dias)")
    print(f"      Factor de escala k = {k:.6f}  "
          f"(last_proxy={last_proxy_price:.4f}, first_real={first_real_price:.4f})")
    print(f"      Volumen pre-launch: forward-fill del primer real "
          f"({first_real_volume:,.0f}) hacia atras (Opcion B).")
    return spliced


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

        Si el ticker es un cripto-ETF registrado en CRYPTO_PROXY_MAP (IBIT,
        ETHA), aplica la sustitucion por su subyacente (BTC-USD, ETH-USD)
        en el periodo previo al lanzamiento del ETF. Ver justificacion al
        inicio del modulo.

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
        # Caso especial: cripto-ETFs con proxy declarado.
        if ticker in CRYPTO_PROXY_MAP:
            return self._fetch_with_crypto_proxy(ticker, start_date, end_date)
        # Caso general: descarga directa.
        return self._fetch_raw(ticker, start_date, end_date)

    def _fetch_raw(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga directa de Yahoo Finance sin aplicar logica de proxy.
        Es la implementacion historica del get_ohlcv, extraida como helper
        para que pueda reutilizarse desde _fetch_with_crypto_proxy.
        """
        import yfinance as yf

        kwargs = {"start": start_date, "end": end_date, "progress": False, "auto_adjust": True}
        if self._curl:
            kwargs["session"] = self._session

        df = yf.download(ticker, **kwargs)
        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index(ascending=True)
        return df

    def _fetch_with_crypto_proxy(self, ticker: str,
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga el ETF y, si el rango incluye periodo pre-lanzamiento, lo
        empalma con el subyacente (BTC-USD para IBIT, ETH-USD para ETHA).

        Comportamiento:
          - Si start_date >= launch_date del ETF: descarga solo el ETF real.
            No se necesita proxy.
          - Si start_date < launch_date: descarga el ETF real desde
            launch_date hasta end_date y el subyacente desde start_date hasta
            launch_date - 1; los empalma con escalado en `_splice_etf_with_proxy`.
        """
        proxy_ticker = CRYPTO_PROXY_MAP[ticker]
        launch_date = pd.Timestamp(ETF_LAUNCH_DATES[ticker])
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        # Si todo el rango es posterior al lanzamiento, no hace falta proxy.
        if start_ts >= launch_date:
            print(f"  [PROXY] {ticker}: rango {start_date} -> {end_date} es "
                  f"posterior al lanzamiento ({launch_date.date()}); "
                  f"sin sustitucion, descarga directa del ETF real.")
            return self._fetch_raw(ticker, start_date, end_date)

        # Caso normal: parte del rango es pre-launch -> aplicar splice.
        # Descargamos el ETF real (puede estar vacio si end_date < launch_date).
        etf_real_start = max(start_ts, launch_date).strftime('%Y-%m-%d')
        etf_df = self._fetch_raw(ticker, etf_real_start, end_date) \
            if end_ts >= launch_date else pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"]
            )

        # Descargamos el subyacente sobre el rango pre-launch (con un dia
        # extra de margen para el ancla del splice).
        proxy_end = (launch_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        proxy_df = self._fetch_raw(proxy_ticker, start_date, proxy_end)

        return _splice_etf_with_proxy(
            etf_df=etf_df,
            proxy_df=proxy_df,
            launch_date=launch_date,
            ticker=ticker,
            proxy_ticker=proxy_ticker,
        )


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
