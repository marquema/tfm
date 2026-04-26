"""
Screener de mercado: selección automática de candidatos de inversión.

Fundamentación teórica:
  La selección del universo de inversión es el primer paso en cualquier estrategia
  cuantitativa (Grinold & Kahn, 2000, "Active Portfolio Management"). Operar sobre
  un universo demasiado amplio diluye la señal; sobre uno demasiado pequeño introduce
  sesgo de selección. El screener implementa el enfoque estándar de la industria
  de gestión cuantitativa (hedge funds como AQR, Two Sigma, Renaissance):

    1. Universo base amplio: S&P 500 (~500 activos). Se elige este índice porque
       garantiza liquidez mínima (criterio de inclusión del propio índice) y
       cobertura sectorial completa según la clasificación GICS.

    2. Filtros cuantitativos secuenciales — cada filtro tiene base teórica:

       - Historial mínimo (2 años): PPO necesita un mínimo de datos para aprender
         patrones temporales. Con menos de 2 años no hay suficientes regímenes de
         mercado observados (documentacion: López de Prado, 2018, cap. 2).

       - Liquidez mínima ($5M/día): la Hipótesis de Mercados Eficientes (documentacion: Fama, 1970)
         se cumple mejor en activos líquidos. En activos ilíquidos, los precios no
         reflejan toda la información disponible y el slippage (diferencia entre
         precio esperado y ejecutado) erosiona cualquier alpha encontrado.

       - Volatilidad en rango (5%-80%): filtro basado en la teoría de la relación
         señal-ruido. Activos con volatilidad < 5% anualizada (ej. money markets)
         no ofrecen suficiente dispersión de retornos para que PPO aprenda — la señal
         es indistinguible del ruido. Activos con volatilidad > 80% 
         tienen una ratio señal/ruido tan baja que el aprendizaje no converge.

       - Sharpe rolling positivo: proxy del factor de momentum (documentacion: Jegadeesh & Titman, 1993).
         Activos con Sharpe negativo reciente tienen tendencia bajista — incluirlos
         fuerza al agente a aprender a shortear (no soportado) o a asignarles peso 0,
         desperdiciando capacidad del observation space.
         Para entender por qué esto es un problema, ver el glosario al final
         (entradas "Posición corta" y "Observation space").

       - Diversificación sectorial (máx. 3 por sector GICS): implementación directa
         del principio de diversificación de Markowitz (documentacion: 1952). Sin este filtro, el
         screener podría seleccionar 15 activos tecnológicos que se mueven de forma
         casi idéntica (correlación > 0.9), anulando los beneficios de diversificación.
         La clasificación GICS (Global Industry Classification Standard) de MSCI/S&P
         es el estándar de la industria para categorización sectorial.

    3. Ranking final por Sharpe: los candidatos que pasan todos los filtros se ordenan
       por Sharpe rolling descendente. Esto implementa un factor de momentum cross-sectional
       — se seleccionan los activos con mejor rendimiento ajustado por riesgo reciente.
       Ver el glosario al final ("Momentum cross-sectional") para una explicación
       intuitiva de qué significa exactamente y por qué funciona en ventanas de 6-12 meses.

Fuente del universo:
  Wikipedia publica la lista actualizada de componentes del S&P 500 en una tabla HTML.
  Se parsea directamente con pandas.read_html() — sin necesidad de API key ni scraping.

Uso:
  screener = MarketScreener()
  candidates = screener.run(start_date='2020-01-01', end_date='2026-01-01', top_n=15)
  # candidates = ['AAPL', 'MSFT', 'JNJ', ...]  → pasa a generate_dataset()

─────────────────────────────────────────────────────────────────────────────
Glosario divulgativo de términos financieros usados en este módulo
─────────────────────────────────────────────────────────────────────────────

Alpha:
    El "extra" de rentabilidad que aporta mi estrategia más allá de lo que
    da el mercado por sí solo. Si el S&P 500 hace +10 % en un año y mi
    cartera hace +12 %, alpha es +2 puntos. Si hace +8 %, alpha es −2
    (peor que estar pasivo: habría salido más barato comprar un ETF y
    no hacer nada). En el TFM, el agente PPO busca generar alpha frente a
    los baselines (documentación: Equal Weight, Buy & Hold, 60/40, Markowitz).

Bid-ask spread:
    En cualquier mercado siempre hay dos precios:
      - Bid: el máximo que alguien está dispuesto a pagar si vendo.
      - Ask: el mínimo al que alguien está dispuesto a vender si compro.
    El spread es la diferencia entre ambos. 
    
Market impact:
    Si quiero comprar 1000 acciones pero solo hay 500 disponibles al
    precio actual, las siguientes 500 se pagan más caras. Mi orden
    mueve el precio en mi contra. 

Por qué bid-ask + market impact se "comen el alpha":
    Si el modelo encuentra un activo que va a subir un 0.5 % al
    día siguiente — eso sería alpha. Pero al entrar se pierde 0.3 % en
    spread y 0.2 % más en market impact: total −0.5 % solo por operar. 
    Alpha se evapora antes de capturarlo. Por eso filtramos por liquidez

Posición corta (short):
    Operación que gana dinero cuando un activo cae (lo opuesto a comprar).
    El entorno PPO de mi TFM solo permite comprar (long-only): los pesos
    de la cartera son ≥ 0. Si incluyéramos un activo con tendencia bajista
    clara, el agente solo podría asignarle peso 0 — espacio desperdiciado.

Observation space:
    Lo que el agente "ve" como entrada de la red neuronal: features de
    cada activo (RSI, MACD, momentum, volatilidad, etc.). Si tienes 15
    activos × 5 features = 75 entradas, esas son neuronas dedicadas. Si un
    activo va a estar siempre con peso 0 porque su Sharpe es negativo,
    estamos ocupando 5 neuronas para nada — mejor llenar ese hueco con un
    activo prometedor.

Momentum:
    Efecto empírico observado durante décadas: lo que ha estado subiendo
    durante un tiempo medio (6-12 meses) tiende a seguir subiendo, y lo
    que ha estado cayendo tiende a seguir cayendo. Documentación: por
    Jegadeesh & Titman (1993) y confirmado en docenas de estudios.

Momentum cross-sectional:
    Comparar el momentum entre activos en un mismo momento, no un activo
    consigo mismo en el tiempo. "De todos los candidatos disponibles, me
    quedo con los que mejor han ido en relación al resto en los últimos
    12 meses". Eso es exactamente lo que hace el ranking final del
    screener al ordenar por Sharpe rolling descendente.

Por qué la ventana 6-12 meses funciona y otras no:
    - Ventanas cortas (1 día - 1 mes): predomina la reversión a la media.
      Lo que sube mucho hoy tiende a corregir mañana. Comprar lo que sube
      perjudica.
    - Ventanas medias (6 - 12 meses): aquí domina el momentum. Es la "zona
      dulce" del efecto. Por eso se usa 252 días = 1 año.
    - Ventanas largas (3 - 5 años): vuelve a aparecer la reversión. Lo que
      sube años acaba estando sobrevalorado y corrigiendo (es lo que
      explota el "value investing" estilo Buffett).
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Optional

try:
    from curl_cffi.requests import Session as CurlSession
    _CURL_AVAILABLE = True
except ImportError:
    _CURL_AVAILABLE = False


# ─────────────────────────────────────────────
# Obtención del universo base
# ─────────────────────────────────────────────

_SP500_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sp500_universe.csv')


def fetch_sp500_tickers() -> pd.DataFrame:
    """
    Descarga la lista actual de componentes del S&P 500 desde Wikipedia.

    Usa curl_cffi con impersonación de navegador Chrome para evitar el bloqueo
    HTTP 403 que Wikipedia aplica a peticiones sin cabecera de navegador real
    (como urllib o requests estándar). Es la misma sesión resiliente que usamos
    para descargar datos de Yahoo Finance en entornos corporativos.

    El S&P 500 se elige como universo base porque su propio criterio de inclusión
    ya garantiza: capitalización > $14.6B, volumen suficiente, domicilio en EE.UU.,
    y al menos 12 meses de historial cotizado. Esto nos ahorra filtros.

    La clasificación sectorial GICS (Global Industry Classification Standard,
    desarrollada por MSCI y S&P Dow Jones) divide el mercado en 11 sectores.
    Se usa para el filtro de diversificación sectorial del screener.

    Si la descarga tiene éxito, guarda una copia local en data/sp500_universe.csv
    como cache para ejecuciones offline futuras.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: ticker, name, sector, sub_industry.
        Ordenado alfabéticamente por ticker.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    print(f"  Descargando lista del S&P 500 desde Wikipedia...")

    # Intentar con curl_cffi
    html = None
    if _CURL_AVAILABLE:
        try:
            session = CurlSession(impersonate="chrome")
            session.verify = False
            response = session.get(url)
            if response.status_code == 200:
                html = response.text
                print(f"  Descarga exitosa con curl_cffi.")
        except Exception as e:
            print(f"  [AVISO] curl_cffi falló: {e}. Intentando con requests...")

    # Fallback: requests con User-Agent
    if html is None:
        try:
            import requests
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            html = resp.text
            print(f"  Descarga exitosa con requests.")
        except Exception as e:
            print(f"  [AVISO] requests falló: {e}. Buscando cache local...")

    # Último recurso: cache local
    if html is None:
        if os.path.exists(_SP500_CSV_PATH):
            print(f"  Usando cache local: {_SP500_CSV_PATH}")
            df = pd.read_csv(_SP500_CSV_PATH)
            print(f"  {len(df)} componentes cargados desde cache.")
            return df.sort_values('ticker').reset_index(drop=True)
        else:
            raise ConnectionError(
                "No se pudo descargar la lista del S&P 500 ni existe cache local. "
                "Verifica la conexión a internet."
            )

    # Parsear HTML a DataFrame
    from io import StringIO
    tables = pd.read_html(StringIO(html))
    df = tables[0]

    df = df.rename(columns={
        'Symbol': 'ticker',
        'Security': 'name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'sub_industry',
    })

    # Limpiar tickers: BRK.B → BRK-B (formato que espera Yahoo Finance)
    df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)

    cols = ['ticker', 'name', 'sector', 'sub_industry']
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values('ticker').reset_index(drop=True)

    # Guardar cache para ejecuciones offline
    try:
        os.makedirs(os.path.dirname(_SP500_CSV_PATH), exist_ok=True)
        df.to_csv(_SP500_CSV_PATH, index=False)
        print(f"  Cache actualizado: {_SP500_CSV_PATH}")
    except Exception:
        pass  # No es crítico si no puede guardar cache

    print(f"  {len(df)} componentes del S&P 500 obtenidos.")
    return df


####################
# Screener principal
####################

class MarketScreener:
    """
    Selecciona automáticamente los mejores candidatos del S&P 500 para
    entrenar al agente PPO.

    Idea intuitiva de selección:
        Empezamos con ~500 acciones (todo el S&P 500) y vamos descartando las
        que no cumplen criterios objetivos hasta quedarnos con un puñado
        (top_n, por defecto 15). Cada filtro responde a una pregunta concreta, que se ve al principio
            de este fichero:

            "¿Tiene suficiente histórico para que PPO aprenda?"
            "¿Es líquido? (¿se puede comprar y vender sin mover el precio?)"
            "¿Tiene volatilidad razonable? (ni inerte ni caos)"
            "¿Está mostrando una tendencia rentable últimamente?"
            "¿Estamos sobrecargando un sector concreto?"

        Los que superan los cinco filtros se ordenan por Sharpe rolling
        (rentabilidad ajustada por riesgo del último año) y se eligen los
        primeros respetando la diversificación sectorial.

    Por qué es importante:
        El universo de inversión es la decisión más importante en cualquier
        estrategia cuantitativa (documentacion: Grinold & Kahn, 2000). Operar sobre demasiados
        activos diluye la señal; sobre pocos, sesga el resultado. El screener
        automatiza esa decisión con criterios reproducibles, evitando sesgo
        humano (data snooping) y dejando trazabilidad para la memoria del TFM.

    Patrón de selección:
        Los filtros se aplican de barato a caro y de poco a muy restrictivo.
        Primero se descarta lo que ni siquiera tiene datos; al final se aplica
        la diversificación sectorial (la más restrictiva). Esto evita gastar
        cómputo evaluando activos que iban a caer de todas formas.

    Parameters
    ----------
    min_history_years : float
        Años mínimos de historial requerido (por defecto 2).
        Justificación: PPO necesita observar al menos dos ciclos estacionales
        completos para distinguir patrones reales de ruido aleatorio
        (documentacion: López de Prado, 2018).
    min_daily_volume_usd : float
        Volumen medio diario mínimo en dólares (por defecto $5M).
        Justificación: en activos poco líquidos, el bid-ask spread y el
        market impact se comen cualquier alpha que el modelo haya encontrado
        (documentacion: Almgren & Chriss, 2001, "Optimal Execution").

        Para entender qué son bid-ask spread, market impact y alpha — y por
        qué juntos se comen las ganancias — ver el glosario al inicio del
        módulo. Resumen: cada vez que entramos y salgo en un activo ilíquido,
        pierdo algo en el spread y muevo yo mismo el precio en mi contra
        al operar. Si tu edge esperado es del 0.5 % al día y los costes de
        ejecución suman 0.5 %, has trabajado para nada.
    vol_range : tuple[float, float]
        Rango aceptable de volatilidad anualizada (por defecto 5%-80%).
        Justificación: si la volatilidad es muy baja (< 5%), el activo apenas
        se mueve y PPO no tiene señal que aprender — es ruido. Si es muy
        alta (> 80%), el ruido domina sobre la señal y el entrenamiento no
        converge en tiempos razonables.
    max_per_sector : int
        Máximo de activos del mismo sector GICS en los candidatos finales
        (por defecto 3).
        Justificación: principio de diversificación de Markowitz (1952).
        Activos del mismo sector tienen correlaciones de 0.6-0.8 — coger
        15 tecnológicas se parecería más a apostar todo a una opción, no sería cartera diversificada.
    sharpe_window : int
        Días de la ventana rolling para calcular Sharpe (por defecto 252 = 1 año).
        Justificación: el efecto momentum cross-sectional (documentacion: Jegadeesh & Titman,
        1993) se manifiesta de forma robusta en ventanas de 6-12 meses.
        252 días es el estándar académico y de la industria.

        Para entender por qué precisamente 6-12 meses funciona (y no 1 día
        ni 5 años), ver el glosario al inicio del módulo. Resumen: en
        ventanas cortas predomina la "reversión a la media" (lo que sube
        hoy corrige mañana); en ventanas largas también aparece reversión
        (lo que sube años acaba sobrevalorado). El intervalo medio de 6-12
        meses es la "zona dulce" donde el momentum domina sobre cualquier
        otro efecto.
    """

    def __init__(self,
                 min_history_years: float = 2.0,
                 min_daily_volume_usd: float = 5_000_000,
                 vol_range: tuple = (0.05, 0.80),
                 max_per_sector: int = 3,
                 sharpe_window: int = 252):
        self.min_history_years = min_history_years
        self.min_daily_volume_usd = min_daily_volume_usd
        self.vol_range = vol_range
        self.max_per_sector = max_per_sector
        self.sharpe_window = sharpe_window

        # Sesión HTTP resiliente: en redes corporativas Yahoo Finance suele
        # bloquear los User-Agent estándar de requests. curl_cffi se hace
        # pasar por Chrome (TLS fingerprinting incluido) y sortea el bloqueo.
        # verify=False evita errores con proxies que reescriben certificados.
        self._session = None
        if _CURL_AVAILABLE:
            self._session = CurlSession(impersonate="chrome")
            self._session.verify = False

    def _download_basic_data(self, ticker: str,
                              start_date: str, end_date: str) -> Optional[dict]:
        """
        Descarga el histórico de un ticker y calcula las métricas de filtro.

        Solo nos interesan cuatro números por activo: cuántos años de
        historia tiene, su volumen medio en dólares, su volatilidad
        anualizada y su Sharpe rolling. Con eso bastan los filtros de
        selección.

        Si el ticker no devuelve datos (delistado, ticker inválido, error
        de red puntual) o devuelve menos de 60 días, retornamos None y el
        screener simplemente lo descarta.

        Parameters
        ----------
        ticker : símbolo del activo (ej. 'AAPL', 'MSFT').
        start_date : fecha inicio 'YYYY-MM-DD'.
        end_date: fecha fin 'YYYY-MM-DD'.

        Returns
        -------
        dict con: ticker, n_days, n_years, avg_volume_usd, annual_vol,
                  sharpe_rolling, last_price.
        None si no hay datos suficientes.
        """
        try:
            kwargs = {"start": start_date, "end": end_date,
                      "progress": False, "auto_adjust": True}
            if self._session:
                kwargs["session"] = self._session

            df = yf.download(ticker, **kwargs)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 60:
                return None

            # Métricas básicas
            n_days = len(df)
            n_years = n_days / 252

            returns = df['Close'].pct_change().dropna()
            avg_volume_usd = (df['Volume'] * df['Close']).mean()
            annual_vol = returns.std() * np.sqrt(252)

            # Sharpe rolling sobre la ventana configurada
            window = min(self.sharpe_window, len(returns) - 1)
            if window > 20:
                recent_returns = returns.tail(window)
                sharpe = (recent_returns.mean() / (recent_returns.std() + 1e-8)) * np.sqrt(252)
            else:
                sharpe = 0.0

            return {
                'ticker': ticker,
                'n_days': n_days,
                'n_years': round(n_years, 1),
                'avg_volume_usd': round(avg_volume_usd),
                'annual_vol': round(annual_vol, 4),
                'sharpe_rolling': round(sharpe, 3),
                'last_price': round(float(df['Close'].iloc[-1]), 2),
            }

        except Exception:
            return None

    def run(self, start_date: str, end_date: str,
            top_n: int = 15,
            universe_df: pd.DataFrame = None,
            force_include: list = None) -> dict:
        """
        Ejecuta el criterio completo de selección y devuelve los candidatos finales.

        Flujo paso a paso:
            1. Conseguir el universo base (S&P 500 desde Wikipedia, o el que
               le pasen).
            2. Descargar histórico de precios y volumen para cada ticker
               (esta es la parte más lenta — cientos de llamadas a Yahoo Finance).
            3. Calcular métricas básicas por activo: años de historia, volumen
               medio en dólares, volatilidad anualizada y Sharpe rolling.
            4. Aplicar los cinco filtros secuenciales en orden de restrictividad.
            5. Ordenar los supervivientes por Sharpe descendente.
            6. Seleccionar top_n respetando el límite de activos por sector,
               garantizando que los `force_include` (IBIT, ETHA, IVV, BND)
               siempre estén en el resultado.

        Parameters
        ----------
        start_date : str
            Fecha de inicio del periodo de evaluación ('YYYY-MM-DD').
            Las métricas (volumen, volatilidad, Sharpe) se calculan sobre
            datos comprendidos entre esta fecha y end_date.
        end_date : str
            Fecha de fin del periodo de evaluación.
        top_n : int
            Número máximo de candidatos a devolver. El resultado real puede
            ser menor si los filtros descartan demasiados activos.
        universe_df : pd.DataFrame, optional
            DataFrame con columnas 'ticker' y 'sector'. Si es None, se
            descarga el S&P 500 actualizado desde Wikipedia.
        force_include : list[str], optional
            Tickers que SIEMPRE deben aparecer en el resultado, aunque no
            cumplan los filtros (ej. IBIT y ETHA, que no están en el S&P 500
            pero son obligatorios por requisito del TFM).

        Returns
        -------
        dict con:
          'candidates' : list[str] — tickers seleccionados, ordenados por
                           Sharpe rolling descendente.
          'details': pd.DataFrame con métricas completas (ticker,
                           sector, sharpe_rolling, annual_vol, avg_volume_usd,
                           n_years, last_price). Listo para mostrar en el
                           panel de admin del frontend.
          'filtered_out' : dict — cuántos activos se descartaron en cada
                           filtro. Útil para diagnosticar por qué un ticker
                           esperado no apareció en el resultado.
        """
        if force_include is None:
            force_include = []

        # ── PASO 1: obtener el universo base ────────────────────────────────
        # Lista de partida sobre la que aplicar la restricción. Por defecto, los
        # ~500 componentes del S&P 500 (descarga desde Wikipedia, con cache
        # local como fallback).
        if universe_df is None:
            universe_df = fetch_sp500_tickers()

        all_tickers = universe_df['ticker'].tolist()
        sectors = dict(zip(universe_df['ticker'], universe_df.get('sector', ['Unknown'] * len(universe_df))))

        print(f"\n{'='*60}")
        print(f"MARKET SCREENER — {len(all_tickers)} activos a evaluar")
        print(f"{'='*60}")

        # ── PASO 2: descargar precios y calcular métricas básicas ───────────
        # Una llamada a Yahoo Finance por ticker. Es la fase más lenta
        # (depende del ancho de banda + límites de la API). Por cada ticker
        # se calculan: años de historial, volumen medio en dólares, volatilidad
        # anualizada y Sharpe rolling. Si Yahoo no devuelve datos suficientes
        # (delistado, ticker inválido, etc.) se descarta.
        print(f"\n  Descargando datos de {len(all_tickers)} activos (esto puede tardar unos minutos)...")
        metrics = []
        for i, ticker in enumerate(all_tickers):
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(all_tickers)} procesados")
            data = self._download_basic_data(ticker, start_date, end_date)
            if data is not None:
                data['sector'] = sectors.get(ticker, 'Unknown')
                metrics.append(data)

        df = pd.DataFrame(metrics)
        n_with_data = len(df)
        print(f"  {n_with_data} activos con datos válidos (de {len(all_tickers)} intentados)")

        if df.empty:
            return {'candidates': force_include, 'details': pd.DataFrame(),
                    'filtered_out': {'no_data': len(all_tickers)}}

        # ── PASO 3: aplicar filtros del embudo ──────────────────────────────
        # Cuatro filtros secuenciales. Cada uno descarta los que no cumplen,
        # y `filtered_out` registra cuántos cayeron en cada paso para poder
        # diagnosticar el resultado a posteriori.
        filtered_out = {}

        # Filtro 1 — Historial mínimo:
        # PPO necesita ver al menos dos ciclos completos de mercado. Activos
        # recién listados (IPOs del último año) no dan suficiente señal.
        mask = df['n_years'] >= self.min_history_years
        filtered_out['history_too_short'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro historial (>= {self.min_history_years} años): {len(df)} pasan")

        # Filtro 2 — Liquidez:
        # Si el volumen diario es bajo, los costes de ejecución (bid-ask spread
        # + market impact) se comen el alpha que el modelo pueda haber
        # encontrado. Mejor descartar antes que perder dinero al operar.
        # Ver glosario del módulo (Alpha, Bid-ask spread, Market impact) para
        # entender por qué un alpha teórico se evapora en activos ilíquidos.
        mask = df['avg_volume_usd'] >= self.min_daily_volume_usd
        filtered_out['low_liquidity'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro liquidez (>= ${self.min_daily_volume_usd:,.0f}/día): {len(df)} pasan")

        # Filtro 3 — Volatilidad en rango:
        # Demasiado plano (<5%) → no hay nada que aprender. Demasiado salvaje
        # (>80%) → el ruido tapa cualquier patrón y el modelo no converge.
        mask = (df['annual_vol'] >= self.vol_range[0]) & (df['annual_vol'] <= self.vol_range[1])
        filtered_out['vol_out_of_range'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro volatilidad ({self.vol_range[0]:.0%}-{self.vol_range[1]:.0%}): {len(df)} pasan")

        # Filtro 4 — Sharpe rolling positivo:
        # Solo nos interesan activos en tendencia rentable reciente. Razón:
        # el entorno PPO es long-only (no permite posiciones cortas), así que
        # ante un activo en clara tendencia bajista solo le quedan dos opciones:
        #   - asignarle peso 0 → ese hueco del observation space (las features
        #     del activo) ocupa neuronas en la red sin aportar nada útil, igual
        #     que un compañero de viaje que no paga gasolina ni hace turnos al volante;
        #   - intentar shortearlo → no puede, el entorno no lo permite.
        # Mejor descartarlo desde el screener y dejar el espacio a un activo
        # prometedor. Ver glosario del módulo (Posición corta, Observation space).
        mask = df['sharpe_rolling'] > 0
        filtered_out['negative_sharpe'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro Sharpe rolling > 0: {len(df)} pasan")

        if df.empty:
            print("  [AVISO] Ningún activo pasó todos los filtros. Retornando force_include.")
            return {'candidates': force_include, 'details': pd.DataFrame(),
                    'filtered_out': filtered_out}

        # ── PASO 4: ranking + diversificación sectorial ─────────────────────
        # Ordenamos por Sharpe rolling descendente: los mejores arriba.
        # Esto implementa un "momentum cross-sectional": de todos los activos
        # que pasaron los filtros, nos quedamos con los que mejor rendimiento
        # ajustado por riesgo han tenido en relación al resto durante el
        # último año. Es uno de los efectos más estudiados en finanzas
        # (Jegadeesh & Titman, 1993). Ver glosario del módulo.
        df = df.sort_values('sharpe_rolling', ascending=False)

        selected = []
        sector_count = {}

        # Primero, garantizar los force_include (IBIT, ETHA, IVV, BND para
        # el TFM). Estos son obligatorios y no respetan el límite de sector
        # — su justificación está en el alcance del trabajo, no en los datos.
        for t in force_include:
            if t in df['ticker'].values:
                row = df[df['ticker'] == t].iloc[0]
                selected.append(row.to_dict())
                s = row['sector']
                sector_count[s] = sector_count.get(s, 0) + 1

        # Después rellenamos hasta top_n recorriendo el ranking. Aplicamos
        # `max_per_sector` para evitar que el resultado sea, por ejemplo,
        # 10 tecnológicas + 5 financieras. La diversificación sectorial es
        # más importante para PPO que coger los 15 mejores Sharpes absolutos.
        for _, row in df.iterrows():
            if len(selected) >= top_n:
                break
            if row['ticker'] in [s['ticker'] for s in selected]:
                continue  # ya entró vía force_include
            s = row['sector']
            if sector_count.get(s, 0) >= self.max_per_sector:
                continue  # sector saturado, saltamos a buscar otro distinto
            selected.append(row.to_dict())
            sector_count[s] = sector_count.get(s, 0) + 1

        df_selected = pd.DataFrame(selected)
        candidates = df_selected['ticker'].tolist()

        print(f"\n{'='*60}")
        print(f"RESULTADO: {len(candidates)} candidatos seleccionados")
        print(f"{'='*60}")
        print(df_selected[['ticker', 'sector', 'sharpe_rolling', 'annual_vol',
                           'avg_volume_usd', 'n_years']].to_string(index=False))

        return {
            'candidates': candidates,
            'details': df_selected,
            'filtered_out': filtered_out,
        }


#todo para implmentar a futuro:
# PPO long only, pero podría ser short, es decir, apostar a corto
    #Sí, pero exige rehacer el entorno y reentrenar todo. Las piezas a tocar:

        #action_space: pasar de Box(0, 1) (simplex long-only) a Box(-1, 1) por activo, con restricción 
        # sum(|w|) ≤ leverage_limit.
        # Cálculo de PnL: ya funciona con pesos negativos sin tocar nada 
        # (return_portfolio = Σ w_i · return_i).
        # Coste de préstamo (lo que de verdad cuesta tiempo modelar): un short paga ~2-5 % anual al 
        # broker por las acciones prestadas. Hay que añadir −borrow_fee · |posición_negativa| al PnL 
        # diario para que sea realista.
        # Restricciones: hoy no todos los activos se pueden shortear ("hard-to-borrow"). En IBIT/ETHA el
        # coste de préstamo puede llegar al 20 %. Habría que modelar quién es shortable.
    
    #Riesgos para el TFM:
        # La varianza del PPO explota: el espacio de acciones es mucho mayor, PPO tarda más en converger 
        # y el clip fraction se dispara.
        # El Sharpe puede colapsar al principio porque el agente aún no sabe usar los shorts con criterio.
        # Las baselines (Equal Weight, 60/40, Markowitz) son long-only por definición → la comparativa 
        # pierde claridad: ¿estás midiendo "PPO bueno" o "shorts ayudan"?
        # Recomendación: encaja mucho mejor en el agente especulativo (GMM+KMeans) que ya tienes. 
        # El discurso del TFM gana así:
            # "PPO institucional (regulado, long-only) vs Especulador con shorts. El primero gana en Sharpe y consistencia; el segundo gana en retorno bruto en mercados con tendencias claras pero sufre MDD del −60 %. Demostramos que el control institucional añade valor."
            # Tocar el PPO actual implica reentrenar todo, recalibrar phi/gamma, rehacer walk-forward y sensitivity → 1-2 semanas de trabajo. Apuntarlo como evolución futura en la memoria es la opción 80/20 razonable a estas alturas del TFM.