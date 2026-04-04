"""
Screener de mercado: selección automática de candidatos de inversión.

Fundamentación teórica:
  La selección del universo de inversión es el primer paso en cualquier estrategia
  cuantitativa (Grinold & Kahn, 2000, "Active Portfolio Management"). Operar sobre
  un universo demasiado amplio diluye la señal; sobre uno demasiado estrecho introduce
  sesgo de selección. El screener implementa el enfoque estándar de la industria
  de gestión cuantitativa (hedge funds como AQR, Two Sigma, Renaissance):

    1. Universo base amplio: S&P 500 (~500 activos). Se elige este índice porque
       garantiza liquidez mínima (criterio de inclusión del propio índice) y
       cobertura sectorial completa según la clasificación GICS.

    2. Filtros cuantitativos secuenciales — cada filtro tiene base teórica:

       - Historial mínimo (2 años): PPO necesita un mínimo de datos para aprender
         patrones temporales. Con menos de 2 años no hay suficientes regímenes de
         mercado observados (López de Prado, 2018, cap. 2).

       - Liquidez mínima ($5M/día): la Hipótesis de Mercados Eficientes (Fama, 1970)
         se cumple mejor en activos líquidos. En activos ilíquidos, los precios no
         reflejan toda la información disponible y el slippage (diferencia entre
         precio esperado y ejecutado) erosiona cualquier alpha encontrado.

       - Volatilidad en rango (5%-80%): filtro basado en la teoría de la relación
         señal-ruido. Activos con volatilidad < 5% anualizada (ej. money markets)
         no ofrecen suficiente dispersión de retornos para que PPO aprenda — la señal
         es indistinguible del ruido. Activos con volatilidad > 80% (ej. meme stocks)
         tienen una ratio señal/ruido tan baja que el aprendizaje no converge.

       - Sharpe rolling positivo: proxy del factor de momentum (Jegadeesh & Titman, 1993).
         Activos con Sharpe negativo reciente tienen tendencia bajista — incluirlos
         fuerza al agente a aprender a shortear (no soportado) o a asignarles peso 0,
         desperdiciando capacidad del observation space.

       - Diversificación sectorial (máx. 3 por sector GICS): implementación directa
         del principio de diversificación de Markowitz (1952). Sin este filtro, el
         screener podría seleccionar 15 activos tecnológicos que se mueven de forma
         casi idéntica (correlación > 0.9), anulando los beneficios de diversificación.
         La clasificación GICS (Global Industry Classification Standard) de MSCI/S&P
         es el estándar de la industria para categorización sectorial.

    3. Ranking final por Sharpe: los candidatos que pasan todos los filtros se ordenan
       por Sharpe rolling descendente. Esto implementa un factor de momentum cross-sectional
       — se seleccionan los activos con mejor rendimiento ajustado por riesgo reciente.

Fuente del universo:
  Wikipedia publica la lista actualizada de componentes del S&P 500 en una tabla HTML.
  Se parsea directamente con pandas.read_html() — sin necesidad de API key ni scraping.

Uso:
  screener = MarketScreener()
  candidates = screener.run(start_date='2020-01-01', end_date='2026-01-01', top_n=15)
  # candidates = ['AAPL', 'MSFT', 'JNJ', ...]  → pasa a generate_dataset()
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
    ya garantiza: capitalización > ~$14.6B, volumen suficiente, domicilio en EE.UU.,
    y al menos 12 meses de historial cotizado. Esto nos ahorra filtros gruesos.

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

    # Intentar con curl_cffi (sortea el 403 de Wikipedia)
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


# ─────────────────────────────────────────────
# Screener principal
# ─────────────────────────────────────────────

class MarketScreener:
    """
    Filtra un universo amplio de activos a un conjunto reducido de candidatos
    óptimos para el agente PPO.

    Implementa el pipeline de selección de universo estándar en gestión cuantitativa
    (Grinold & Kahn, 2000): universo amplio → filtros cuantitativos → ranking → selección.

    Los filtros se aplican en orden de coste computacional creciente y restrictividad
    decreciente (primero se eliminan los que no tienen datos, luego los ilíquidos, etc.),
    siguiendo el patrón de "embudo de selección" de la industria.

    Parameters
    ----------
    min_history_years : float
        Años mínimos de historial requerido (por defecto 2).
        Fundamentación: PPO necesita observar al menos dos ciclos estacionales
        completos para distinguir patrones de ruido (López de Prado, 2018).
    min_daily_volume_usd : float
        Volumen medio diario mínimo en dólares (por defecto $5M).
        Fundamentación: por debajo de este umbral, el bid-ask spread y el
        market impact hacen que cualquier alpha teórico se pierda en costes
        de ejecución (Almgren & Chriss, 2001, "Optimal Execution").
    vol_range : tuple[float, float]
        Rango aceptable de volatilidad anualizada (por defecto 5%-80%).
        Fundamentación: la ratio señal/ruido (SNR) de un activo financiero
        es inversamente proporcional a su volatilidad. Activos con vol < 5%
        tienen SNR ≈ 0 (indistinguible de ruido); con vol > 80% la señal
        existe pero PPO no converge en tiempos de entrenamiento razonables.
    max_per_sector : int
        Máximo de activos del mismo sector GICS en los candidatos finales
        (por defecto 3).
        Fundamentación: aplicación directa del principio de diversificación
        de Markowitz (1952). Activos del mismo sector tienen correlación
        media ~0.6-0.8; incluir demasiados reduce los beneficios de
        diversificación y sesga la cartera hacia ese sector.
    sharpe_window : int
        Días de la ventana rolling para calcular Sharpe (por defecto 252 = 1 año).
        Fundamentación: el efecto momentum cross-sectional (Jegadeesh & Titman, 1993)
        es más robusto con ventanas de 6-12 meses. 252 días (1 año) es el estándar
        académico y de la industria.
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

        # Sesión HTTP resiliente para entornos corporativos
        self._session = None
        if _CURL_AVAILABLE:
            self._session = CurlSession(impersonate="chrome")
            self._session.verify = False

    def _download_basic_data(self, ticker: str,
                              start_date: str, end_date: str) -> Optional[dict]:
        """
        Descarga datos mínimos de un ticker para evaluación rápida.

        No descarga el histórico completo — solo lo necesario para calcular
        las métricas de filtrado (volumen, volatilidad, Sharpe).

        Parameters
        ----------
        ticker     : símbolo del activo
        start_date : fecha inicio 'YYYY-MM-DD'
        end_date   : fecha fin 'YYYY-MM-DD'

        Returns
        -------
        dict con métricas o None si no hay datos suficientes.
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
        Ejecuta el proceso completo de screening.

        Parameters
        ----------
        start_date     : fecha inicio para evaluación de métricas
        end_date       : fecha fin
        top_n          : número de candidatos finales a devolver
        universe_df    : DataFrame con columnas 'ticker' y 'sector'.
                         Si None, descarga los componentes del S&P 500.
        force_include  : lista de tickers que siempre se incluyen en el resultado
                         (ej. ['IVV', 'BND'] para mantener benchmarks obligatorios)

        Returns
        -------
        dict con:
          'candidates': list[str] — tickers seleccionados
          'details': pd.DataFrame — métricas de todos los candidatos
          'filtered_out': dict — cuántos activos se descartaron en cada filtro
        """
        if force_include is None:
            force_include = []

        # Paso 1: Obtener universo
        if universe_df is None:
            universe_df = fetch_sp500_tickers()

        all_tickers = universe_df['ticker'].tolist()
        sectors = dict(zip(universe_df['ticker'], universe_df.get('sector', ['Unknown'] * len(universe_df))))

        print(f"\n{'='*60}")
        print(f"MARKET SCREENER — {len(all_tickers)} activos a evaluar")
        print(f"{'='*60}")

        # Paso 2: Descargar métricas básicas de cada activo
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

        # Paso 3: Aplicar filtros secuenciales
        filtered_out = {}

        # Filtro 1: Historial mínimo
        mask = df['n_years'] >= self.min_history_years
        filtered_out['history_too_short'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro historial (>= {self.min_history_years} años): {len(df)} pasan")

        # Filtro 2: Liquidez mínima
        mask = df['avg_volume_usd'] >= self.min_daily_volume_usd
        filtered_out['low_liquidity'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro liquidez (>= ${self.min_daily_volume_usd:,.0f}/día): {len(df)} pasan")

        # Filtro 3: Volatilidad en rango
        mask = (df['annual_vol'] >= self.vol_range[0]) & (df['annual_vol'] <= self.vol_range[1])
        filtered_out['vol_out_of_range'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro volatilidad ({self.vol_range[0]:.0%}-{self.vol_range[1]:.0%}): {len(df)} pasan")

        # Filtro 4: Sharpe rolling positivo
        mask = df['sharpe_rolling'] > 0
        filtered_out['negative_sharpe'] = int((~mask).sum())
        df = df[mask]
        print(f"  Filtro Sharpe rolling > 0: {len(df)} pasan")

        if df.empty:
            print("  [AVISO] Ningún activo pasó todos los filtros. Retornando force_include.")
            return {'candidates': force_include, 'details': pd.DataFrame(),
                    'filtered_out': filtered_out}

        # Paso 4: Ordenar por Sharpe y aplicar diversificación sectorial
        df = df.sort_values('sharpe_rolling', ascending=False)

        selected = []
        sector_count = {}

        # Primero, asegurar que los force_include están
        for t in force_include:
            if t in df['ticker'].values:
                row = df[df['ticker'] == t].iloc[0]
                selected.append(row.to_dict())
                s = row['sector']
                sector_count[s] = sector_count.get(s, 0) + 1

        # Luego, rellenar hasta top_n respetando diversificación
        for _, row in df.iterrows():
            if len(selected) >= top_n:
                break
            if row['ticker'] in [s['ticker'] for s in selected]:
                continue  # Ya incluido via force_include
            s = row['sector']
            if sector_count.get(s, 0) >= self.max_per_sector:
                continue  # Sector saturado
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
