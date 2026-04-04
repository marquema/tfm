"""
Registro centralizado de activos del universo de inversión.

Fundamentación teórica:
  La construcción del universo de inversión es una decisión crítica que condiciona
  todos los resultados posteriores del modelo (Fabozzi et al., 2007, "Robust Portfolio
  Optimization and Management"). Un universo mal diseñado introduce sesgos que ningún
  algoritmo de optimización puede corregir:

    - Sesgo de supervivencia (survivorship bias): si solo incluimos activos que existen
      hoy, ignoramos los que quebraron o fueron deslistados — inflando artificialmente
      el rendimiento histórico (Brown et al., 1992).

    - Sesgo de selección: elegir activos "a dedo" basándose en su buen rendimiento
      pasado (data snooping) produce resultados no reproducibles out-of-sample
      (White, 2000, "A Reality Check for Data Snooping").

  El CORE_UNIVERSE de este TFM mitiga estos riesgos con una selección basada en
  criterios de diversificación estructural (no de rendimiento pasado):
    - 1 proxy de mercado (IVV): benchmark y cálculo de beta
    - 1 renta fija (BND): correlación históricamente negativa con equities en crisis
    - 1 activo digital (IBIT): alta volatilidad, correlación variable — test del agente
    - 5 acciones de sectores distintos: dividendo (MO), defensivo (JNJ), utilities (AWK),
      seguros (CB), alternativo (SCU) — diversificación sectorial

  El EXTENDED_UNIVERSE añade clases de activos ausentes en el core (materias primas,
  REITs, renta fija alternativa, volatilidad) para demostrar que la arquitectura
  escala sin modificar el pipeline.

Uso:
  from src.pipeline_getdata.asset_registry import get_universe, get_asset_info, get_tickers

  tickers = get_tickers('core')       # ['AWK', 'BND', 'CB', 'IBIT', 'IVV', 'JNJ', 'MO', 'SCU']
  info    = get_asset_info('IVV')     # {'name': 'iShares Core S&P 500', 'category': 'Equity', ...}
  df      = get_universe('extended')  # DataFrame con core + extendido
"""

import pandas as pd


# ─────────────────────────────────────────────
# Universo core — 8 activos del TFM
# ─────────────────────────────────────────────
# La selección sigue el principio de "cartera representativa" (Sharpe, 1964, CAPM):
# incluir activos de distintas clases para que el agente PPO aprenda a gestionar
# la diversificación entre categorías con dinámicas de retorno fundamentalmente
# distintas (equity vs fixed income vs crypto vs defensivos vs cíclicos).
#
# Cada activo cumple un rol específico en la cartera:
#   - IVV: proxy del mercado (factor de mercado del CAPM)
#   - BND: cobertura en crisis (correlación negativa histórica con equities)
#   - IBIT: activo de alta volatilidad (test de gestión de riesgo del agente)
#   - MO: retorno por dividendo dominante (income investing)
#   - JNJ: activo defensivo de baja beta (protección en drawdowns)
#   - SCU: activo alternativo descorrelado (diversificación real)
#   - AWK: utility regulada (ingresos predecibles, baja vol)
#   - CB: financiero anticíclico (sensibilidad a tipos de interés)

CORE_UNIVERSE = {
    'IVV': {
        'name': 'iShares Core S&P 500 ETF',
        'category': 'Equity',
        'sector': 'US Large Cap Index',
        'instrument': 'ETF',
        'description': (
            'Replica el índice S&P 500 — las 500 mayores empresas de EE.UU. '
            'Proxy estándar del mercado de renta variable americano. '
            'Se usa como benchmark de mercado y referencia para el cálculo de beta.'
        ),
    },
    'BND': {
        'name': 'Vanguard Total Bond Market ETF',
        'category': 'Fixed Income',
        'sector': 'US Aggregate Bond',
        'instrument': 'ETF',
        'description': (
            'Exposición diversificada al mercado de bonos de EE.UU. '
            '(gobierno, corporativo, titulizaciones). Históricamente con correlación '
            'negativa a renta variable en crisis — actúa como cobertura natural. '
            'Componente clave de la cartera 60/40.'
        ),
    },
    'IBIT': {
        'name': 'iShares Bitcoin Trust ETF',
        'category': 'Digital Asset',
        'sector': 'Cryptocurrency',
        'instrument': 'ETF',
        'description': (
            'ETF que replica el precio de Bitcoin. Disponible desde enero 2024. '
            'Activo de alta volatilidad y correlación variable con el mercado: '
            'en períodos de riesgo actúa como activo especulativo (correlación alta con IVV), '
            'en períodos de calma puede actuar como diversificador.'
        ),
    },
    'MO': {
        'name': 'Altria Group Inc.',
        'category': 'Equity',
        'sector': 'Consumer Staples / Tobacco',
        'instrument': 'Stock',
        'description': (
            'Empresa tabacalera líder en EE.UU. (Marlboro). Activo de alto dividendo '
            '(yield ~8-9%) con crecimiento limitado. Interesante para el agente porque '
            'el dividendo es una parte dominante del retorno total — la dinámica del '
            'pago (crecimiento, estabilidad) es más relevante que el precio.'
        ),
    },
    'JNJ': {
        'name': 'Johnson & Johnson',
        'category': 'Equity',
        'sector': 'Healthcare / Pharma',
        'instrument': 'Stock',
        'description': (
            'Conglomerado farmacéutico y de consumo. Considerado activo defensivo: '
            'baja beta, dividendo estable creciente durante 60+ años consecutivos '
            '(Dividend King). En crisis de mercado tiende a caer menos que el índice.'
        ),
    },
    'SCU': {
        'name': 'Sculptor Capital Management',
        'category': 'Equity',
        'sector': 'Financials / Alternative Asset Management',
        'instrument': 'Stock',
        'description': (
            'Gestora de activos alternativos (hedge funds, crédito, real estate). '
            'Activo de nicho con alta volatilidad y baja correlación con el mercado. '
            'Puede tener problemas de liquidez — volumen de negociación bajo. '
            'Nota: fue adquirida por Rithm Capital en 2023, puede tener datos '
            'limitados en períodos recientes.'
        ),
    },
    'AWK': {
        'name': 'American Water Works Co.',
        'category': 'Equity',
        'sector': 'Utilities / Water',
        'instrument': 'Stock',
        'description': (
            'Mayor empresa de agua y saneamiento cotizada en EE.UU. '
            'Activo típicamente defensivo y regulado: ingresos predecibles, '
            'baja volatilidad, dividendo moderado pero creciente. '
            'Correlación baja con el mercado — aporta estabilidad a la cartera.'
        ),
    },
    'CB': {
        'name': 'Chubb Limited',
        'category': 'Equity',
        'sector': 'Financials / Insurance',
        'instrument': 'Stock',
        'description': (
            'Mayor aseguradora cotizada del mundo por capitalización. '
            'Negocio anticíclico parcial: los ingresos por primas son estables, '
            'pero los resultados de inversión dependen de los tipos de interés. '
            'Activo de calidad con dividendo moderado y volatilidad intermedia.'
        ),
    },
}


# ─────────────────────────────────────────────
# Universo extendido — para fases futuras
# ─────────────────────────────────────────────
# Completa las clases de activos ausentes en el core según la taxonomía
# de asignación estratégica de activos (Strategic Asset Allocation, Ibbotson & Kaplan, 2000):
#   - Equity internacional: desarrollados (VEA) y emergentes (VWO) para exposición global
#   - Renta fija alternativa: bonos largos (TLT) sensibles a tipos, high yield (HYG) con riesgo crédito
#   - Materias primas: oro (GLD) como refugio inflacionario, petróleo (USO) como activo cíclico
#   - REITs: inmobiliario cotizado (VNQ) — clase de activo con baja correlación con equity pura
#   - Volatilidad: VIX (VIXY) — el único activo que sube cuando todo baja (cobertura de cola)
#
# Estos activos no se usan en el TFM actual pero demuestran que la arquitectura
# escala sin modificar el pipeline ni el entorno de entrenamiento.

EXTENDED_UNIVERSE = {
    # Renta variable internacional
    'VEA':  {'name': 'Vanguard FTSE Developed Markets ETF',  'category': 'Equity',       'sector': 'International Developed', 'instrument': 'ETF',
             'description': 'Renta variable de mercados desarrollados ex-US (Europa, Japón, Australia).'},
    'VWO':  {'name': 'Vanguard FTSE Emerging Markets ETF',   'category': 'Equity',       'sector': 'Emerging Markets',        'instrument': 'ETF',
             'description': 'Renta variable de mercados emergentes (China, India, Brasil, Taiwán).'},
    'QQQ':  {'name': 'Invesco QQQ Trust',                    'category': 'Equity',       'sector': 'US Tech / Nasdaq 100',    'instrument': 'ETF',
             'description': 'Replica el Nasdaq 100 — concentrado en tecnología (Apple, Microsoft, NVIDIA).'},

    # Renta fija alternativa
    'TLT':  {'name': 'iShares 20+ Year Treasury Bond ETF',   'category': 'Fixed Income', 'sector': 'US Long-Term Treasury',   'instrument': 'ETF',
             'description': 'Bonos del Tesoro a largo plazo (20+ años). Muy sensible a tipos de interés.'},
    'HYG':  {'name': 'iShares iBoxx High Yield Corp Bond',   'category': 'Fixed Income', 'sector': 'High Yield Corporate',    'instrument': 'ETF',
             'description': 'Bonos corporativos de alto rendimiento (high yield / junk bonds). Más riesgo que BND.'},

    # Materias primas
    'GLD':  {'name': 'SPDR Gold Shares',                     'category': 'Commodity',    'sector': 'Precious Metals / Gold',  'instrument': 'ETF',
             'description': 'Replica el precio del oro. Activo refugio clásico en crisis e inflación.'},
    'USO':  {'name': 'United States Oil Fund',               'category': 'Commodity',    'sector': 'Energy / Crude Oil',      'instrument': 'ETF',
             'description': 'Replica el precio del petróleo WTI. Alta volatilidad y estacionalidad.'},

    # REITs (inmobiliario cotizado)
    'VNQ':  {'name': 'Vanguard Real Estate ETF',             'category': 'Real Estate',  'sector': 'US REITs',                'instrument': 'ETF',
             'description': 'Exposición diversificada a inmobiliario cotizado (REITs) de EE.UU.'},

    # Volatilidad
    'VIXY': {'name': 'ProShares VIX Short-Term Futures',     'category': 'Volatility',   'sector': 'VIX Futures',             'instrument': 'ETN',
             'description': 'Exposición a la volatilidad implícita del S&P 500 (VIX). Sube en pánico de mercado.'},
}


# ─────────────────────────────────────────────
# Funciones de acceso al registro
# ─────────────────────────────────────────────

def get_tickers(level: str = 'core') -> list:
    """
    Retorna la lista de tickers del universo solicitado.

    Parameters
    ----------
    level : str
        'core' → los 8 activos del TFM
        'extended' → core + activos adicionales para fases futuras
        'all' → sinónimo de 'extended'

    Returns
    -------
    list[str]
        Lista de símbolos de ticker ordenados alfabéticamente.
    """
    if level == 'core':
        return sorted(CORE_UNIVERSE.keys())
    elif level in ('extended', 'all'):
        combined = {**CORE_UNIVERSE, **EXTENDED_UNIVERSE}
        return sorted(combined.keys())
    else:
        raise ValueError(f"Nivel '{level}' no reconocido. Usa 'core', 'extended' o 'all'.")


def get_asset_info(ticker: str) -> dict:
    """
    Retorna los metadatos completos de un activo.

    Parameters
    ----------
    ticker : str
        Símbolo del activo (ej. 'IVV', 'BND').

    Returns
    -------
    dict
        Diccionario con claves: name, category, sector, instrument, description.
        Si el ticker no está registrado, retorna un dict con valores genéricos.
    """
    if ticker in CORE_UNIVERSE:
        return {'ticker': ticker, **CORE_UNIVERSE[ticker]}
    elif ticker in EXTENDED_UNIVERSE:
        return {'ticker': ticker, **EXTENDED_UNIVERSE[ticker]}
    else:
        return {
            'ticker': ticker,
            'name': ticker,
            'category': 'Unknown',
            'sector': 'Unknown',
            'instrument': 'Unknown',
            'description': f'Activo {ticker} no registrado en el diccionario.'
        }


def get_universe(level: str = 'core') -> pd.DataFrame:
    """
    Retorna el universo de activos como DataFrame con todos los metadatos.

    Parameters
    ----------
    level : str
        'core', 'extended' o 'all'.

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por ticker con columnas: name, category, sector,
        instrument, description. Útil para mostrar en el dashboard o exportar
        a la memoria del TFM.
    """
    tickers = get_tickers(level)
    rows = [get_asset_info(t) for t in tickers]
    df = pd.DataFrame(rows).set_index('ticker')
    return df


def get_display_name(ticker: str) -> str:
    """
    Retorna el nombre legible de un activo para mostrar en gráficas.

    Parameters
    ----------
    ticker : str
        Símbolo del activo.

    Returns
    -------
    str
        Formato 'TICKER — Nombre completo' (ej. 'IVV — iShares Core S&P 500 ETF').
        Si no está registrado, retorna solo el ticker.
    """
    info = get_asset_info(ticker)
    if info['name'] != ticker:
        return f"{ticker} — {info['name']}"
    return ticker
