"""
Perfiles de riesgo para el agente PPO.

Define las configuraciones de hiperparámetros de la función de recompensa
(phi y gamma) como perfiles con nombre legible. Cada perfil representa
una filosofía de inversión distinta:

  - balanced:     equilibrio entre retorno y control de riesgo (configuración base)
  - conservative: prioriza preservar capital, penaliza drawdowns más fuertemente
  - low_turnover: fuerza al agente a operar menos, reduciendo costes de transacción
  - aggressive:   mínimas penalizaciones, máxima libertad para el agente

Los perfiles se usan en:
  - train_academic(): el admin elige con qué filosofía entrenar el modelo
  - PortfolioEnv: recibe phi y gamma del perfil seleccionado
  - sensitivity_analysis: compara los 4 perfiles automáticamente
  - BD (universe_metadata): se guarda qué perfil se usó para cada modelo

Fundamentación de los valores:
  Los valores están calibrados para que el reward tenga escalas comparables:
    - log_return diario típico: ~0.001 (0.1%)
    - phi=0.02, MDD=20% → penalty=0.004 (mismo orden que el retorno)
    - gamma=0.01, turnover completo → penalty=0.02 (~20 días de retorno)

Referencia:
  Los valores óptimos se determinaron mediante el análisis de sensibilidad
  (sensitivity_analysis.py) sobre el período out-of-sample.
"""

from typing import Optional


RISK_PROFILES = {
    'balanced': {
        'phi': 0.02,
        'gamma': 0.01,
        'name': 'Equilibrado',
        'description': (
            'Balance entre retorno y control de riesgo. '
            'Configuración base del TFM. phi=0.02 penaliza drawdowns '
            'proporcionalmente al retorno diario; gamma=0.01 desincentiva '
            'rotación excesiva sin impedir al agente reaccionar a cambios de mercado.'
        ),
    },
    'conservative': {
        'phi': 0.05,
        'gamma': 0.01,
        'name': 'Conservador',
        'description': (
            'Prioriza preservar capital sobre maximizar retorno. '
            'phi=0.05 penaliza drawdowns 2.5x más que el perfil equilibrado. '
            'El agente aprende a reducir exposición a activos volátiles (IBIT, ETHA) '
            'en momentos de incertidumbre, a costa de menor retorno en mercados alcistas.'
        ),
    },
    'low_turnover': {
        'phi': 0.02,
        'gamma': 0.02,
        'name': 'Bajo Turnover',
        'description': (
            'Fuerza al agente a mantener posiciones, reduciendo costes de transacción. '
            'gamma=0.02 hace que una rotación completa de cartera cueste 0.04 en reward, '
            'equivalente a ~40 días de retorno positivo. El agente solo opera cuando '
            'la mejora esperada compensa ampliamente el coste de mover capital. '
            'Mejor Sharpe en el análisis de sensibilidad.'
        ),
    },
    'aggressive': {
        'phi': 0.01,
        'gamma': 0.005,
        'name': 'Agresivo',
        'description': (
            'Mínimas penalizaciones: el agente tiene máxima libertad para buscar retorno. '
            'phi=0.01 apenas penaliza drawdowns; gamma=0.005 permite rotar la cartera '
            'frecuentemente. Mayor potencial de retorno pero también mayor riesgo '
            'y costes de transacción. Adecuado para mercados con tendencias claras.'
        ),
    },
}


def get_profile(name: str) -> dict:
    """
    Retorna los parámetros de un perfil de riesgo por nombre.

    Parameters
    ----------
    name : str
        Nombre del perfil: 'balanced', 'conservative', 'low_turnover', 'aggressive'.

    Returns
    -------
    dict con claves: phi, gamma, name, description.

    Raises
    ------
    ValueError si el nombre no es válido.
    """
    if name not in RISK_PROFILES:
        valid = ', '.join(RISK_PROFILES.keys())
        raise ValueError(
            f"Perfil '{name}' no reconocido. Perfiles disponibles: {valid}"
        )
    return RISK_PROFILES[name]


def get_phi_gamma(name: str) -> tuple:
    """
    Retorna (phi, gamma) de un perfil de riesgo.

    Atajo para no tener que acceder al dict completo cuando solo
    se necesitan los hiperparámetros numéricos.

    Parameters
    ----------
    name : str
        Nombre del perfil.

    Returns
    -------
    tuple (phi: float, gamma: float)
    """
    profile = get_profile(name)
    return profile['phi'], profile['gamma']


def list_profiles() -> list:
    """
    Retorna la lista de perfiles disponibles con toda su información.

    Útil para el endpoint que expone los perfiles al frontend.

    Returns
    -------
    list[dict] con claves: id, phi, gamma, name, description.
    """
    return [
        {'id': key, **value}
        for key, value in RISK_PROFILES.items()
    ]
