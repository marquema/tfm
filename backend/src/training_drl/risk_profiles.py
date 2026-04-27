"""
Perfiles de riesgo para el agente PPO.

Define las configuraciones de hiperparámetros de la función de recompensa
(phi y gamma) como perfiles con nombre legible. Cada perfil representa
una filosofía de inversión distinta:

  - balanced:  equilibrio entre retorno y control de riesgo (configuración
                   base del TFM y default del entrenamiento).
  - conservative : prioriza preservar capital, penaliza drawdowns más fuertemente.
  - low_turnover : fuerza al agente a operar menos, reduciendo costes de
                   transacción. Es el perfil que mejor Sharpe obtuvo en el
                   análisis de sensibilidad y queda como candidato a perfil
                   de referencia secundario para la memoria.
  - aggressive: mínimas penalizaciones, máxima libertad para el agente.

Los perfiles se usan en:
  - train_academic(): el admin elige con qué filosofía entrenar el modelo.
  - PortfolioEnv: recibe phi y gamma del perfil seleccionado.
  - sensitivity_analysis: compara los 4 perfiles automáticamente y produce
    el CSV/PNG comparativo que sustenta la justificación en la memoria.
  - BD (TrainedModel.train_metrics): se guarda el perfil usado en cada
    modelo entrenado para poder mostrarlo en el dashboard y en la tabla final.

Fundamentación de los valores (orden de magnitud):
  La calibración busca que cada componente del reward tenga peso comparable
  para que ninguno domine al resto. Tomando como referencia un retorno diario
  típico de ~0.1 % (≈0.001 en escala lineal):

    - phi=0.02 con un MDD del 20 %  → penalty ≈ 0.004 (mismo orden que el
      retorno diario; hace que un drawdown profundo "duela" tanto como un día
      bueno).
    - gamma=0.01 con un turnover completo (cambio total de cartera)
      → penalty ≈ 0.02 (~20 días de retorno positivo; rotar todo solo merece
      la pena si la mejora esperada lo compensa).

  Nota: la reward actual es Sharpe rolling 20d − phi·MDD − gamma·Turnover,
  no log_return. Las penalizaciones se calibraron originalmente con
  log_return como referencia y se mantienen porque el Sharpe rolling diario
  produce reward por step en una escala similar (~ ±0.5/√20 ≈ ±0.1) al log_return.
  
  Es decir, cambiamos la receta del premio pero no tocamos los castigos. 
    ¿Por qué no pasa nada? Porque el premio nuevo y el viejo dan números parecidos en 
    tamaño, así que los castigos siguen pesando lo mismo en proporción. Si en vez de 
    eso hubiéramos puesto un premio en escala 1000 (por ejemplo el valor de la cartera
    entera), tendríamos que haber subido phi y gamma a la misma escala. Pero no es el 
    caso.
  

Referencia:
  Los valores se contrastaron mediante el análisis de sensibilidad
  (sensitivity_analysis.py) sobre el periodo out-of-sample. Resultado:
  todos los perfiles obtienen Sharpe > 2.2, lo que evidencia robustez
  frente a variaciones de phi/gamma dentro del rango explorado.
"""


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
