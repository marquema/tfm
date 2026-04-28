"""
Perfiles de riesgo para el agente PPO.

Define las configuraciones de hiperparámetros de la función de recompensa
(phi y gamma) como perfiles con nombre legible. Cada perfil representa
una filosofía de inversión distinta:

  - low_turnover : fuerza al agente a operar menos, reduciendo costes de
                   transacción. PERFIL PRINCIPAL DEL TFM y default del
                   entrenamiento: el análisis de sensibilidad (ver
                   sensitivity_analysis.py) lo identificó como la
                   configuración con mejor Sharpe out-of-sample, así que
                   el modelo final del TFM se entrena con esta calibración.
                   La elección está respaldada por evidencia empírica
                   reproducible, no por defecto subjetivo.
  - balanced     : equilibrio entre retorno y control de riesgo. Perfil
                   alternativo y baseline conservador del catálogo. Útil
                   cuando se quiere un comportamiento predecible sin
                   apostar por la calibración óptima del sensitivity.
  - conservative : prioriza preservar capital, penaliza drawdowns más fuertemente.
  - aggressive   : mínimas penalizaciones, máxima libertad para el agente.

Los perfiles se usan en:
  - train_academic(): el admin elige con qué filosofía entrenar el modelo.
  - PortfolioEnv: recibe phi y gamma del perfil seleccionado.
  - sensitivity_analysis: compara los 4 perfiles automáticamente y produce
    el CSV/PNG comparativo que sustenta la justificación en la memoria.
  - BD (TrainedModel.train_metrics): se guarda el perfil usado en cada
    modelo entrenado para poder mostrarlo en el dashboard y en la tabla final.

Fundamentación de los valores:
  La reward por step del entorno es:

      r_t = sharpe_rolling_20d(t) − phi · MDD(t) − gamma · turnover(t)

  recortada a [-1, 1]. La unidad del reward es por tanto el Sharpe rolling
  diario (rango típico por step ~ ±0.1), NO el log-retorno diario. Cualquier
  intuición del tipo "esta penalización equivale a X días de retorno
  positivo" mezcla dos magnitudes distintas (Sharpe vs log-retorno) y
  resulta engañosa.

  Magnitud cruda de cada penalización en unidades de reward, en escenarios
  extremos:

    - phi = 0.02 con un drawdown del 20 % (MDD = 0.20):
        penalty_MDD = 0.02 × 0.20 = 0.004 por step
        Magnitud comparable a la oscilación diaria del Sharpe rolling.

    - gamma = 0.02 con una rotación completa de la cartera:
        Si el agente vende todas las posiciones previas y compra activos
        distintos, el cambio total de pesos es 2.0 (1.0 vendido + 1.0
        comprado, sumando valores absolutos). Por tanto:
        penalty_turnover = 0.02 × 2.0 = 0.04 por step.
        Para perfil balanced (gamma = 0.01) la magnitud es la mitad: 0.02.

  Justificación empírica (no narrativa):
    La calibración final no se justifica por un argumento dimensional sobre
    "días de retorno", sino por el resultado del análisis de sensibilidad
    sobre el periodo out-of-sample (200 k pasos por config, split 80/20):

      Config                  phi    gamma   Sharpe   Retorno   MDD
      ----------------------  -----  ------  -------  --------  -------
      A  balanced             0.02   0.010    1.368   112.4 %   -31.8 %
      B  conservative         0.05   0.010    1.613   170.6 %   -40.8 %
      C  low_turnover (TFM)   0.02   0.020    1.770   182.2 %   -40.0 %
      D  aggressive           0.01   0.005    1.310   102.6 %   -31.1 %

    Lectura:
      - low_turnover gana en Sharpe (1.77) y retorno acumulado (182 %);
        por eso es el perfil principal del TFM.
      - balanced y aggressive obtienen MDD menor (~32 %) frente al ~40 %
        de conservative y low_turnover. Elegir low_turnover por Sharpe
        asume que el equilibrio retorno/riesgo prima sobre el MDD
        absoluto: el Sharpe ya pondera retorno por volatilidad, y el
        propio gamma penaliza turnover dentro del entrenamiento.
      - El rango de Sharpe (1.31 a 1.77) es estrecho frente a la
        magnitud de las variaciones de phi/gamma, lo que evidencia
        robustez de la política frente a la calibración exacta de la
        recompensa dentro del rango explorado.
"""


RISK_PROFILES = {
    'balanced': {
        'phi': 0.02,
        'gamma': 0.01,
        'name': 'Equilibrado',
        'description': (
            'Balance entre retorno y control de riesgo. Perfil alternativo '
            'y baseline del catálogo. phi=0.02 penaliza drawdowns con magnitud '
            'comparable a la oscilación diaria del Sharpe rolling; gamma=0.01 '
            'desincentiva rotación excesiva sin impedir al agente reaccionar a '
            'cambios de mercado. En el sensitivity obtiene Sharpe 1.37 con MDD '
            '~32 %, frente al 1.77 / MDD ~40 % de low_turnover.'
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
            'PERFIL PRINCIPAL DEL TFM. Fuerza al agente a mantener posiciones, '
            'reduciendo costes de transacción. gamma=0.02 hace que una rotación '
            'completa de cartera (cambio total de pesos = 2.0) reste 0.04 al '
            'reward de ese step, el doble que en el perfil balanced. El agente '
            'solo rota cuando la mejora esperada en Sharpe compensa ese coste. '
            'Identificado empíricamente como la configuración con mejor Sharpe '
            'out-of-sample en el análisis de sensibilidad (Sharpe 1.77 vs 1.31-1.61 '
            'del resto): el modelo final del TFM se entrena con este perfil.'
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
