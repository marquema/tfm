"""
Módulo de baselines financieros para benchmarking comparativo con el agente DRL.

Implementa las cuatro estrategias de referencia definidas en el TFM:
  1. Equal-Weight mensual: 1/N entre todos los activos, rebalanceo el primer día de cada mes
  2. Cartera 60/40: 60% renta variable (IVV) / 40% renta fija (BND), rebalanceo mensual
  3. Buy & Hold: asignación 1/N fija desde el inicio, sin ningún rebalanceo posterior
  4. Markowitz Media-Varianza: maximización del Ratio de Sharpe con ventana de estimación
     de 252 días (~12 meses), reoptimización mensual con scipy.optimize

Todas las estrategias:
  - Reciben df_precios con fechas en orden ASCENDENTE (fecha más antigua primero)
  - Aplican comisiones de transacción en cada rebalanceo
  - Devuelven pd.Series de valores absolutos de cartera indexados por fecha

La función calcular_metricas() calcula: Sharpe, Sortino, MDD, CAGR, Volatilidad y Retorno Total.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────

def _rebalancear(balance: float, weights_nuevo: np.ndarray,
                 weights_actual: np.ndarray, commission: float) -> tuple:
    """
    Aplica el coste de rebalanceo y actualiza los pesos.

    El coste es proporcional al turnover total (suma de cambios absolutos en pesos)
    multiplicado por el valor actual de la cartera y la tasa de comisión.
    """
    turnover = np.sum(np.abs(weights_nuevo - weights_actual))
    coste    = turnover * balance * commission
    return balance - coste, weights_nuevo.copy()


def _es_inicio_mes(fecha_hoy: pd.Timestamp, fecha_ayer: pd.Timestamp) -> bool:
    """Retorna True si se ha cruzado el inicio de un nuevo mes calendario."""
    return fecha_hoy.month != fecha_ayer.month


# ─────────────────────────────────────────────
# Baseline 1: Equal-Weight con rebalanceo mensual
# ─────────────────────────────────────────────

def simular_equal_weight(df_precios: pd.DataFrame,
                          initial_balance: float = 10000,
                          commission: float = 0.001) -> pd.Series:
    """
    Cartera equiponderada con rebalanceo mensual (1/N por activo).

    Estrategia:
      - Asigna el mismo peso (1/N) a cada activo al inicio y en cada rebalanceo mensual
      - Entre rebalanceos los pesos flotan con el mercado (algunos activos ganan más peso)
      - Aplica comisiones sobre el turnover generado en cada rebalanceo

    Parámetros
    ----------
    df_precios      : DataFrame de precios de cierre (columnas = activos, índice = fechas ASC)
    initial_balance : capital inicial en dólares
    commission      : tasa de comisión por transacción (0.001 = 0.1%)

    Retorna
    -------
    pd.Series con el valor de la cartera en cada día, indexada por fecha
    """
    assert df_precios.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente (fecha más antigua primero)."

    n             = len(df_precios.columns)
    weights_obj   = np.ones(n) / n       # Pesos objetivo: siempre 1/N
    weights_curr  = weights_obj.copy()   # Pesos actuales (derivan con el mercado)
    balance       = initial_balance
    valores       = [initial_balance]

    retornos = df_precios.pct_change().fillna(0)

    for i in range(1, len(df_precios)):
        fecha_hoy  = pd.to_datetime(df_precios.index[i])
        fecha_ayer = pd.to_datetime(df_precios.index[i - 1])

        # Rebalanceo mensual: restaurar los pesos a 1/N el primer día de cada mes
        if _es_inicio_mes(fecha_hoy, fecha_ayer):
            balance, weights_curr = _rebalancear(balance, weights_obj, weights_curr, commission)

        # Actualizar valor de la cartera con los retornos del día
        rets    = retornos.iloc[i].values
        balance = balance * np.sum(weights_curr * (1 + rets))
        balance = max(balance, 1e-6)

        # Actualizar pesos: derivan orgánicamente según el rendimiento de cada activo
        new_vals     = weights_curr * (1 + rets)
        weights_curr = new_vals / (new_vals.sum() + 1e-8)

        valores.append(balance)

    return pd.Series(valores, index=df_precios.index, name='Equal_Weight_Mensual')


# ─────────────────────────────────────────────
# Baseline 2: Cartera 60/40
# ─────────────────────────────────────────────

def simular_60_40(df_precios: pd.DataFrame,
                   ticker_rv: str = 'IVV_Close',
                   ticker_rf: str = 'BND_Close',
                   initial_balance: float = 10000,
                   commission: float = 0.001) -> pd.Series:
    """
    Cartera clásica 60/40: 60% renta variable / 40% renta fija.

    En el contexto del TFM:
      - 60%: IVV (S&P 500 ETF, renta variable de máxima liquidez)
      - 40%: BND (Vanguard Total Bond Market ETF, renta fija diversificada)

    Rebalanceo mensual para restaurar los pesos objetivo tras la deriva del mercado.

    Parámetros
    ----------
    ticker_rv : nombre de la columna de renta variable en df_precios
    ticker_rf : nombre de la columna de renta fija en df_precios
    """
    assert df_precios.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    if ticker_rv not in df_precios.columns or ticker_rf not in df_precios.columns:
        raise ValueError(
            f"Columnas '{ticker_rv}' o '{ticker_rf}' no encontradas. "
            f"Columnas disponibles: {list(df_precios.columns)}"
        )

    # Trabajar solo con los dos activos de la estrategia
    p             = df_precios[[ticker_rv, ticker_rf]].copy()
    weights_obj   = np.array([0.6, 0.4])
    weights_curr  = weights_obj.copy()
    balance       = initial_balance
    valores       = [initial_balance]
    retornos      = p.pct_change().fillna(0)

    for i in range(1, len(p)):
        fecha_hoy  = pd.to_datetime(p.index[i])
        fecha_ayer = pd.to_datetime(p.index[i - 1])

        if _es_inicio_mes(fecha_hoy, fecha_ayer):
            balance, weights_curr = _rebalancear(balance, weights_obj, weights_curr, commission)

        rets    = retornos.iloc[i].values
        balance = balance * np.sum(weights_curr * (1 + rets))
        balance = max(balance, 1e-6)

        new_vals     = weights_curr * (1 + rets)
        weights_curr = new_vals / (new_vals.sum() + 1e-8)

        valores.append(balance)

    return pd.Series(valores, index=p.index, name='Cartera_60_40')


# ─────────────────────────────────────────────
# Baseline 3: Buy & Hold
# ─────────────────────────────────────────────

def simular_buy_and_hold(df_precios: pd.DataFrame,
                          initial_balance: float = 10000) -> pd.Series:
    """
    Estrategia Buy & Hold: asignación inicial 1/N, sin ningún rebalanceo posterior.

    El activo más rentable irá ganando peso progresivamente (efecto drift).
    No genera costes de transacción más allá de la compra inicial.
    Es la estrategia pasiva de menor fricción.
    """
    assert df_precios.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    n        = len(df_precios.columns)
    weights  = np.ones(n) / n
    retornos = df_precios.pct_change().fillna(0)

    # Retorno de la cartera = promedio ponderado con pesos fijos iniciales
    retorno_cartera = (retornos * weights).sum(axis=1)
    valores         = initial_balance * (1 + retorno_cartera).cumprod()

    return pd.Series(valores.values, index=df_precios.index, name='Buy_and_Hold')


# ─────────────────────────────────────────────
# Baseline 4: Markowitz Media-Varianza
# ─────────────────────────────────────────────

def _maximizar_sharpe(retornos_hist: pd.DataFrame, rf_anual: float = 0.04) -> np.ndarray:
    """
    Optimización cuadrática de Markowitz: maximiza el Ratio de Sharpe.

    Problema:
      max  (μ_p - rf) / σ_p
      s.t. Σw_i = 1,  w_i ≥ 0  (solo posiciones largas, sin short selling)

    Usa scipy.optimize.minimize con método SLSQP.
    Si la optimización falla, retorna pesos equiponderados (fallback seguro).
    """
    n     = retornos_hist.shape[1]
    mu    = retornos_hist.mean().values * 252    # Retornos esperados anualizados
    sigma = retornos_hist.cov().values * 252     # Matriz de covarianza anualizada

    def neg_sharpe(w):
        """Función objetivo negativa (minimizamos el negativo del Sharpe)."""
        ret_p = float(np.dot(w, mu))
        var_p = float(np.dot(w, np.dot(sigma, w)))
        vol_p = np.sqrt(max(var_p, 1e-10))
        return -(ret_p - rf_anual) / vol_p

    restricciones = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    limites       = [(0.0, 1.0)] * n
    w0            = np.ones(n) / n  # Punto de partida: equiponderado

    try:
        res = minimize(
            neg_sharpe, w0,
            method='SLSQP',
            bounds=limites,
            constraints=restricciones,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0.0, 1.0)
            return w / (w.sum() + 1e-8)
    except Exception as e:
        pass  # Fallback silencioso a igual ponderación

    return w0  # Fallback: Equal-Weight si la optimización no converge


def simular_markowitz(df_precios: pd.DataFrame,
                       ventana_estimacion: int = 252,
                       initial_balance: float = 10000,
                       commission: float = 0.001,
                       rf_anual: float = 0.04) -> pd.Series:
    """
    Cartera de Media-Varianza de Markowitz con reoptimización mensual.

    Proceso mensual:
      1. Toma los últimos 'ventana_estimacion' días de retornos históricos
      2. Optimiza los pesos para maximizar el Ratio de Sharpe
      3. Rebalancea la cartera (con comisiones) hacia los nuevos pesos óptimos

    Parámetros
    ----------
    ventana_estimacion : días de historia para estimar μ y Σ (252 ≈ 12 meses)
    rf_anual           : tasa libre de riesgo para el cálculo del Sharpe (anualizada)

    Nota: La serie resultado empieza 'ventana_estimacion' días después del inicio
    de df_precios, por lo que es más corta que las demás estrategias.
    """
    assert df_precios.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    n        = len(df_precios.columns)
    retornos = df_precios.pct_change().dropna()

    valores                  = []
    indice                   = []
    balance                  = initial_balance
    weights_curr             = np.ones(n) / n
    ultimo_mes_rebalanceo    = None

    for i in range(ventana_estimacion, len(df_precios)):
        fecha     = pd.to_datetime(df_precios.index[i])
        mes_actual = fecha.month

        # Reoptimización mensual con la ventana de historia más reciente
        if mes_actual != ultimo_mes_rebalanceo:
            retornos_hist  = retornos.iloc[i - ventana_estimacion: i]
            weights_nuevos = _maximizar_sharpe(retornos_hist, rf_anual)

            balance, weights_curr = _rebalancear(
                balance, weights_nuevos, weights_curr, commission
            )
            ultimo_mes_rebalanceo = mes_actual

        # Actualizar valor con el retorno del día
        # retornos.iloc[i-1] = retorno del día i respecto al día i-1
        if i - 1 < len(retornos):
            rets    = retornos.iloc[i - 1].values
            balance = balance * np.sum(weights_curr * (1 + rets))
            balance = max(balance, 1e-6)

            new_vals     = weights_curr * (1 + rets)
            weights_curr = new_vals / (new_vals.sum() + 1e-8)

        valores.append(balance)
        indice.append(df_precios.index[i])

    return pd.Series(valores, index=indice, name='Markowitz_MV')


# ─────────────────────────────────────────────
# Ejecución conjunta de baselines
# ─────────────────────────────────────────────

def ejecutar_baselines(df_precios: pd.DataFrame,
                        initial_balance: float = 10000,
                        commission: float = 0.001,
                        ticker_rv: str = 'IVV_Close',
                        ticker_rf: str = 'BND_Close') -> dict:
    """
    Ejecuta los cuatro baselines sobre el mismo conjunto de precios.

    Retorna
    -------
    dict {nombre_estrategia: pd.Series de valores de cartera}
    """
    resultados = {}

    print("  Simulando Equal-Weight mensual...")
    resultados['Equal_Weight_Mensual'] = simular_equal_weight(
        df_precios, initial_balance, commission
    )

    print("  Simulando Buy & Hold...")
    resultados['Buy_and_Hold'] = simular_buy_and_hold(df_precios, initial_balance)

    # 60/40 solo si están disponibles IVV y BND
    if ticker_rv in df_precios.columns and ticker_rf in df_precios.columns:
        print("  Simulando cartera 60/40...")
        resultados['Cartera_60_40'] = simular_60_40(
            df_precios, ticker_rv, ticker_rf, initial_balance, commission
        )
    else:
        print(f"  [AVISO] 60/40 no disponible: columnas '{ticker_rv}' o '{ticker_rf}' ausentes.")

    print("  Simulando Markowitz Media-Varianza (puede tardar unos segundos)...")
    resultados['Markowitz_MV'] = simular_markowitz(df_precios, initial_balance=initial_balance)

    return resultados


# ─────────────────────────────────────────────
# Métricas financieras
# ─────────────────────────────────────────────

def calcular_metricas(serie: pd.Series, rf_anual: float = 0.04) -> dict:
    """
    Calcula el conjunto estándar de métricas de rendimiento financiero.

    Métricas calculadas:
      - Retorno Total (%):          (V_final / V_inicial - 1) × 100
      - CAGR (%):                   Tasa de crecimiento anual compuesta
      - Volatilidad Anualizada (%): Desviación estándar diaria × √252
      - Sharpe Ratio:               Exceso de retorno / Volatilidad total
      - Sortino Ratio:              Exceso de retorno / Volatilidad a la baja
                                    (solo penaliza los retornos negativos)
      - Max Drawdown (%):           Mayor caída acumulada desde un máximo histórico
      - Valor Final ($):            Valor absoluto final de la cartera

    Parámetros
    ----------
    serie    : pd.Series de valores absolutos de cartera (ej. empezando en 10000)
    rf_anual : tasa libre de riesgo anualizada para el cálculo de Sharpe y Sortino
    """
    retornos = serie.pct_change().dropna()
    n_dias   = len(retornos)

    if n_dias < 2:
        return {'Retorno Total (%)': 0, 'CAGR (%)': 0, 'Volatilidad Anualizada (%)': 0,
                'Sharpe Ratio': 0, 'Sortino Ratio': 0,
                'Max Drawdown (%)': 0, 'Valor Final ($)': float(serie.iloc[-1])}

    retorno_total = (serie.iloc[-1] / serie.iloc[0]) - 1
    cagr          = (1 + retorno_total) ** (252 / n_dias) - 1
    vol_anual     = retornos.std() * np.sqrt(252)

    rf_diaria  = rf_anual / 252
    exceso_ret = retornos - rf_diaria

    # Sharpe: exceso de retorno sobre volatilidad total
    sharpe = (exceso_ret.mean() / (retornos.std() + 1e-8)) * np.sqrt(252)

    # Sortino: exceso de retorno sobre volatilidad de los retornos negativos
    rets_negativos = retornos[retornos < rf_diaria]
    downside_std   = rets_negativos.std() * np.sqrt(252) if len(rets_negativos) > 1 else 1e-8
    sortino        = (exceso_ret.mean() * 252) / downside_std

    # Maximum Drawdown: mayor caída desde un máximo histórico
    rolling_max = serie.cummax()
    drawdown    = (serie - rolling_max) / (rolling_max + 1e-8)
    max_dd      = drawdown.min()

    return {
        'Retorno Total (%)':          round(retorno_total * 100, 2),
        'CAGR (%)':                   round(cagr * 100, 2),
        'Volatilidad Anualizada (%)': round(vol_anual * 100, 2),
        'Sharpe Ratio':               round(sharpe, 3),
        'Sortino Ratio':              round(sortino, 3),
        'Max Drawdown (%)':           round(max_dd * 100, 2),
        'Valor Final ($)':            round(float(serie.iloc[-1]), 2),
    }


def tabla_comparativa(resultados: dict, rf_anual: float = 0.04) -> pd.DataFrame:
    """
    Genera una tabla comparativa de métricas para múltiples estrategias.

    Parámetros
    ----------
    resultados : dict {nombre_estrategia: pd.Series de valores de cartera}

    Retorna
    -------
    DataFrame con una fila por estrategia y una columna por métrica
    (listo para exportar a LaTeX o CSV para la memoria del TFM)
    """
    filas = {}
    for nombre, serie in resultados.items():
        if serie is not None and len(serie) > 1:
            filas[nombre] = calcular_metricas(serie, rf_anual)

    return pd.DataFrame(filas).T
