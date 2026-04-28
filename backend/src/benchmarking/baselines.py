"""
Modulo de baselines(estrategias de referencia) financieros para benchmarking comparativo con el agente DRL.

Implementa las seis estrategias de referencia definidas en el TFM:
  1. Equal-Weight mensual: 1/N entre todos los activos, rebalanceo el primer dia de cada mes, para que estén iguales
  2. Cartera 60/40: 60% renta variable (IVV) / 40% renta fija (BND), rebalanceo mensual.
        Se comra lo que bajó y se vende lo que subió. Fondos de pensiones usan esta estrategia.
  3. Buy & Hold: asignacion 1/N fija desde el inicio, sin ningun rebalanceo posterior.
        En mercados alcistas es lo mejor: se compra, no se hace nada, y simplemente se espera.
  4. Markowitz Media-Varianza: maximizacion del Ratio de Sharpe con ventana de estimacion
     de 252 dias (~12 meses), reoptimizacion mensual con scipy.optimize.
  5. Random Uniform: cartera con pesos uniformemente aleatorios sobre el simplex,
     rebalanceo mensual con seed fijo para reproducibilidad. Lower bound de cordura
     (si el agente DRL no supera al random, no esta aprendiendo).
  6. Momentum Top-K: cada mes, asigna pesos equiponderados al top-K de activos con
     mejor retorno acumulado en los ultimos N dias. Baseline competitiva del sector
     (factor momentum cross-sectional).
        Retorno esperado, volatilidad y correlación entre ellos. Frontera eficiente: no puedo ganar mas sin más riesgo.
        La idea central: no miramos activos individuales, miramos cómo se combinan
     Sharpe = (retorno_anual - tasa_libre_de_riesgo) / volatilidad_anual
        ¿cuánto me pagan por cada unidad de riesgo que asumo?
        
        ¿Qué hace Markowitz en nuestro código?

        Cada mes:
        Mira los retornos y correlaciones de los últimos 252 días (1 año)
        Prueba miles de combinaciones de pesos
        Encuentra los pesos que maximizan el Sharpe — el punto de la frontera eficiente donde el retorno por unidad de riesgo es máximo
        Rebalancea la cartera a esos pesos

Todas las estrategias:
  - Reciben df_prices con fechas en orden ASCENDENTE (fecha mas antigua primero)
  - Aplican comisiones de transaccion en cada rebalanceo
  - Devuelven pd.Series de valores absolutos de cartera indexados por fecha

La funcion compute_metrics() calcula: Sharpe, Sortino, MDD, CAGR, Volatilidad y Retorno Total.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────
def _rebalance(balance: float, target_weights: np.ndarray,
               current_weights: np.ndarray, commission: float) -> tuple:
    """
    Aplica el coste de rebalanceo y actualiza los pesos.

    El coste es proporcional al turnover total (suma de cambios absolutos en pesos)
    multiplicado por el valor actual de la cartera y la tasa de comision.

    Parameters
    ----------
    balance : float
        Valor actual de la cartera en dolares.
    target_weights : np.ndarray
        Vector de pesos objetivo tras el rebalanceo.
    current_weights : np.ndarray
        Vector de pesos actuales antes del rebalanceo.
    commission : float
        Tasa de comision por transaccion (ej. 0.001 = 0.1%).

    Returns
    -------
    tuple
        (nuevo_balance, nuevos_pesos) tras descontar el coste de rebalanceo.
    """
    turnover = np.sum(np.abs(target_weights - current_weights))# cuánto rota la cartera: suma de de cambios aboslutos
    cost     = turnover * balance * commission
    return balance - cost, target_weights.copy()


def _is_month_start(today: pd.Timestamp, yesterday: pd.Timestamp) -> bool:
    """
    Determina si se ha cruzado el inicio de un nuevo mes calendario.

    Parameters
    ----------
    today : pd.Timestamp
        Fecha del dia actual.
    yesterday : pd.Timestamp
        Fecha del dia anterior.

    Returns
    -------
    bool
        True si 'today' pertenece a un mes distinto al de 'yesterday'.
    """
    return today.month != yesterday.month


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Baseline 1: Equal-Weight con rebalanceo mensual
# # # # # # # # # # # # # # # # # # # # # # # # # 

def simulate_equal_weight(df_prices: pd.DataFrame,
                          initial_balance: float = 10000,
                          commission: float = 0.001) -> pd.Series:
    """
    Cartera equiponderada con rebalanceo mensual (1/N por activo).

    Estrategia:
      - Asigna el mismo peso (1/N) a cada activo al inicio y en cada rebalanceo mensual
      - Entre rebalanceos los pesos varían con el mercado (algunos activos ganan mas peso)
      - Aplica comisiones sobre el turnover generado en cada rebalanceo
      - turnover: cuánto rota la cartera del último día de mes al primero(en este caso).

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).

    Returns
    -------
    pd.Series
        Valor de la cartera en cada dia, indexada por fecha.
    """
    # condición que debe cumplirse, sino, stop
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente (fecha mas antigua primero)."

    n = len(df_prices.columns)
    target_weights = np.ones(n) / n       # Pesos objetivo: siempre 1/N
    current_weights = target_weights.copy()   # Pesos actuales (varían con el mercado)
    balance = initial_balance
    values  = [initial_balance]

    returns = df_prices.pct_change().fillna(0)#(precio_hoy - precio_ayer) / precio_ayer

    for i in range(1, len(df_prices)):
        today = pd.to_datetime(df_prices.index[i])
        yesterday = pd.to_datetime(df_prices.index[i - 1])

        # Rebalanceo mensual: restaurar los pesos a 1/N el primer dia de cada mes
        if _is_month_start(today, yesterday):
            balance, current_weights = _rebalance(balance, target_weights, current_weights, commission)

        # Actualizar valor de la cartera con los retornos del dia
        rets= returns.iloc[i].values
        balance = balance * np.sum(current_weights * (1 + rets))
        balance = max(balance, 1e-6)#que nunca sea 0 exacto
        # Actualizar pesos: derivan segun el rendimiento de cada activo
        new_vals = current_weights * (1 + rets)
        current_weights = new_vals / (new_vals.sum() + 1e-8)

        values.append(balance)

    return pd.Series(values, index=df_prices.index, name='Equal_Weight_Mensual')


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Baseline 2: Cartera 60/40
# # # # # # # # # # # # # # # # # # # # # # # # # 

def simulate_60_40(df_prices: pd.DataFrame,
                   ticker_equity: str = 'IVV_Close',
                   ticker_bond: str = 'BND_Close',
                   initial_balance: float = 10000,
                   commission: float = 0.001) -> pd.Series:
    """
    Cartera clasica 60/40: 60% renta variable / 40% renta fija.

    En el contexto del TFM:
      - 60%: IVV (S&P 500 ETF, renta variable de maxima liquidez)
      - 40%: BND (Vanguard Total Bond Market ETF, renta fija diversificada)

    Rebalanceo mensual para restaurar los pesos objetivo tras el mes de negocio.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    ticker_equity : str
        Nombre de la columna de renta variable en df_prices.
    ticker_bond : str
        Nombre de la columna de renta fija en df_prices.
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).

    Returns
    -------
    pd.Series
        Valor de la cartera 60/40 en cada dia, indexada por fecha.
    """
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    if ticker_equity not in df_prices.columns or ticker_bond not in df_prices.columns:
        raise ValueError(
            f"Columnas '{ticker_equity}' o '{ticker_bond}' no encontradas. "
            f"Columnas disponibles: {list(df_prices.columns)}"
        )

    # Trabajar solo con los dos activos de la estrategia
    p = df_prices[[ticker_equity, ticker_bond]].copy()
    target_weights = np.array([0.6, 0.4])
    current_weights = target_weights.copy()
    balance = initial_balance
    values = [initial_balance]
    returns = p.pct_change().fillna(0)

    for i in range(1, len(p)):
        today     = pd.to_datetime(p.index[i])
        yesterday = pd.to_datetime(p.index[i - 1])

        if _is_month_start(today, yesterday):
            balance, current_weights = _rebalance(balance, target_weights, current_weights, commission)

        rets = returns.iloc[i].values
        balance = balance * np.sum(current_weights * (1 + rets))
        balance = max(balance, 1e-6)

        new_vals = current_weights * (1 + rets)
        current_weights = new_vals / (new_vals.sum() + 1e-8)

        values.append(balance)

    return pd.Series(values, index=p.index, name='Cartera_60_40')


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Baseline 3: Buy & Hold
# # # # # # # # # # # # # # # # # # # # # # # # # 
def simulate_buy_and_hold(df_prices: pd.DataFrame,
                          initial_balance: float = 10000) -> pd.Series:
    """
    Estrategia Buy & Hold: asignacion inicial 1/N, sin ningun rebalanceo posterior.

    El activo mas rentable ira ganando peso progresivamente (efecto drift).
    No genera costes de transaccion mas alla de la compra inicial.
    Es la estrategia pasiva de menor friccion.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    initial_balance : float
        Capital inicial en dolares.

    Returns
    -------
    pd.Series
        Valor de la cartera Buy & Hold en cada dia, indexada por fecha.
    """
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    n = len(df_prices.columns)
    weights = np.ones(n) / n
    returns = df_prices.pct_change().fillna(0)

    # Retorno de la cartera = promedio ponderado con pesos fijos iniciales
    portfolio_return = (returns * weights).sum(axis=1)
    values  = initial_balance * (1 + portfolio_return).cumprod()

    return pd.Series(values.values, index=df_prices.index, name='Buy_and_Hold')


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Baseline 4: Markowitz Media-Varianza
# # # # # # # # # # # # # # # # # # # # # # # # # 

def _maximize_sharpe(historical_returns: pd.DataFrame, annual_rf: float = 0.04) -> np.ndarray:
    """
    Optimizacion cuadratica de Markowitz: maximiza el Ratio de Sharpe.

    Problema de optimizacion:
        Sharpe = (retorno_anual - tasa_libre_de_riesgo) / volatilidad_anual
        max  (mu_p - rf) / sigma_p
        s.t. sum(w_i) = 1,  w_i >= 0  (solo posiciones largas, sin short selling)

    Usa scipy.optimize.minimize con metodo SLSQP.
    Si la optimizacion falla, retorna pesos equiponderados (fallback seguro).

    Parameters
    ----------
    historical_returns : pd.DataFrame
        DataFrame de retornos diarios historicos (ventana de estimacion).
    annual_rf : float
        Tasa libre de riesgo anualizada para el calculo del Sharpe.

    Returns
    -------
    np.ndarray
        Vector de pesos optimos normalizados (suman 1, todos >= 0).
    """
    n = historical_returns.shape[1]
    mu= historical_returns.mean().values * 252    # Retornos esperados anualizados
    sigma = historical_returns.cov().values * 252     # Matriz de covarianza anualizada

    def neg_sharpe(w):
        """Funcion objetivo negativa (minimizamos el negativo del Sharpe).
            Scypi solo sabe minimizar. Por eso el valor más negativo es maximizar sharpe.
        """
        ret_p = float(np.dot(w, mu))
        var_p = float(np.dot(w, np.dot(sigma, w)))
        vol_p = np.sqrt(max(var_p, 1e-10))
        return -(ret_p - annual_rf) / vol_p

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds= [(0.0, 1.0)] * n
    w0 = np.ones(n) / n  # Punto de partida: equiponderado

    try:
        res = minimize(
            neg_sharpe, w0,
            method='SLSQP',#secuential least squares programming
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        if res.success and np.all(np.isfinite(res.x)):
            w = np.clip(res.x, 0.0, 1.0)
            return w / (w.sum() + 1e-8)
    except Exception:
        pass  # Fallback  a igual ponderacion

    return w0  # Fallback: Equal-Weight si la optimizacion no converge


def simulate_markowitz(df_prices: pd.DataFrame,
                       estimation_window: int = 252,
                       initial_balance: float = 10000,
                       commission: float = 0.001,
                       annual_rf: float = 0.04) -> pd.Series:
    """
    Cartera de Media-Varianza de Markowitz con reoptimizacion mensual.

    Proceso mensual:
      1. Toma los ultimos 'estimation_window' dias de retornos historicos
      2. Optimiza los pesos para maximizar el Ratio de Sharpe
      3. Rebalancea la cartera (con comisiones) hacia los nuevos pesos optimos

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    estimation_window : int
        Dias de historia para estimar mu y Sigma (252 = 12 meses).
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).
    annual_rf : float
        Tasa libre de riesgo anualizada para el calculo del Sharpe.

    Returns
    -------
    pd.Series
        Valor de la cartera Markowitz en cada dia. La serie empieza
        'estimation_window' dias despues del inicio de df_prices,
        por lo que es mas corta que las demas estrategias.
    """
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    n = len(df_prices.columns)
    returns = df_prices.pct_change().dropna()

    # Adaptar ventana de estimación al tamaño del dataset
    # Si el dataset es más corto que la ventana, usamos la mitad del dataset
    estimation_window = min(estimation_window, max(20, len(df_prices) // 2))

    # Rellenar los primeros días con el capital inicial (Markowitz aún no tiene
    # suficiente historia para optimizar). Esto garantiza que la serie tenga
    # el mismo número de puntos que las demás estrategias.
    values = [initial_balance] * estimation_window
    index = list(df_prices.index[:estimation_window])
    balance = initial_balance
    current_weights = np.ones(n) / n
    last_rebalance_month = None

    for i in range(estimation_window, len(df_prices)):
        date = pd.to_datetime(df_prices.index[i])
        current_month = date.month

        # Reoptimizacion mensual con la ventana de historia mas reciente
        if current_month != last_rebalance_month:
            hist_returns = returns.iloc[i - estimation_window: i]
            new_weights = _maximize_sharpe(hist_returns, annual_rf)

            balance, current_weights = _rebalance(
                balance, new_weights, current_weights, commission
            )
            last_rebalance_month = current_month

        # Actualizar valor con el retorno del dia
        # returns.iloc[i-1] = retorno del dia i respecto al dia i-1
        if i - 1 < len(returns):
            rets    = returns.iloc[i - 1].values
            balance = balance * np.sum(current_weights * (1 + rets))
            balance = max(balance, 1e-6)

            new_vals = current_weights * (1 + rets)
            current_weights = new_vals / (new_vals.sum() + 1e-8)

        values.append(balance)
        index.append(df_prices.index[i])

    return pd.Series(values, index=index, name='Markowitz_MV')


# # # # # # # # # # # # # # # # # # # # # # # # # 
# Ejecucion conjunta de baselines
# # # # # # # # # # # # # # # # # # # # # # # # # 

def run_baselines(df_prices: pd.DataFrame,
                  initial_balance: float = 10000,
                  commission: float = 0.001,
                  ticker_equity: str = 'IVV_Close',
                  ticker_bond: str = 'BND_Close') -> dict:
    """
    Ejecuta los seis baselines sobre el mismo conjunto de precios.

    Estrategias incluidas:
      - Equal_Weight_Mensual: 1/N con rebalanceo mensual
      - Buy_and_Hold: 1/N inicial sin rebalanceo
      - Cartera_60_40: 60% IVV / 40% BND (solo si ambos disponibles)
      - Markowitz_MV: media-varianza con ventana 252d, reoptimizacion mensual
      - Random_Uniform: pesos aleatorios del simplex, rebalanceo mensual (seed fijo)
      - Momentum_TopK: top-3 por momentum a 60d, rebalanceo mensual

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).
    ticker_equity : str
        Nombre de la columna de renta variable para la estrategia 60/40.
    ticker_bond : str
        Nombre de la columna de renta fija para la estrategia 60/40.

    Returns
    -------
    dict
        Diccionario {nombre_estrategia: pd.Series de valores de cartera}.
    """
    results = {}

    print("  Simulando Equal-Weight mensual...")
    results['Equal_Weight_Mensual'] = simulate_equal_weight(
        df_prices, initial_balance, commission
    )

    print("  Simulando Buy & Hold...")
    results['Buy_and_Hold'] = simulate_buy_and_hold(df_prices, initial_balance)

    # 60/40 solo si estan disponibles IVV y BND
    if ticker_equity in df_prices.columns and ticker_bond in df_prices.columns:
        print("  Simulando cartera 60/40...")
        results['Cartera_60_40'] = simulate_60_40(
            df_prices, ticker_equity, ticker_bond, initial_balance, commission
        )
    else:
        print(f"  [AVISO] 60/40 no disponible: columnas '{ticker_equity}' o '{ticker_bond}' ausentes.")

    print("  Simulando Markowitz Media-Varianza (puede tardar unos segundos)...")
    results['Markowitz_MV'] = simulate_markowitz(df_prices, initial_balance=initial_balance)

    print("  Simulando Random Uniform (lower bound de cordura)...")
    results['Random_Uniform'] = simulate_random_uniform(
        df_prices, initial_balance, commission
    )

    print("  Simulando Momentum Top-K (factor momentum cross-sectional)...")
    results['Momentum_TopK'] = simulate_momentum_topk(
        df_prices, lookback=60, top_k=3,
        initial_balance=initial_balance, commission=commission
    )

    return results


# # # # # # # # # # # # # # # # # # # # # # # # #
# Baseline 5: Random Uniform (lower bound de cordura)
# # # # # # # # # # # # # # # # # # # # # # # # #

def simulate_random_uniform(df_prices: pd.DataFrame,
                            initial_balance: float = 10000,
                            commission: float = 0.001,
                            seed: int = 42) -> pd.Series:
    """
    Cartera con pesos uniformemente aleatorios sobre el simplex, rebalanceo mensual.

    En cada primer dia de mes se muestrean nuevos pesos w ~ Dirichlet(1, ..., 1)
    (equivalente a muestrear uniformemente del simplex de pesos no-negativos que
    suman 1) y se rebalancea la cartera a esos pesos, aplicando comisiones por el
    turnover generado. Entre rebalanceos los pesos derivan con el mercado.

    Rol en el TFM:
      Es la baseline mas elemental: lower bound de cordura. Si el agente DRL no
      supera consistentemente al Random Uniform en metricas ajustadas por riesgo
      (Sharpe, Sortino), entonces el aprendizaje del agente no esta aportando
      valor por encima del azar y la propuesta del TFM no se sostiene. Es la
      primera baseline que un tribunal exigira para descartar suerte.

    Reproducibilidad:
      Se fija una seed (default 42) para que el resultado sea identico en
      ejecuciones sucesivas. Si se desea estimar la varianza del random (no
      una sola realizacion), llamar varias veces con seeds distintas y promediar.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).
    seed : int
        Semilla para el generador aleatorio (reproducibilidad).

    Returns
    -------
    pd.Series
        Valor de la cartera en cada dia, indexada por fecha.
    """
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    rng = np.random.default_rng(seed)
    n = len(df_prices.columns)

    # Pesos iniciales: muestra uniforme del simplex (Dirichlet con alphas=1)
    current_weights = rng.dirichlet(np.ones(n))
    balance = initial_balance
    values = [initial_balance]
    returns = df_prices.pct_change().fillna(0)

    for i in range(1, len(df_prices)):
        today = pd.to_datetime(df_prices.index[i])
        yesterday = pd.to_datetime(df_prices.index[i - 1])

        # Rebalanceo mensual: nuevos pesos aleatorios del simplex
        if _is_month_start(today, yesterday):
            target_weights = rng.dirichlet(np.ones(n))
            balance, current_weights = _rebalance(
                balance, target_weights, current_weights, commission
            )

        rets = returns.iloc[i].values
        balance = balance * np.sum(current_weights * (1 + rets))
        balance = max(balance, 1e-6)

        new_vals = current_weights * (1 + rets)
        current_weights = new_vals / (new_vals.sum() + 1e-8)

        values.append(balance)

    return pd.Series(values, index=df_prices.index, name='Random_Uniform')


# # # # # # # # # # # # # # # # # # # # # # # # #
# Baseline 6: Momentum Top-K (factor momentum cross-sectional)
# # # # # # # # # # # # # # # # # # # # # # # # #

def simulate_momentum_topk(df_prices: pd.DataFrame,
                           lookback: int = 60,
                           top_k: int = 3,
                           initial_balance: float = 10000,
                           commission: float = 0.001) -> pd.Series:
    """
    Cartera momentum cross-sectional con rebalanceo mensual.

    En cada primer dia de mes:
      1. Calcula el retorno acumulado de cada activo en los ultimos `lookback` dias.
      2. Selecciona los `top_k` activos con mejor retorno (mayor momentum).
      3. Asigna peso 1/top_k a cada uno de esos activos; resto a 0.
      4. Aplica comisiones por el turnover.

    Antes de disponer de `lookback` dias de historico, la cartera mantiene
    una asignacion equiponderada (1/N) entre todos los activos.

    Justificacion academica:
      El factor momentum (Jegadeesh & Titman, 1993) documenta que los activos
      que han subido en los ultimos 6-12 meses tienden a seguir subiendo en el
      corto plazo. Con `lookback=60` (~3 meses) se captura un momentum de medio
      plazo. Es un baseline competitivo del sector cuantitativo: si el agente
      DRL solo replica momentum, no aporta valor sobre esta baseline.

    Rol en el TFM:
      Sirve para verificar que el agente PPO no es solo un proxy del momentum,
      sino que aprende una politica con valor anadido (por ejemplo, anticipar
      cambios de regimen y reducir exposicion antes de las correcciones).

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre (columnas = activos, indice = fechas ASC).
    lookback : int
        Numero de dias hacia atras para calcular el momentum (default 60 ~ 3 meses).
    top_k : int
        Numero de activos a mantener en la cartera tras el ranking de momentum.
        Si top_k > n_activos, se usa min(top_k, n_activos).
    initial_balance : float
        Capital inicial en dolares.
    commission : float
        Tasa de comision por transaccion (0.001 = 0.1%).

    Returns
    -------
    pd.Series
        Valor de la cartera en cada dia, indexada por fecha.
    """
    assert df_prices.index.is_monotonic_increasing, \
        "Los precios deben estar ordenados ascendentemente."

    n = len(df_prices.columns)
    k = min(top_k, n)

    # Pesos iniciales: equiponderado hasta tener histórico suficiente
    current_weights = np.ones(n) / n
    balance = initial_balance
    values = [initial_balance]
    returns = df_prices.pct_change().fillna(0)
    prices_arr = df_prices.values  # acceso rapido por indice

    for i in range(1, len(df_prices)):
        today = pd.to_datetime(df_prices.index[i])
        yesterday = pd.to_datetime(df_prices.index[i - 1])

        if _is_month_start(today, yesterday):
            if i >= lookback:
                # Retorno acumulado en ventana lookback: (p_t / p_{t-lookback}) - 1
                past = prices_arr[i - lookback]
                now = prices_arr[i]
                # Evitar division por cero o NaN
                with np.errstate(divide='ignore', invalid='ignore'):
                    momentum = np.where(past > 0, now / past - 1.0, -np.inf)
                # Top-K activos por momentum descendente
                top_idx = np.argsort(momentum)[-k:]
                target_weights = np.zeros(n)
                target_weights[top_idx] = 1.0 / k
            else:
                target_weights = np.ones(n) / n  # fallback equiponderado

            balance, current_weights = _rebalance(
                balance, target_weights, current_weights, commission
            )

        rets = returns.iloc[i].values
        balance = balance * np.sum(current_weights * (1 + rets))
        balance = max(balance, 1e-6)

        new_vals = current_weights * (1 + rets)
        current_weights = new_vals / (new_vals.sum() + 1e-8)

        values.append(balance)

    return pd.Series(values, index=df_prices.index, name='Momentum_TopK')


# # # # # # # # # # # # # # # # # # # # # # # # #
# Metricas financieras
# # # # # # # # # # # # # # # # # # # # # # # # #

def compute_metrics(series: pd.Series, annual_rf: float = 0.04) -> dict:
    """
    Calcula el conjunto estandar de metricas de rendimiento financiero. 
    Se calculan retornos diarios y se escalan a unidades anuales.

    Metricas calculadas:
      - Retorno Total (%):(V_final / V_inicial - 1) x 100
      - CAGR (%): Tasa de crecimiento anual compuesta
      - Volatilidad Anualizada (%): Desviacion estandar diaria x sqrt(252). Mira cuanto oscila el valor diariamente.
      - Sharpe Ratio: Exceso de retorno / Volatilidad total
      - Sortino Ratio: exceso de retorno / Volatilidad negativa
                        (solo penaliza los retornos negativos)
      - Max Drawdown (%):  Mayor caida acumulada desde un maximo historico
      - Valor Final ($): Valor absoluto final de la cartera

    Parameters
    ----------
    series : pd.Series
        Serie de valores absolutos de cartera (ej. empezando en 10000).
    annual_rf : float
        Tasa libre de riesgo anualizada para el calculo de Sharpe y Sortino.

    Returns
    -------
    dict
        Diccionario con las metricas calculadas y sus valores redondeados.
    """
    returns = series.pct_change().dropna()
    n_days  = len(returns)

    if n_days < 2:
        return {'Retorno Total (%)': 0, 'CAGR (%)': 0, 'Volatilidad Anualizada (%)': 0,
                'Sharpe Ratio': 0, 'Sortino Ratio': 0,
                'Max Drawdown (%)': 0, 'Valor Final ($)': float(series.iloc[-1])}

    total_return  = (series.iloc[-1] / series.iloc[0]) - 1
    cagr  = (1 + total_return) ** (252 / n_days) - 1
    annual_vol  = returns.std() * np.sqrt(252)

    daily_rf = annual_rf / 252
    excess_ret = returns - daily_rf

    # Sharpe: exceso de retorno sobre volatilidad total
    sharpe = (excess_ret.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

    # Sortino: exceso de retorno sobre volatilidad de los retornos negativos
    negative_returns = returns[returns < daily_rf]
    downside_std  = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 1e-8
    sortino   = (excess_ret.mean() * 252) / downside_std

    # Maximum Drawdown: mayor caida desde un maximo historico
    rolling_max = series.cummax()
    drawdown  = (series - rolling_max) / (rolling_max + 1e-8)
    max_dd  = drawdown.min()

    return {
        'Retorno Total (%)':  round(total_return * 100, 2),
        'CAGR (%)':round(cagr * 100, 2),
        'Volatilidad Anualizada (%)': round(annual_vol * 100, 2),
        'Sharpe Ratio': round(sharpe, 3),
        'Sortino Ratio': round(sortino, 3),
        'Max Drawdown (%)': round(max_dd * 100, 2),
        'Valor Final ($)': round(float(series.iloc[-1]), 2),
    }


def comparison_table(results: dict, annual_rf: float = 0.04) -> pd.DataFrame:
    """
    Genera una tabla comparativa de metricas para multiples estrategias.

    Parameters
    ----------
    results : dict
        Diccionario {nombre_estrategia: pd.Series de valores de cartera}.
    annual_rf : float
        Tasa libre de riesgo anualizada para el calculo de metricas.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por estrategia y una columna por metrica
        (listo para exportar a LaTeX o CSV para la memoria del TFM).
    """
    rows = {}
    for name, series in results.items():
        if series is not None and len(series) > 1:
            rows[name] = compute_metrics(series, annual_rf)

    return pd.DataFrame(rows).T


# ─────────────────────────────────────────────
# Aliases de compatibilidad hacia atras
# ─────────────────────────────────────────────
# Estos nombres se importan externamente; se mantienen como aliases para
# no romper el codigo existente que depende de ellos.

ejecutar_baselines = run_baselines
calcular_metricas = compute_metrics
tabla_comparativa = comparison_table
