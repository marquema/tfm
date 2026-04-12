"""
Módulo de análisis de regímenes de volatilidad y cambios de mercado.

Objetivo del TFM:
  Validar la robustez del agente DRL ante cambios de régimen:
    - Régimen de ALTA volatilidad: correcciones bruscas en el mercado cripto (IBIT)
    - Régimen de BAJA volatilidad: calma en renta fija y renta variable estable

Metodología:
  Clasificación basada en percentiles de volatilidad rolling del activo de referencia.
  Criterio:
    - Bajo (0):  vol_20d <= percentil 33 de vol_20d histórico (período de entrenamiento)
    - Medio (1): percentil 33 < vol_20d <= percentil 67
    - Alto (2):  vol_20d > percentil 67

  No requiere dependencias externas adicionales (hmmlearn es opcional).
  La clasificación usa solo el período de ENTRENAMIENTO para calcular los umbrales,
  evitando así el data leakage al analizar el período de test.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla para guardar PNGs en servidor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("src/reports", exist_ok=True)


# ---------------------------------------------------------------------------
# Clasificación de regímenes
# ---------------------------------------------------------------------------

def classify_regimes(df_prices: pd.DataFrame,
                     reference_ticker: str = None,
                     window: int = 20,
                     low_percentile: float = 33,
                     high_percentile: float = 67,
                     train_threshold_pct: float = 0.8) -> pd.Series:
    """
    Clasifica cada día del conjunto de precios en un régimen de volatilidad.

    Régimen 0 (BAJA):  volatilidad <= percentil bajo del período de entrenamiento.
    Régimen 1 (MEDIA): percentil bajo < volatilidad <= percentil alto.
    Régimen 2 (ALTA):  volatilidad > percentil alto.

    Los umbrales se calculan exclusivamente sobre el período de entrenamiento
    para no contaminar el análisis out-of-sample.

    Parameters
    ----------
    df_prices : pd.DataFrame
        DataFrame de precios de cierre en orden cronológico ascendente.
    reference_ticker : str or None
        Columna de referencia para calcular volatilidad.
        Por defecto: primera columna con 'IBIT' en el nombre, o la primera
        columna del DataFrame si no hay IBIT.
    window : int
        Días para el cálculo de volatilidad rolling (por defecto 20).
    low_percentile : float
        Percentil inferior para separar régimen bajo del medio (0-100).
    high_percentile : float
        Percentil superior para separar régimen medio del alto (0-100).
    train_threshold_pct : float
        Fracción de datos usada como período de entrenamiento para calibrar
        los umbrales sin data leakage (0.0-1.0).

    Returns
    -------
    pd.Series
        Serie con régimen (0, 1, 2) para cada fecha, indexada igual que df_prices.
    """
    # Selección automática del activo de referencia
    if reference_ticker is None:
        ibit_cols = [c for c in df_prices.columns if 'IBIT' in c.upper()]
        reference_ticker = ibit_cols[0] if ibit_cols else df_prices.columns[0]

    if reference_ticker not in df_prices.columns:
        raise ValueError(
            f"Ticker de referencia '{reference_ticker}' no encontrado en df_prices."
        )

    # Retornos logarítmicos del activo de referencia
    ref_prices = df_prices[reference_ticker]
    log_returns = np.log(ref_prices / ref_prices.shift(1))
    rolling_vol = log_returns.rolling(window).std()

    # Umbrales calibrados solo sobre el período de entrenamiento
    n_train = int(len(rolling_vol) * train_threshold_pct)
    vol_train = rolling_vol.iloc[:n_train].dropna()

    threshold_low  = np.percentile(vol_train, low_percentile)
    threshold_high = np.percentile(vol_train, high_percentile)

    # Clasificación: 0=Baja, 1=Media, 2=Alta
    regime = pd.Series(1, index=df_prices.index, name='regime', dtype=int)
    regime[rolling_vol <= threshold_low]  = 0
    regime[rolling_vol > threshold_high]  = 2
    regime[rolling_vol.isna()]= 1  # NaN al inicio -> régimen neutro

    return regime


# ---------------------------------------------------------------------------
# Ejecución del agente por subperíodo
# ---------------------------------------------------------------------------
def _run_agent(model, features_path: str, prices_path: str,
               start_idx: int, end_idx: int) -> list:
    """
    Ejecuta el agente en un rango de índices del dataset y recoge los valores de cartera.

    Importación tardía de PortfolioEnv para evitar dependencia circular.

    Parameters
    ----------
    model : stable_baselines3.PPO
        Modelo PPO ya cargado/entrenado.
    features_path : str
        Ruta al CSV de features normalizadas.
    prices_path : str
        Ruta al CSV de precios originales.
    start_idx : int
        Índice de inicio del subconjunto.
    end_idx : int or None
        Índice final del subconjunto (None = hasta el final).

    Returns
    -------
    list[float]
        Lista de valores de cartera en cada paso (incluido el valor inicial).
    """
    try:
        from src.training_drl.environment_trading import PortfolioEnv
    except ImportError:
        from src.training_drl.environment_trading import PortfolioEnv

    env = PortfolioEnv(features_path, prices_path,
                       start_idx=start_idx, end_idx=end_idx)
    obs, _ = env.reset()
    done   = False
    values = [env.initial_balance]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        values.append(info['value'])

    return values


# ---------------------------------------------------------------------------
# Análisis de métricas por régimen
# ---------------------------------------------------------------------------

def metrics_by_regime(value_series: pd.Series,
                      regimes: pd.Series,
                      annual_rf: float = 0.04) -> pd.DataFrame:
    """
    Calcula métricas financieras para cada régimen de volatilidad.

    Segmenta la serie de valores de cartera según el régimen activo en cada
    fecha y calcula retorno total, volatilidad anualizada, Sharpe, Sortino
    y max drawdown para cada segmento.

    Parameters
    ----------
    value_series : pd.Series
        Serie de valores de cartera indexada por fecha.
    regimes : pd.Series
        Serie con régimen (0, 1, 2) para cada fecha (mismo índice que value_series).
    annual_rf : float
        Tasa libre de riesgo anualizada para el cálculo de Sharpe/Sortino.

    Returns
    -------
    pd.DataFrame
        DataFrame con regímenes como índice y métricas como columnas.
        Vacío si no hay suficientes datos en ningún régimen.
    """
    labels = {0: 'Baja Volatilidad', 1: 'Vol. Media', 2: 'Alta Volatilidad'}
    rows   = {}

    # Alinear los dos índices por intersección
    common_index = value_series.index.intersection(regimes.index)
    aligned_values  = value_series.loc[common_index]
    aligned_regimes = regimes.loc[common_index]

    for code, name in labels.items():
        mask   = aligned_regimes == code
        n_days = mask.sum()

        if n_days < 10:
            # Insuficientes días para calcular métricas robustas
            continue

        subset = aligned_values[mask].reset_index(drop=True)

        # Reconstruir serie continua para métricas correctas
        returns = subset.pct_change().dropna()

        if len(returns) < 2:
            continue

        total_return = (subset.iloc[-1] / subset.iloc[0]) - 1
        annual_vol = returns.std() * np.sqrt(252)
        daily_rf = annual_rf / 252
        excess = returns - daily_rf
        sharpe = (excess.mean() / (returns.std() + 1e-8)) * np.sqrt(252)

        neg_returns  = returns[returns < daily_rf]
        downside_std = (neg_returns.std() * np.sqrt(252)
                        if len(neg_returns) > 1 else 1e-8)
        sortino = (excess.mean() * 252) / downside_std

        rolling_max  = subset.cummax()
        max_dd = ((subset - rolling_max) / (rolling_max + 1e-8)).min()

        rows[name] = {
            'N días': int(n_days),
            'Retorno Total (%)': round(total_return * 100, 2),
            'Volatilidad Anualizada (%)':  round(annual_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 3),
            'Sortino Ratio': round(sortino, 3),
            'Max Drawdown (%)': round(max_dd * 100, 2),
        }

    return pd.DataFrame(rows).T if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Análisis completo: agente + baselines por régimen
# ---------------------------------------------------------------------------
def analyze_regimes(features_path: str = 'data/normalized_features.csv',
                    prices_path: str= 'data/original_prices.csv',
                    model_path: str  = None,
                    split_pct: float = 0.8,
                    initial_balance: float = 10000) -> dict:
    """
    Comparativo del agente DRL vs baselines segmentado por régimen de volatilidad.

    Proceso completo:
      1. Carga datos y clasifica régimen para todo el período.
      2. Ejecuta el agente en el conjunto de test (si se proporciona model_path).
      3. Ejecuta los baselines (Buy & Hold, Equal-Weight) en el conjunto de test.
      4. Calcula métricas por régimen para cada estrategia.
      5. Genera visualización y guarda resultados en src/reports/.

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas.
    prices_path : str
        Ruta al CSV de precios originales.
    model_path : str or None
        Ruta al modelo PPO entrenado (.zip). Si es None, solo se analiza
        la distribución de regímenes sin ejecutar el agente.
    split_pct : float
        Fracción de datos usada como entrenamiento (el test empieza después).
    initial_balance : float
        Capital inicial de la cartera en USD.

    Returns
    -------
    dict
        Diccionario con claves: 'regimenes', 'Buy_and_Hold', 'Equal_Weight',
        'IA_PPO' (si hay modelo), y 'metricas_*' con DataFrames por estrategia.
    """
    # Borrar reportes anteriores para evitar mostrar datos incoherentes
    for old_file in ['src/reports/regime_analysis.png',
                     'src/reports/regime_metrics.csv']:
        if os.path.exists(old_file):
            os.remove(old_file)

    # Carga de datos
    df_features = pd.read_csv(features_path, index_col=0)
    df_prices   = pd.read_csv(prices_path,   index_col=0)
    df_features.index = pd.to_datetime(df_features.index)
    df_prices.index   = pd.to_datetime(df_prices.index)

    split_idx = int(len(df_features) * split_pct)

    # Clasificación de regímenes sobre todo el dataset (umbrales del train)
    full_regimes = classify_regimes(df_prices, train_threshold_pct=split_pct)
    test_regimes = full_regimes.iloc[split_idx:]
    test_prices  = df_prices.iloc[split_idx:]

    print(f"\nDistribución de regímenes en el período de test ({len(test_prices)} días):")
    distribution = test_regimes.value_counts().sort_index()
    for code, n in distribution.items():
        name = {0: 'Baja Vol.', 1: 'Vol. Media', 2: 'Alta Vol.'}[code]
        print(f"  Régimen {code} ({name}): {n} días ({n/len(test_regimes)*100:.1f}%)")

    results = {'regimenes': test_regimes}

    # Baselines simples (no requieren modelo)
    try:
        from src.benchmarking.baselines import simulate_buy_and_hold, simulate_equal_weight
    except ImportError:
        from src.benchmarking.baselines import simulate_buy_and_hold, simulate_equal_weight

    series_bh = simulate_buy_and_hold(test_prices, initial_balance)
    series_ew = simulate_equal_weight(test_prices, initial_balance)

    results['Buy_and_Hold'] = series_bh
    results['Equal_Weight'] = series_ew

    print("\nMétricas por régimen — Buy & Hold:")
    df_bh = metrics_by_regime(series_bh, test_regimes)
    if not df_bh.empty:
        print(df_bh.to_string())
        results['metricas_bh'] = df_bh

    print("\nMétricas por régimen — Equal-Weight:")
    df_ew = metrics_by_regime(series_ew, test_regimes)
    if not df_ew.empty:
        print(df_ew.to_string())
        results['metricas_ew'] = df_ew

    # Agente DRL (solo si se proporciona el modelo)
    if model_path and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model     = PPO.load(model_path)
            ia_values = _run_agent(model, features_path, prices_path, split_idx, None)
            series_ia = pd.Series(
                ia_values[:len(test_prices)],
                index=test_prices.index,
                name='IA_PPO'
            )
            results['IA_PPO'] = series_ia

            print("\nMétricas por régimen — Agente IA (PPO):")
            df_ia = metrics_by_regime(series_ia, test_regimes)
            if not df_ia.empty:
                print(df_ia.to_string())
                results['metricas_ia'] = df_ia
        except Exception as e:
            print(f"  [AVISO] No se pudo ejecutar el agente: {e}")

    # todo: faltará implementar comparativa frente al agente especulativo?
    
    # Visualización
    _plot_regimes(results, test_regimes, test_prices)
    _save_metrics(results)

    return results


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

# Paleta de colores para los regímenes
_REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
_REGIME_NAMES  = {0: 'Baja Volatilidad', 1: 'Volatilidad Media', 2: 'Alta Volatilidad'}


def _shade_backgrounds(ax, regimes: pd.Series) -> None:
    """
    Sombrea el fondo de un eje matplotlib según el régimen de volatilidad activo.

    Verde = Baja volatilidad, Naranja = Media, Rojo = Alta volatilidad.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde aplicar el sombreado.
    regimes : pd.Series
        Serie de regímenes indexada por fecha.

    Returns
    -------
    None
    """
    if regimes.empty:
        return

    # Detectar bloques continuos del mismo régimen
    changes = regimes.ne(regimes.shift()).cumsum()
    for _, block in regimes.groupby(changes):
        if block.empty:
            continue
        code  = int(block.iloc[0])
        start = block.index[0]
        end   = block.index[-1]
        ax.axvspan(start, end,
                   alpha=0.15,
                   color=_REGIME_COLORS.get(code, 'gray'),
                   linewidth=0)


def _plot_regimes(results: dict, test_regimes: pd.Series,
                  test_prices: pd.DataFrame) -> None:
    """
    Genera la gráfica principal de análisis de regímenes.

    Panel superior: evolución de carteras con zonas sombreadas por régimen.
    Panel inferior: indicador de régimen a lo largo del tiempo.

    Parameters
    ----------
    results : dict
        Diccionario con series de valores por estrategia.
    test_regimes : pd.Series
        Regímenes del período de test.
    test_prices : pd.DataFrame
        Precios del período de test (para dimensionar el eje X).

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax1, ax2 = axes

    # -- Panel superior: evolución de carteras --
    strategy_styles = {
        'IA_PPO':          ('#1f77b4', '-',  2.5),
        'Buy_and_Hold':    ('#ff7f0e', '--', 1.5),
        'Equal_Weight':    ('#2ca02c', '--', 1.5),
        'Cartera_60_40':   ('#9467bd', '--', 1.5),
        'Markowitz_MV':    ('#8c564b', ':',  1.5),
    }

    for name, (color, style, width) in strategy_styles.items():
        if name in results and results[name] is not None:
            series = results[name]
            ax1.plot(series.index, series.values,
                     label=name, color=color, linestyle=style, linewidth=width)

    # Fondo sombreado por régimen
    _shade_backgrounds(ax1, test_regimes)

    # Leyenda de regímenes
    patches = [
        mpatches.Patch(color=_REGIME_COLORS[k], alpha=0.5, label=_REGIME_NAMES[k])
        for k in sorted(_REGIME_COLORS.keys())
    ]
    ax1.legend(
        handles=ax1.get_legend_handles_labels()[0] + patches,
        labels=ax1.get_legend_handles_labels()[1] + [p.get_label() for p in patches],
        loc='upper left', fontsize=9
    )
    ax1.set_title('Análisis de Regímenes de Volatilidad — Conjunto Out-of-Sample',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valor de Cartera ($)')
    ax1.grid(True, alpha=0.3)

    # -- Panel inferior: indicador de régimen --
    regime_colors = [_REGIME_COLORS.get(int(v), 'gray')
                     for v in test_regimes.values]
    ax2.bar(test_regimes.index, test_regimes.values,
            color=regime_colors, width=1, alpha=0.8)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Baja', 'Media', 'Alta'])
    ax2.set_title('Indicador de Régimen (0=Baja, 1=Media, 2=Alta Volatilidad)')
    ax2.set_ylabel('Régimen')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = 'src/reports/regime_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nGráfica de regímenes guardada: {output_path}")
    plt.close(fig)


def _save_metrics(results: dict) -> None:
    """
    Guarda las tablas de métricas por régimen en CSV para la memoria del TFM.

    Parameters
    ----------
    results : dict
        Diccionario con claves 'metricas_ia', 'metricas_bh', 'metricas_ew'
        conteniendo DataFrames de métricas por régimen.

    Returns
    -------
    None
    """
    output_path = 'src/reports/regime_metrics.csv'
    tables = []

    for key in ['metricas_ia', 'metricas_bh', 'metricas_ew']:
        if key in results and not results[key].empty:
            strategy_name = key.replace('metricas_', '').upper()
            df = results[key].copy()
            df.insert(0, 'Estrategia', strategy_name)
            tables.append(df)

    if tables:
        pd.concat(tables).to_csv(output_path, encoding='utf-8-sig')
        print(f"Métricas por régimen guardadas: {output_path}")


# ---------------------------------------------------------------------------
# Compatibilidad hacia atrás: aliases de funciones y nombres renombrados
# ---------------------------------------------------------------------------
clasificar_regimenes = classify_regimes
metricas_por_regimen = metrics_by_regime
analizar_regimenes = analyze_regimes


# ---------------------------------------------------------------------------
# Ejecución directa
# ---------------------------------------------------------------------------

#if __name__ == "__main__":
#    resultados = analyze_regimes(
#        features_path='data/normalized_features.csv',
#        prices_path='data/original_prices.csv',
#        model_path='models/best_model/best_model.zip',
#    )
