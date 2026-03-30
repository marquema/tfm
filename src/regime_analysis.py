"""
Módulo de análisis de regímenes de volatilidad y cambios de mercado.

Objetivo del TFM:
  Validar la robustez del agente DRL ante cambios de régimen:
    - Régimen de ALTA volatilidad: correcciones bruscas en el mercado cripto (IBIT)
    - Régimen de BAJA volatilidad: calma en renta fija y renta variable estable

Metodología:
  Clasificación basada en percentiles de volatilidad rolling del activo de referencia.
  Criterio:
    - Bajo (0):  vol_20d ≤ percentil 33 de vol_20d histórico (período de entrenamiento)
    - Medio (1): percentil 33 < vol_20d ≤ percentil 67
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


# ─────────────────────────────────────────────
# Clasificación de regímenes
# ─────────────────────────────────────────────

def clasificar_regimenes(df_precios: pd.DataFrame,
                          ticker_ref: str = None,
                          ventana: int = 20,
                          percentil_bajo: float = 33,
                          percentil_alto: float = 67,
                          umbral_train_pct: float = 0.8) -> pd.Series:
    """
    Clasifica cada día del conjunto de precios en un régimen de volatilidad.

    Régimen 0 (BAJA):  volatilidad ≤ percentil_bajo del período de entrenamiento
    Régimen 1 (MEDIA): percentil_bajo < volatilidad ≤ percentil_alto
    Régimen 2 (ALTA):  volatilidad > percentil_alto

    Los umbrales se calculan exclusivamente sobre el período de entrenamiento
    para no contaminar el análisis out-of-sample.

    Parámetros
    ----------
    df_precios        : DataFrame de precios de cierre en orden ascendente
    ticker_ref        : columna de referencia para calcular volatilidad
                        (por defecto: primera columna con 'IBIT' en el nombre,
                         o la primera columna si no hay IBIT)
    ventana           : días para el cálculo de volatilidad rolling
    percentil_bajo    : percentil inferior para separar régimen bajo del medio
    percentil_alto    : percentil superior para separar régimen medio del alto
    umbral_train_pct  : fracción de datos usada como período de entrenamiento
                        (para calibrar los umbrales sin data leakage)

    Retorna
    -------
    pd.Series con régimen (0, 1, 2) para cada fecha, indexada igual que df_precios
    """
    # Selección automática del activo de referencia
    if ticker_ref is None:
        cols_ibit = [c for c in df_precios.columns if 'IBIT' in c.upper()]
        ticker_ref = cols_ibit[0] if cols_ibit else df_precios.columns[0]

    if ticker_ref not in df_precios.columns:
        raise ValueError(f"Ticker de referencia '{ticker_ref}' no encontrado en df_precios.")

    # Retornos logarítmicos del activo de referencia
    precios_ref = df_precios[ticker_ref]
    retornos    = np.log(precios_ref / precios_ref.shift(1))
    vol_rolling = retornos.rolling(ventana).std()

    # Umbrales calibrados solo sobre el período de entrenamiento
    n_train    = int(len(vol_rolling) * umbral_train_pct)
    vol_train  = vol_rolling.iloc[:n_train].dropna()

    p_bajo = np.percentile(vol_train, percentil_bajo)
    p_alto = np.percentile(vol_train, percentil_alto)

    # Clasificación: 0=Baja, 1=Media, 2=Alta
    regimen = pd.Series(1, index=df_precios.index, name='regimen', dtype=int)
    regimen[vol_rolling <= p_bajo]  = 0
    regimen[vol_rolling > p_alto]   = 2
    regimen[vol_rolling.isna()]     = 1  # NaN al inicio → régimen neutro

    return regimen


# ─────────────────────────────────────────────
# Ejecución del agente por subperíodo
# ─────────────────────────────────────────────

def _ejecutar_agente(modelo, features_path: str, prices_path: str,
                     start_idx: int, end_idx: int) -> list:
    """
    Ejecuta el agente en un rango de índices del dataset y recoge los valores de cartera.
    Importación tardía de PortfolioEnv para evitar dependencia circular.
    """
    try:
        from src.environment_trading import PortfolioEnv
    except ImportError:
        from environment_trading import PortfolioEnv

    env = PortfolioEnv(features_path, prices_path, start_idx=start_idx, end_idx=end_idx)
    obs, _ = env.reset()
    done   = False
    valores = [env.initial_balance]

    while not done:
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(accion)
        valores.append(info['value'])

    return valores


# ─────────────────────────────────────────────
# Análisis de métricas por régimen
# ─────────────────────────────────────────────

def metricas_por_regimen(serie_valores: pd.Series,
                          regimenes: pd.Series,
                          rf_anual: float = 0.04) -> pd.DataFrame:
    """
    Calcula métricas financieras para cada régimen de volatilidad.

    Parámetros
    ----------
    serie_valores : pd.Series de valores de cartera indexada por fecha
    regimenes     : pd.Series con régimen (0,1,2) para cada fecha (mismo índice)

    Retorna
    -------
    DataFrame con regímenes como índice y métricas como columnas
    """
    etiquetas = {0: 'Baja Volatilidad', 1: 'Vol. Media', 2: 'Alta Volatilidad'}
    filas     = {}

    # Alinear los dos índices por intersección
    indice_comun = serie_valores.index.intersection(regimenes.index)
    valores_al   = serie_valores.loc[indice_comun]
    regimen_al   = regimenes.loc[indice_comun]

    for codigo, nombre in etiquetas.items():
        mask   = regimen_al == codigo
        n_dias = mask.sum()

        if n_dias < 10:
            # Insuficientes días para calcular métricas robustas
            continue

        sub = valores_al[mask].reset_index(drop=True)

        # Reconstruir serie continua para métricas correctas
        retornos = sub.pct_change().dropna()

        if len(retornos) < 2:
            continue

        retorno_total = (sub.iloc[-1] / sub.iloc[0]) - 1
        vol_anual     = retornos.std() * np.sqrt(252)
        rf_diaria     = rf_anual / 252
        exceso        = retornos - rf_diaria
        sharpe        = (exceso.mean() / (retornos.std() + 1e-8)) * np.sqrt(252)

        rets_neg      = retornos[retornos < rf_diaria]
        downside_std  = rets_neg.std() * np.sqrt(252) if len(rets_neg) > 1 else 1e-8
        sortino       = (exceso.mean() * 252) / downside_std

        rolling_max   = sub.cummax()
        max_dd        = ((sub - rolling_max) / (rolling_max + 1e-8)).min()

        filas[nombre] = {
            'N días':                      int(n_dias),
            'Retorno Total (%)':           round(retorno_total * 100, 2),
            'Volatilidad Anualizada (%)':  round(vol_anual * 100, 2),
            'Sharpe Ratio':                round(sharpe, 3),
            'Sortino Ratio':               round(sortino, 3),
            'Max Drawdown (%)':            round(max_dd * 100, 2),
        }

    return pd.DataFrame(filas).T if filas else pd.DataFrame()


# ─────────────────────────────────────────────
# Análisis completo: agente + baselines por régimen
# ─────────────────────────────────────────────

def analizar_regimenes(features_path: str = 'data/normalized_features.csv',
                        prices_path: str   = 'data/original_prices.csv',
                        modelo_path: str   = None,
                        split_pct: float   = 0.8,
                        initial_balance: float = 10000) -> dict:
    """
    Análisis comparativo del agente DRL vs baselines segmentado por régimen.

    Proceso:
      1. Carga datos y clasifica régimen para todo el período
      2. Ejecuta el agente en el conjunto de test (si se proporciona modelo_path)
      3. Ejecuta los baselines (Buy & Hold, Equal-Weight) en el conjunto de test
      4. Calcula métricas por régimen para cada estrategia
      5. Genera visualización y guarda en reports/

    Parámetros
    ----------
    modelo_path : ruta al modelo PPO entrenado (.zip). Si es None, solo se analiza
                  la distribución de regímenes sin ejecutar el agente.

    Retorna
    -------
    dict con resultados por estrategia y régimen
    """
    # Carga de datos
    df_features = pd.read_csv(features_path, index_col=0)
    df_precios  = pd.read_csv(prices_path,   index_col=0)
    df_features.index = pd.to_datetime(df_features.index)
    df_precios.index  = pd.to_datetime(df_precios.index)

    split_idx = int(len(df_features) * split_pct)

    # Clasificación de regímenes sobre todo el dataset (umbrales del train)
    regimenes_full = clasificar_regimenes(df_precios, umbral_train_pct=split_pct)
    regimenes_test = regimenes_full.iloc[split_idx:]
    precios_test   = df_precios.iloc[split_idx:]

    print(f"\nDistribución de regímenes en el período de test ({len(precios_test)} días):")
    distribucion = regimenes_test.value_counts().sort_index()
    for codigo, n in distribucion.items():
        nombre = {0: 'Baja Vol.', 1: 'Vol. Media', 2: 'Alta Vol.'}[codigo]
        print(f"  Régimen {codigo} ({nombre}): {n} días ({n/len(regimenes_test)*100:.1f}%)")

    resultados = {'regimenes': regimenes_test}

    # Baselines simples (no requieren modelo)
    try:
        from src.benchmarking.baselines import simular_buy_and_hold, simular_equal_weight
    except ImportError:
        from src.benchmarking.baselines import simular_buy_and_hold, simular_equal_weight

    serie_bh = simular_buy_and_hold(precios_test, initial_balance)
    serie_ew = simular_equal_weight(precios_test, initial_balance)

    resultados['Buy_and_Hold'] = serie_bh
    resultados['Equal_Weight'] = serie_ew

    print("\nMétricas por régimen — Buy & Hold:")
    df_bh = metricas_por_regimen(serie_bh, regimenes_test)
    if not df_bh.empty:
        print(df_bh.to_string())
        resultados['metricas_bh'] = df_bh

    print("\nMétricas por régimen — Equal-Weight:")
    df_ew = metricas_por_regimen(serie_ew, regimenes_test)
    if not df_ew.empty:
        print(df_ew.to_string())
        resultados['metricas_ew'] = df_ew

    # Agente DRL (solo si se proporciona el modelo)
    if modelo_path and os.path.exists(modelo_path):
        try:
            from stable_baselines3 import PPO
            modelo    = PPO.load(modelo_path)
            vals_ia   = _ejecutar_agente(modelo, features_path, prices_path, split_idx, None)
            serie_ia  = pd.Series(
                vals_ia[:len(precios_test)],
                index=precios_test.index,
                name='IA_PPO'
            )
            resultados['IA_PPO'] = serie_ia

            print("\nMétricas por régimen — Agente IA (PPO):")
            df_ia = metricas_por_regimen(serie_ia, regimenes_test)
            if not df_ia.empty:
                print(df_ia.to_string())
                resultados['metricas_ia'] = df_ia
        except Exception as e:
            print(f"  [AVISO] No se pudo ejecutar el agente: {e}")

    # Visualización
    _plot_regimenes(resultados, regimenes_test, precios_test)
    _guardar_metricas(resultados)

    return resultados


# ─────────────────────────────────────────────
# Visualización
# ─────────────────────────────────────────────

# Paleta de colores para los regímenes
_COLORES_REGIMEN = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
_NOMBRE_REGIMEN  = {0: 'Baja Volatilidad', 1: 'Volatilidad Media', 2: 'Alta Volatilidad'}


def _sombrear_fondos(ax, regimenes: pd.Series):
    """
    Sombrea el fondo de un eje matplotlib según el régimen de volatilidad activo.
    Verde=Baja, Naranja=Media, Rojo=Alta volatilidad.
    """
    if regimenes.empty:
        return

    # Detectar bloques continuos del mismo régimen
    cambios = regimenes.ne(regimenes.shift()).cumsum()
    for _, bloque in regimenes.groupby(cambios):
        if bloque.empty:
            continue
        codigo = int(bloque.iloc[0])
        inicio = bloque.index[0]
        fin    = bloque.index[-1]
        ax.axvspan(inicio, fin,
                   alpha=0.15,
                   color=_COLORES_REGIMEN.get(codigo, 'gray'),
                   linewidth=0)


def _plot_regimenes(resultados: dict, regimenes_test: pd.Series, precios_test: pd.DataFrame):
    """
    Genera la gráfica principal de análisis de regímenes:
      - Panel superior: evolución de carteras con zonas sombreadas por régimen
      - Panel inferior: indicador de régimen a lo largo del tiempo
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax1, ax2 = axes

    # ── Panel superior: evolución de carteras ──────────────────────────────────
    colores_estrategias = {
        'IA_PPO':          ('#1f77b4', '-',  2.5),
        'Buy_and_Hold':    ('#ff7f0e', '--', 1.5),
        'Equal_Weight':    ('#2ca02c', '--', 1.5),
        'Cartera_60_40':   ('#9467bd', '--', 1.5),
        'Markowitz_MV':    ('#8c564b', ':',  1.5),
    }

    for nombre, (color, estilo, grosor) in colores_estrategias.items():
        if nombre in resultados and resultados[nombre] is not None:
            serie = resultados[nombre]
            ax1.plot(serie.index, serie.values,
                     label=nombre, color=color, linestyle=estilo, linewidth=grosor)

    # Fondo sombreado por régimen
    _sombrear_fondos(ax1, regimenes_test)

    # Leyenda de regímenes
    parches = [
        mpatches.Patch(color=_COLORES_REGIMEN[k], alpha=0.5, label=_NOMBRE_REGIMEN[k])
        for k in sorted(_COLORES_REGIMEN.keys())
    ]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + parches,
               labels=ax1.get_legend_handles_labels()[1]  + [p.get_label() for p in parches],
               loc='upper left', fontsize=9)
    ax1.set_title('Análisis de Regímenes de Volatilidad — Conjunto Out-of-Sample',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valor de Cartera ($)')
    ax1.grid(True, alpha=0.3)

    # ── Panel inferior: indicador de régimen ──────────────────────────────────
    colores_reg = [_COLORES_REGIMEN.get(int(v), 'gray') for v in regimenes_test.values]
    ax2.bar(regimenes_test.index, regimenes_test.values,
            color=colores_reg, width=1, alpha=0.8)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['Baja', 'Media', 'Alta'])
    ax2.set_title('Indicador de Régimen (0=Baja, 1=Media, 2=Alta Volatilidad)')
    ax2.set_ylabel('Régimen')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    ruta = 'src/reports/regime_analysis.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"\nGráfica de regímenes guardada: {ruta}")
    plt.close(fig)


def _guardar_metricas(resultados: dict):
    """
    Guarda las tablas de métricas por régimen en CSV para la memoria del TFM.
    """
    ruta = 'src/reports/regime_metrics.csv'
    tablas = []

    for clave in ['metricas_ia', 'metricas_bh', 'metricas_ew']:
        if clave in resultados and not resultados[clave].empty:
            nombre = clave.replace('metricas_', '').upper()
            df     = resultados[clave].copy()
            df.insert(0, 'Estrategia', nombre)
            tablas.append(df)

    if tablas:
        pd.concat(tablas).to_csv(ruta, encoding='utf-8-sig')
        print(f"Métricas por régimen guardadas: {ruta}")


# ─────────────────────────────────────────────
# Ejecución directa
# ─────────────────────────────────────────────

if __name__ == "__main__":
    resultados = analizar_regimenes(
        features_path='data/normalized_features.csv',
        prices_path='data/original_prices.csv',
        modelo_path='models/best_model/best_model.zip',
    )
