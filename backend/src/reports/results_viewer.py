"""
Módulo de visualización y evaluación de resultados del TFM.

Ejecuta el backtest completo comparando el agente IA (PPO) contra
los cuatro baselines financieros definidos en los objetivos del trabajo:
  1. Equal-Weight con rebalanceo mensual
  2. Cartera 60/40 (IVV/BND)
  3. Buy & Hold
  4. Markowitz Media-Varianza (ventana 12 meses)
  

Genera:
  - Tabla comparativa de métricas (Sharpe, Sortino, MDD, CAGR, Retorno Total)
  - Gráfica de evolución de carteras guardada en reports/
  - Análisis de regímenes de volatilidad

Uso:
  python results_viewer.py
"""

# todo: comparativa contra el agente especulativo.

import os
import sys
# Añadir raíz del proyecto al path para que 'src' sea importable
# cuando se ejecuta directamente: python src/reports/results_viewer.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla (compatible con servidor)
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.training_drl.environment_trading import PortfolioEnv
from src.benchmarking.baselines import (
   ejecutar_baselines, calcular_metricas, tabla_comparativa
)
from src.training_drl.regime_analysis import analizar_regimenes

os.makedirs("src/reports", exist_ok=True)

# todo: adaptar a nombres en inglés

# ─────────────────────────────────────────────
# Rutas por defecto
# ─────────────────────────────────────────────
FEATURES_PATH   = 'data/normalized_features.csv'
PRICES_PATH     = 'data/original_prices.csv'
MODEL_PATH      = 'models/best_model_academic/best_model.zip'
SPLIT_PCT       = 0.8
INITIAL_BALANCE = 10_000


# ─────────────────────────────────────────────
# Progreso del entrenamiento (TensorBoard logs)
# ─────────────────────────────────────────────

def plot_training_progress(log_dir: str = './logs/'):
    """
    Lee los logs de TensorBoard y genera la curva de recompensa media del entrenamiento.

    Busca el archivo de eventos más reciente dentro de log_dir, extrae la serie
    temporal de 'rollout/ep_rew_mean' y genera una gráfica con la curva original
    y una media móvil suavizada.

    Guarda la gráfica en reports/training_progress.png.

    Parámetros
    ----------
    log_dir : str
        Directorio donde se encuentran los logs de TensorBoard.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[AVISO] TensorBoard no disponible. Omitiendo gráfica de entrenamiento.")
        return

    log_files = glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True)
    if not log_files:
        print(f"[AVISO] No se encontraron logs en '{log_dir}'.")
        return

    # Cargar el archivo de log más reciente
    latest_file = max(log_files, key=os.path.getctime)
    ea = EventAccumulator(latest_file)
    ea.Reload()

    tag = 'rollout/ep_rew_mean'
    if tag not in ea.Tags().get('scalars', []):
        print(f"[AVISO] Etiqueta '{tag}' no encontrada en los logs.")
        return

    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, label='Recompensa media (episodio)',
            color='#2ca02c', linewidth=1.5, alpha=0.7)

    # Media móvil para suavizar la curva
    if len(values) > 10:
        smoothed = pd.Series(values).rolling(window=10).mean()
        ax.plot(steps, smoothed, label='Tendencia (media móvil 10)',
                color='#d62728', linewidth=2, linestyle='--')

    ax.set_title('Progreso del Entrenamiento del Agente PPO', fontsize=13)
    ax.set_xlabel('Pasos de entrenamiento')
    ax.set_ylabel('Recompensa media por episodio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = 'src/reports/training_progress.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica de entrenamiento guardada: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Backtest principal
# ─────────────────────────────────────────────

def run_full_backtest(model_path: str= MODEL_PATH,
                      features_path: str = FEATURES_PATH,
                      prices_path:  str = PRICES_PATH,
                      split_pct: float= SPLIT_PCT,
                      initial_balance: float = INITIAL_BALANCE) -> dict:
    """
    Ejecuta el backtest completo del agente IA y los cuatro baselines.

    Proceso:
      1. Carga el modelo PPO entrenado
      2. Ejecuta el episodio de test (20% final del dataset, datos no vistos)
      3. Simula los cuatro baselines sobre el mismo período
      4. Calcula métricas comparativas (Sharpe, Sortino, MDD, CAGR, etc.)
      5. Imprime la tabla comparativa
      6. Genera y guarda la gráfica en reports/

    Parámetros
    ----------
    model_path : str
        Ruta al archivo .zip del modelo PPO entrenado.
    features_path : str
        Ruta al CSV de features normalizadas.
    prices_path : str
        Ruta al CSV de precios originales.
    split_pct : float
        Porcentaje del dataset destinado a entrenamiento (el resto es test).
    initial_balance : float
        Capital inicial para la simulación.

    Retorna
    -------
    dict con:
      'valores': dict {nombre_estrategia: pd.Series}
      'metricas': pd.DataFrame con la tabla comparativa
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        print("  Ejecuta primero el entrenamiento desde /fase3/entrenar o train.py")
        return {}

    print("\n=== BACKTEST OUT-OF-SAMPLE ===")
    print(f"Modelo: {model_path}")

    # ── 1. Cargar datos y calcular el índice de corte
    df_f = pd.read_csv(features_path, index_col=0).dropna()
    df_p = pd.read_csv(prices_path,   index_col=0)
    df_p.index = pd.to_datetime(df_p.index)

    split_idx   = int(len(df_f) * split_pct)
    test_prices = df_p.iloc[split_idx:].copy()

    print(f"  Período de test: {test_prices.index[0].date()} -> {test_prices.index[-1].date()}")
    print(f"  Días de trading: {len(test_prices)}")

    # ── 2. Ejecutar el agente PPO 
    print("\n  Ejecutando agente IA (PPO)...")
    loaded_model = PPO.load(model_path)
    env = PortfolioEnv(features_path, prices_path, start_idx=split_idx)
    obs, _ = env.reset()
    done = False
    ia_values = [initial_balance]

    while not done:
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        ia_values.append(info['value'])

    # Alinear longitudes: el agente produce un valor por paso
    n_test    = len(test_prices)
    ia_series = pd.Series(
        ia_values[:n_test],
        index=test_prices.index[:len(ia_values[:n_test])],
        name='IA_PPO'
    )

    # ── 3. Simular los cuatro baselines ───────────────────────────────────────
    print("\n  Ejecutando baselines financieros...")
    baselines = ejecutar_baselines(test_prices, initial_balance, commission=0.001)

    # ── 3b. Agente especulativo (GMM + K-Means) ─────────────────────────────
    speculative_path = 'models/speculative_gmm.pkl'
    if os.path.exists(speculative_path):
        print("\n  Ejecutando agente especulativo (GMM + K-Means)...")
        import pickle
        with open(speculative_path, 'rb') as f:
            spec_agent = pickle.load(f)
        df_f_full = pd.read_csv(features_path, index_col=0)
        df_f_test = df_f_full.iloc[split_idx:]
        spec_series = spec_agent.backtest(
            df_f_test, test_prices,
            initial_balance=initial_balance, commission=0.001
        )
        baselines['Especulativo_HMM'] = spec_series
    else:
        print("  [AVISO] Agente especulativo no ajustado. "
              "Ejecuta POST /fase4/ajustar-especulativo")

    # ── 4. Unir todos los resultados ─────────────────────────────────────────
    all_values = {'IA_PPO': ia_series}
    all_values.update(baselines)

    # ── 5. Tabla comparativa de métricas ──────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'BENCHMARKING COMPARATIVO — PERÍODO OUT-OF-SAMPLE':^65}")
    print("="*65)

    df_metrics = tabla_comparativa(all_values)

    # Mostrar tabla con formato
    print(df_metrics.to_string())
    print("="*65)

    # Guardar tabla en CSV para la memoria del TFM
    csv_path = 'src/reports/metrics_table.csv'
    df_metrics.to_csv(csv_path, encoding='utf-8-sig')
    print(f"\nTabla de métricas guardada: {csv_path}")

    # ── 6. Gráfica comparativa ────────────────────────────────────────────────
    _plot_backtest(all_values, df_metrics)

    return {'valores': all_values, 'metricas': df_metrics}


# Alias de compatibilidad hacia atrás
ejecutar_backtest_completo = run_full_backtest


def _plot_backtest(portfolio_values: dict, metrics: pd.DataFrame):
    """
    Genera la gráfica principal de comparación de carteras.

    Crea una figura con dos paneles:
      - Panel superior: evolución del valor de cada cartera a lo largo del test.
      - Panel inferior: retornos diarios del agente IA (PPO) como barras.

    Guarda el resultado en reports/backtest_principal.png.

    Parámetros
    ----------
    portfolio_values : dict
        Diccionario {nombre_estrategia: pd.Series} con la evolución de cada cartera.
    metrics : pd.DataFrame
        Tabla de métricas comparativas (Sharpe, MDD, etc.) indexada por nombre.
    """
    # Configuración visual por estrategia
    styles = {
        'IA_PPO':              {'color': '#1f77b4', 'ls': '-',  'lw': 2.5, 'zorder': 5},
        'Equal_Weight_Mensual':{'color': '#2ca02c', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Cartera_60_40':       {'color': '#9467bd', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Buy_and_Hold':        {'color': '#ff7f0e', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Markowitz_MV':        {'color': '#8c564b', 'ls': ':',  'lw': 1.5, 'zorder': 3},
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2  = axes

    # ── Panel superior: evolución de valor de cartera 
    for name, series in portfolio_values.items():
        if series is None or len(series) < 2:
            continue
        style  = styles.get(name, {'color': 'gray', 'ls': '-', 'lw': 1, 'zorder': 1})
        sharpe = metrics.loc[name, 'Sharpe Ratio'] if name in metrics.index else '—'
        mdd    = metrics.loc[name, 'Max Drawdown (%)'] if name in metrics.index else '—'
        ax1.plot(series.index, series.values,
                 label=f'{name}  |  Sharpe: {sharpe}  |  MDD: {mdd}%',
                 color=style['color'], linestyle=style['ls'],
                 linewidth=style['lw'], zorder=style['zorder'])

    ax1.set_title('Validación Out-of-Sample: IA PPO vs Baselines Financieros',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valor de Cartera ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Área de drawdown de la IA
    if 'IA_PPO' in portfolio_values and portfolio_values['IA_PPO'] is not None:
        ia_series = portfolio_values['IA_PPO']
        ax1.fill_between(ia_series.index, ia_series.values, ia_series.cummax().values,
                         color='#1f77b4', alpha=0.08, label='_nolegend_')

    # ── Panel inferior: retornos diarios del agente 
    if 'IA_PPO' in portfolio_values and portfolio_values['IA_PPO'] is not None:
        returns = portfolio_values['IA_PPO'].pct_change().fillna(0)
        bar_colors = ['#2ca02c' if r >= 0 else '#d62728' for r in returns]
        ax2.bar(returns.index, returns.values * 100, color=bar_colors, width=1, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('Retorno diario IA (%)')
        ax2.set_title('Retornos Diarios del Agente IA (PPO)')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = 'src/reports/backtest_principal.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica de backtest guardada: {output_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Curva de entrenamiento (si existen logs)
    plot_training_progress('./logs/')

    # 2. Backtest completo con los 4 baselines
    results = run_full_backtest()

    # 3. Análisis de regímenes de volatilidad
    if results:
        print("\n=== ANÁLISIS DE REGÍMENES DE VOLATILIDAD ===")
        analizar_regimenes(
            features_path=FEATURES_PATH,
            prices_path=PRICES_PATH,
            model_path=MODEL_PATH,
        )
