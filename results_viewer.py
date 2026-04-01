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
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin pantalla (compatible con servidor)
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.environment_trading import PortfolioEnv
from src.benchmarking.baselines import (
    ejecutar_baselines, calcular_metricas, tabla_comparativa
)
from src.regime_analysis import analizar_regimenes

os.makedirs("src/reports", exist_ok=True)

# ─────────────────────────────────────────────
# Rutas por defecto
# ─────────────────────────────────────────────
FEATURES_PATH  = 'data/normalized_features.csv'
PRICES_PATH    = 'data/original_prices.csv'
MODELO_PATH    = 'models/best_model_academic/best_model.zip'
SPLIT_PCT      = 0.8
INITIAL_BALANCE = 10_000


# ─────────────────────────────────────────────
# Progreso del entrenamiento (TensorBoard logs)
# ─────────────────────────────────────────────

def plot_training_progress(log_dir: str = './logs/'):
    """
    Lee los logs de TensorBoard y genera la curva de recompensa media del entrenamiento.

    Guarda la gráfica en reports/training_progress.png.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("[AVISO] TensorBoard no disponible. Omitiendo gráfica de entrenamiento.")
        return

    archivos = glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True)
    if not archivos:
        print(f"[AVISO] No se encontraron logs en '{log_dir}'.")
        return

    # Cargar el archivo de log más reciente
    ultimo = max(archivos, key=os.path.getctime)
    ea     = EventAccumulator(ultimo)
    ea.Reload()

    etiqueta = 'rollout/ep_rew_mean'
    if etiqueta not in ea.Tags().get('scalars', []):
        print(f"[AVISO] Etiqueta '{etiqueta}' no encontrada en los logs.")
        return

    eventos = ea.Scalars(etiqueta)
    pasos   = [e.step  for e in eventos]
    valores = [e.value for e in eventos]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pasos, valores, label='Recompensa media (episodio)',
            color='#2ca02c', linewidth=1.5, alpha=0.7)

    # Media móvil para suavizar la curva
    if len(valores) > 10:
        suavizado = pd.Series(valores).rolling(window=10).mean()
        ax.plot(pasos, suavizado, label='Tendencia (media móvil 10)',
                color='#d62728', linewidth=2, linestyle='--')

    ax.set_title('Progreso del Entrenamiento del Agente PPO', fontsize=13)
    ax.set_xlabel('Pasos de entrenamiento')
    ax.set_ylabel('Recompensa media por episodio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta = 'src/reports/training_progress.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"Gráfica de entrenamiento guardada: {ruta}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Backtest principal
# ─────────────────────────────────────────────

def ejecutar_backtest_completo(model_path:    str   = MODELO_PATH,
                                features_path: str   = FEATURES_PATH,
                                prices_path:   str   = PRICES_PATH,
                                split_pct:     float = SPLIT_PCT,
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

    # ── 1. Cargar datos y calcular el índice de corte ─────────────────────────
    df_f = pd.read_csv(features_path, index_col=0).dropna()
    df_p = pd.read_csv(prices_path,   index_col=0)
    df_p.index = pd.to_datetime(df_p.index)

    split_idx  = int(len(df_f) * split_pct)
    precios_test = df_p.iloc[split_idx:].copy()

    print(f"  Período de test: {precios_test.index[0].date()} → {precios_test.index[-1].date()}")
    print(f"  Días de trading: {len(precios_test)}")

    # ── 2. Ejecutar el agente PPO ─────────────────────────────────────────────
    print("\n  Ejecutando agente IA (PPO)...")
    modelo = PPO.load(model_path)
    env    = PortfolioEnv(features_path, prices_path, start_idx=split_idx)
    obs, _ = env.reset()
    done   = False
    vals_ia = [initial_balance]

    while not done:
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(accion)
        vals_ia.append(info['value'])

    # Alinear longitudes: el agente produce un valor por paso
    n_test   = len(precios_test)
    serie_ia = pd.Series(
        vals_ia[:n_test],
        index=precios_test.index[:len(vals_ia[:n_test])],
        name='IA_PPO'
    )

    # ── 3. Simular los cuatro baselines ───────────────────────────────────────
    print("\n  Ejecutando baselines financieros...")
    baselines = ejecutar_baselines(precios_test, initial_balance, commission=0.001)

    # ── 4. Unir todos los resultados ─────────────────────────────────────────
    todos_los_valores = {'IA_PPO': serie_ia}
    todos_los_valores.update(baselines)

    # ── 5. Tabla comparativa de métricas ──────────────────────────────────────
    print("\n" + "="*65)
    print(f"{'BENCHMARKING COMPARATIVO — PERÍODO OUT-OF-SAMPLE':^65}")
    print("="*65)

    df_metricas = tabla_comparativa(todos_los_valores)

    # Mostrar tabla con formato
    print(df_metricas.to_string())
    print("="*65)

    # Guardar tabla en CSV para la memoria del TFM
    ruta_csv = 'src/reports/metrics_table.csv'
    df_metricas.to_csv(ruta_csv, encoding='utf-8-sig')
    print(f"\nTabla de métricas guardada: {ruta_csv}")

    # ── 6. Gráfica comparativa ────────────────────────────────────────────────
    _plot_backtest(todos_los_valores, df_metricas)

    return {'valores': todos_los_valores, 'metricas': df_metricas}


def _plot_backtest(valores: dict, metricas: pd.DataFrame):
    """
    Genera la gráfica principal de comparación de carteras.
    Guarda en reports/backtest_principal.png.
    """
    # Configuración visual por estrategia
    estilos = {
        'IA_PPO':              {'color': '#1f77b4', 'ls': '-',  'lw': 2.5, 'zorder': 5},
        'Equal_Weight_Mensual':{'color': '#2ca02c', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Cartera_60_40':       {'color': '#9467bd', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Buy_and_Hold':        {'color': '#ff7f0e', 'ls': '--', 'lw': 1.5, 'zorder': 3},
        'Markowitz_MV':        {'color': '#8c564b', 'ls': ':',  'lw': 1.5, 'zorder': 3},
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                              gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2  = axes

    # ── Panel superior: evolución de valor de cartera ─────────────────────────
    for nombre, serie in valores.items():
        if serie is None or len(serie) < 2:
            continue
        est    = estilos.get(nombre, {'color': 'gray', 'ls': '-', 'lw': 1, 'zorder': 1})
        sharpe = metricas.loc[nombre, 'Sharpe Ratio'] if nombre in metricas.index else '—'
        mdd    = metricas.loc[nombre, 'Max Drawdown (%)'] if nombre in metricas.index else '—'
        ax1.plot(serie.index, serie.values,
                 label=f'{nombre}  |  Sharpe: {sharpe}  |  MDD: {mdd}%',
                 color=est['color'], linestyle=est['ls'],
                 linewidth=est['lw'], zorder=est['zorder'])

    ax1.set_title('Validación Out-of-Sample: IA PPO vs Baselines Financieros',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valor de Cartera ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Área de drawdown de la IA
    if 'IA_PPO' in valores and valores['IA_PPO'] is not None:
        s_ia = valores['IA_PPO']
        ax1.fill_between(s_ia.index, s_ia.values, s_ia.cummax().values,
                         color='#1f77b4', alpha=0.08, label='_nolegend_')

    # ── Panel inferior: retornos diarios del agente ──────────────────────────
    if 'IA_PPO' in valores and valores['IA_PPO'] is not None:
        rets = valores['IA_PPO'].pct_change().fillna(0)
        colores = ['#2ca02c' if r >= 0 else '#d62728' for r in rets]
        ax2.bar(rets.index, rets.values * 100, color=colores, width=1, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('Retorno diario IA (%)')
        ax2.set_title('Retornos Diarios del Agente IA (PPO)')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    ruta = 'src/reports/backtest_principal.png'
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    print(f"Gráfica de backtest guardada: {ruta}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Curva de entrenamiento (si existen logs)
    plot_training_progress('./logs/')

    # 2. Backtest completo con los 4 baselines
    resultados = ejecutar_backtest_completo()

    # 3. Análisis de regímenes de volatilidad
    if resultados:
        print("\n=== ANÁLISIS DE REGÍMENES DE VOLATILIDAD ===")
        analizar_regimenes(
            features_path=FEATURES_PATH,
            prices_path=PRICES_PATH,
            modelo_path=MODELO_PATH,
        )
