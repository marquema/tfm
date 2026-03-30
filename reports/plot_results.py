import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Ensure project root is on sys.path so `src.*` imports work regardless of
# whether this script is run as `python reports/plot_results.py` (from root)
# or opened directly inside the reports/ directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.trading_env import PortfolioEnv

# Save all generated images next to this script (inside reports/)
_REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def calcular_sharpe(retornos, rf=0.0):
    """Sharpe Ratio anualizado (252 días de trading)."""
    if len(retornos) < 2:
        return 0.0
    std = retornos.std()
    if std == 0:
        return 0.0
    return (retornos.mean() - rf) / std * np.sqrt(252)


def calcular_max_drawdown(valores):
    """Max Drawdown como fracción (ej: -0.15 = caída máxima del 15%)."""
    serie = pd.Series(valores)
    return (serie / serie.cummax() - 1).min()


def _simular(model, env):
    """Ejecuta un episodio completo con el modelo y devuelve (valores, drawdowns, weights)."""
    obs, _ = env.reset()
    done = False
    valores, drawdowns, weights_hist = [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        valores.append(info['value'])
        drawdowns.append(info.get('drawdown', 0.0))
        weights_hist.append(info.get('weights', np.zeros(env.n_assets)).copy())

    return valores, drawdowns, weights_hist


def _benchmark_bh(env):
    """Buy & Hold 1/N sobre los precios del entorno."""
    retornos = env.df_precios.pct_change().dropna()
    retorno_bh = retornos.mean(axis=1)
    valores_bh = 10000 * (1 + retorno_bh).cumprod().values
    return np.insert(valores_bh, 0, 10000)


def _crear_entorno_test(split_pct=0.8, reward_mode='rolling_sharpe'):
    df_f      = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)
    return PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        start_idx=split_idx, reward_mode=reward_mode,
        episode_length=None, random_start=False,  # full test window, no randomness
    ), split_idx


# ---------------------------------------------------------------------------
# Backtest principal
# ---------------------------------------------------------------------------

def ejecutar_backtest_pro(split_pct=0.8):
    """
    Backtest del modelo principal (sharpe_drawdown) vs Buy & Hold.
    Muestra tabla de métricas y gráfica con área de drawdown.
    """
    print("--- Backtest principal (sharpe_drawdown) ---")
    model = PPO.load("models/best_model/best_model.zip")
    env, _ = _crear_entorno_test(split_pct, reward_mode='sharpe_drawdown')

    valores_ia, drawdowns_ia, _ = _simular(model, env)
    valores_bh = _benchmark_bh(env)[:len(valores_ia)]

    df_res = pd.DataFrame({'IA': valores_ia, 'BH': valores_bh})
    rets   = df_res.pct_change().dropna()

    sharpe_ia = calcular_sharpe(rets['IA'])
    sharpe_bh = calcular_sharpe(rets['BH'])
    dd_ia     = calcular_max_drawdown(valores_ia)
    dd_bh     = calcular_max_drawdown(valores_bh)

    print("\n" + "=" * 38)
    print(f"{'MÉTRICA':<16} | {'IA PPO':>9} | {'B&H 1/N':>9}")
    print("-" * 38)
    print(f"{'Valor final':<16} | ${df_res['IA'].iloc[-1]:>8.2f} | ${df_res['BH'].iloc[-1]:>8.2f}")
    print(f"{'Sharpe Ratio':<16} | {sharpe_ia:>9.2f} | {sharpe_bh:>9.2f}")
    print(f"{'Max Drawdown':<16} | {dd_ia*100:>8.2f}% | {dd_bh*100:>8.2f}%")
    print("=" * 38)

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df_res['IA'], label=f'IA PPO — Sharpe {sharpe_ia:.2f}', color='#1f77b4', lw=2)
    ax1.plot(df_res['BH'], label=f'B&H 1/N — Sharpe {sharpe_bh:.2f}', color='#ff7f0e', ls='--')
    ax1.fill_between(range(len(df_res)), df_res['IA'], df_res['IA'].cummax(),
                     color='#1f77b4', alpha=0.1, label='Drawdown IA')
    ax1.set_title('Validación Ex-Post: IA Portfolio Manager vs Benchmark')
    ax1.set_ylabel('Valor de la Cartera ($)')
    ax1.legend()

    ax2.fill_between(range(len(drawdowns_ia)), [-d * 100 for d in drawdowns_ia],
                     color='red', alpha=0.4)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Días de Negociación')

    plt.tight_layout()
    out = os.path.join(_REPORTS_DIR, 'backtest_principal.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Chart saved: {out}")


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def ejecutar_backtest_comparativo(split_pct=0.8):
    """
    Compara el modelo principal (sharpe_drawdown) contra el baseline (log_return).
    Responde: ¿cuánto aporta la penalización por drawdown?
    Requiere haber entrenado previamente con entrenar_ablacion().
    """
    print("--- Ablation study: sharpe_drawdown vs log_return ---")

    model_path_principal = "models/best_model/best_model.zip"
    model_path_ablacion  = "models/ablation_log_return/best_model.zip"

    if not os.path.exists(model_path_ablacion):
        print(f"No se encontró {model_path_ablacion}. Ejecuta entrenar_ablacion() primero.")
        return

    model_sd  = PPO.load(model_path_principal)
    model_lr  = PPO.load(model_path_ablacion)

    env_sd, _ = _crear_entorno_test(split_pct, reward_mode='sharpe_drawdown')
    env_lr, _ = _crear_entorno_test(split_pct, reward_mode='log_return')

    valores_sd, dd_sd, _ = _simular(model_sd, env_sd)
    valores_lr, dd_lr, _ = _simular(model_lr, env_lr)
    valores_bh            = _benchmark_bh(env_sd)[:len(valores_sd)]

    n = min(len(valores_sd), len(valores_lr), len(valores_bh))
    df = pd.DataFrame({
        'IA sharpe_drawdown': valores_sd[:n],
        'IA log_return':      valores_lr[:n],
        'B&H 1/N':            valores_bh[:n],
    })
    rets = df.pct_change().dropna()

    print("\n" + "=" * 52)
    print(f"{'MÉTRICA':<16} | {'SD':>9} | {'LR':>9} | {'B&H':>9}")
    print("-" * 52)
    for col in df.columns:
        sharpe = calcular_sharpe(rets[col])
        dd     = calcular_max_drawdown(df[col].values)
        label  = col.replace('IA ', '').replace('B&H 1/N', 'B&H')
        print(f"{'Sharpe ' + label:<16} | {sharpe:>9.2f}")
        print(f"{'MaxDD  ' + label:<16} | {dd*100:>8.2f}%")
    print("=" * 52)

    plt.style.use('ggplot')
    plt.figure(figsize=(13, 6))
    plt.plot(df['IA sharpe_drawdown'], label='IA sharpe_drawdown', color='#1f77b4', lw=2)
    plt.plot(df['IA log_return'],      label='IA log_return (baseline)', color='#2ca02c', lw=2, ls='-.')
    plt.plot(df['B&H 1/N'],            label='B&H 1/N', color='#ff7f0e', ls='--')
    plt.title('Ablation Study: impacto del risk-shaping (φ·drawdown)')
    plt.ylabel('Valor de la Cartera ($)')
    plt.xlabel('Días de Negociación')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(_REPORTS_DIR, 'ablation_study.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Chart saved: {out}")


# ---------------------------------------------------------------------------
# Progreso de entrenamiento (TensorBoard)
# ---------------------------------------------------------------------------

def plot_training_progress(log_dir='./logs/'):
    """Extrae ep_rew_mean de los logs de TensorBoard y muestra la curva de aprendizaje."""
    files = glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True)
    if not files:
        print(f"No se encontraron logs en {log_dir}")
        return

    latest_file = max(files, key=os.path.getctime)
    ea = EventAccumulator(latest_file)
    ea.Reload()

    tag = 'rollout/ep_rew_mean'
    if tag not in ea.Tags().get('scalars', []):
        print(f"Tag '{tag}' no encontrado. Espera a que el entrenamiento avance.")
        return

    events    = ea.Scalars(tag)
    steps     = [e.step  for e in events]
    vals      = [e.value for e in events]

    plt.style.use('ggplot')
    plt.figure(figsize=(11, 5))
    plt.plot(steps, vals, label='Recompensa media (train)', color='#2ca02c', lw=1.5, alpha=0.6)
    if len(vals) > 10:
        suavizado = pd.Series(vals).rolling(window=10).mean()
        plt.plot(steps, suavizado, label='Tendencia (media móvil 10)', ls='--', color='red', lw=2)
    plt.title('Curva de Aprendizaje PPO')
    plt.xlabel('Pasos')
    plt.ylabel('Log-Return Medio por Episodio')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(_REPORTS_DIR, 'training_progress.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Chart saved: {out}")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ejecutar_backtest_pro()
    plot_training_progress()
    # ejecutar_backtest_comparativo()  # descomentar tras ejecutar entrenar_ablacion()
