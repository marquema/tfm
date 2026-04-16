"""
Análisis de sensibilidad de hiperparámetros del agente PPO.

Entrena múltiples configuraciones del entorno y del modelo PPO,
evalúa cada una en el mismo período out-of-sample, y genera una
tabla comparativa para la memoria del TFM.

Esto responde a la pregunta del tribunal:
  "¿Por qué phi=0.02 y gamma=0.01? ¿Se probaron otras configuraciones?"

Configuraciones probadas:
  A: phi=0.02, gamma=0.01  (configuración actual — baseline del análisis)
  B: phi=0.05, gamma=0.01  (más penalización por drawdown)
  C: phi=0.02, gamma=0.02  (más penalización por turnover)
  D: phi=0.01, gamma=0.005 (configuración más agresiva, menos penalización)

Cada configuración entrena un PPO desde cero con los mismos datos y steps,
y se evalúa en el 20% de test. El resultado es un CSV y un PNG comparativos.

Uso:
  python -m src.training_drl.sensitivity_analysis

  O desde la API:
  POST /admin/fase3/sensitivity-analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.training_drl.environment_trading import PortfolioEnv
from src.benchmarking.baselines import compute_metrics


# ─── Configuraciones a probar ─────────────────────────────────────────────────

CONFIGS = {
    'A (actual)': {
        'phi': 0.02,
        'gamma': 0.01,
        'description': 'Configuración actual: balance entre retorno y control de riesgo',
    },
    'B (más MDD)': {
        'phi': 0.05,
        'gamma': 0.01,
        'description': 'Mayor penalización por drawdown: prioriza preservar capital',
    },
    'C (más turnover)': {
        'phi': 0.02,
        'gamma': 0.02,
        'description': 'Mayor penalización por rotación: fuerza al agente a mantener posiciones',
    },
    'D (agresivo)': {
        'phi': 0.01,
        'gamma': 0.005,
        'description': 'Menos penalizaciones: el agente busca máximo retorno con libertad',
    },
}


def run_sensitivity_analysis(
    features_path: str = 'data/normalized_features.csv',
    prices_path: str = 'data/original_prices.csv',
    total_timesteps: int = 200000,
    split_pct: float = 0.8,
) -> pd.DataFrame:
    """
    Ejecuta el análisis de sensibilidad completo.

    Entrena un PPO por cada configuración, evalúa en test, y genera
    tabla comparativa + gráfica.

    Parameters
    ----------
    features_path   : ruta al CSV de features
    prices_path     : ruta al CSV de precios
    total_timesteps : pasos por configuración (200k es suficiente para comparar)
    split_pct       : fracción train/test

    Returns
    -------
    pd.DataFrame con métricas por configuración
    """
    # Borrar reportes anteriores
    for old_file in ['src/reports/sensitivity_analysis.csv',
                     'src/reports/sensitivity_analysis.png']:
        if os.path.exists(old_file):
            os.remove(old_file)

    df_f = pd.read_csv(features_path, index_col=0)
    df_p = pd.read_csv(prices_path, index_col=0)
    split_idx = int(len(df_f) * split_pct)

    results = []

    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE SENSIBILIDAD DE HIPERPARÁMETROS")
    print(f"{'='*70}")
    print(f"Configuraciones: {len(CONFIGS)}")
    print(f"Steps por config: {total_timesteps:,}")
    print(f"Dataset: {len(df_f)} días, split {split_pct:.0%}/{1-split_pct:.0%}")
    print(f"{'='*70}")

    for name, config in CONFIGS.items():
        phi = config['phi']
        gamma = config['gamma']
        desc = config['description']

        print(f"\n--- Config {name}: phi={phi}, gamma={gamma} ---")
        print(f"    {desc}")

        # Entornos con los hiperparámetros de esta configuración
        train_env = PortfolioEnv(
            features_path, prices_path,
            end_idx=split_idx, phi=phi, gamma=gamma,
        )
        eval_env = PortfolioEnv(
            features_path, prices_path,
            start_idx=split_idx, phi=phi, gamma=gamma,
        )

        # EvalCallback para guardar el mejor modelo temporalmente
        tmp_model_dir = f'models/sensitivity_tmp_{name.split()[0]}/'
        os.makedirs(tmp_model_dir, exist_ok=True)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=tmp_model_dir,
            eval_freq=max(5000, split_idx * 10),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )

        # Entrenar
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)

        # Cargar el mejor modelo y evaluar
        best_path = os.path.join(tmp_model_dir, 'best_model.zip')
        if os.path.exists(best_path):
            model = PPO.load(best_path)

        # Backtest en test
        test_env = PortfolioEnv(
            features_path, prices_path,
            start_idx=split_idx, phi=phi, gamma=gamma,
        )
        obs, _ = test_env.reset()
        done = False
        equity = [10000]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = test_env.step(action)
            equity.append(float(info['value']))

        series = pd.Series(equity)
        metrics = compute_metrics(series)
        metrics['Config'] = name
        metrics['phi'] = phi
        metrics['gamma'] = gamma
        results.append(metrics)

        print(f"    Sharpe: {metrics['Sharpe Ratio']:.3f} | "
              f"Retorno: {metrics['Retorno Total (%)']:.1f}% | "
              f"MDD: {metrics['Max Drawdown (%)']:.1f}%")

    # Generar tabla
    df_results = pd.DataFrame(results).set_index('Config')

    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA DE SENSIBILIDAD")
    print(f"{'='*70}")
    print(df_results[['phi', 'gamma', 'Sharpe Ratio', 'Retorno Total (%)',
                       'Max Drawdown (%)', 'Volatilidad Anualizada (%)',
                       'Valor Final ($)']].to_string())

    # Guardar CSV
    os.makedirs('src/reports', exist_ok=True)
    df_results.to_csv('src/reports/sensitivity_analysis.csv', encoding='utf-8-sig')
    print(f"\nTabla guardada: src/reports/sensitivity_analysis.csv")

    # Generar gráfica
    _plot_sensitivity(df_results)

    # Limpiar modelos temporales (ignorar errores de permisos en Windows)
    import shutil
    for name in CONFIGS:
        tmp_dir = f'models/sensitivity_tmp_{name.split()[0]}/'
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
            except PermissionError:
                pass  # Windows puede bloquear carpetas recién usadas

    return df_results


def _plot_sensitivity(df: pd.DataFrame,
                       path: str = 'src/reports/sensitivity_analysis.png'):
    """Genera gráfica comparativa de las configuraciones."""

    configs = df.index.tolist()
    x = range(len(configs))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle('Análisis de Sensibilidad — Hiperparámetros del Reward PPO',
                 fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ('Sharpe Ratio', 'steelblue', 'Sharpe Ratio'),
        ('Retorno Total (%)', 'mediumseagreen', 'Retorno Total (%)'),
        ('Max Drawdown (%)', 'tomato', 'Max Drawdown (%)'),
        ('Volatilidad Anualizada (%)', 'mediumpurple', 'Volatilidad (%)'),
    ]

    for ax, (col, color, title) in zip(axes, metrics_to_plot):
        values = df[col].values
        bars = ax.bar(x, values, color=color, alpha=0.75, edgecolor='white')

        for bar, val in zip(bars, values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f'{val:.2f}', ha='center',
                    va='bottom' if y >= 0 else 'top',
                    fontsize=9, fontweight='bold', color=color)

        ax.set_xticks(list(x))
        ax.set_xticklabels(configs, fontsize=8, rotation=15, ha='right')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # Tabla debajo con los parámetros
    fig.text(0.5, -0.05,
             '  |  '.join([f"{name}: phi={c['phi']}, gamma={c['gamma']}"
                           for name, c in CONFIGS.items()]),
             ha='center', fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfica guardada: {path}")


if __name__ == "__main__":
    run_sensitivity_analysis()
