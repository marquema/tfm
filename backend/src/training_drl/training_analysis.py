"""
Análisis académico del entrenamiento del agente PPO.

Implementa las técnicas de validación apropiadas para Deep Reinforcement Learning
aplicado a series temporales financieras:

  1. Walk-Forward Validation  — equivalente temporal del cross-validation
  2. Detección de sobreajuste — gap entre reward de train y evaluación
  3. Early stopping informado — curvas de entropía y explained variance
  4. Estabilidad out-of-sample — varianza del Sharpe en múltiples ventanas de test

Nota sobre métricas clásicas de ML:
  - K-fold cross-validation y ROC/AUC no aplican a DRL sobre series temporales.
    K-fold mezclaría futuro y pasado (data leakage). ROC es para clasificación binaria.
  - Walk-forward es el estándar académico para modelos financieros secuenciales.
    Ver: López de Prado (2018), "Advances in Financial Machine Learning", cap. 7.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

try:
    from src.training_drl.environment_trading import PortfolioEnv
    from src.benchmarking.baselines import calcular_metricas
except ImportError:
    from src.training_drl.environment_trading import PortfolioEnv
    from benchmarking.baselines import calcular_metricas


# ---------------------------------------------------------------------------
# Callback: monitorización académica durante el entrenamiento
# ---------------------------------------------------------------------------

class AcademicMonitorCallback(BaseCallback):
    """
    Captura métricas internas de SB3 en cada update de PPO para diagnóstico académico.
        SB3: stable baselines
    Métricas capturadas:
      - entropy_loss   : entropía de la política. Debe decrecer lentamente. Sino, hay overfitting.
                         Colapso rápido indica memorización del conjunto de entrenamiento.
      - value_loss     : error cuadrático del value function. Debe estabilizarse.
                         Divergencia significa que la red de valor no aprende la función de retorno.
      - explained_var  : fracción de varianza de los returns explicada por el value function.
                         Valores < 0 indican que el value function es peor que predecir la media.
                         Objetivo: > 0.5 para una política bien calibrada.
                         Entiende por qué gana dinero.
      - approx_kl      : divergencia KL aproximada entre política vieja y nueva. que no haya un cambio grande entre politivas
                         Valores > 0.05 indican actualizaciones demasiado grandes (inestabilidad).
      - clip_fraction  : fracción de actualizaciones recortadas por PPO clip_range.
                         > 0.3 indica clip_range demasiado pequeño;
                         < 0.01 indica configuración innecesariamente conservadora.

    Parameters
    ----------
    verbose : int
        Nivel de verbosidad (0 = silencioso).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = {
            'timesteps': [],
            'entropy':  [],
            'value_loss': [],
            'explained_var': [],
            'approx_kl':  [],
            'clip_fraction': [],
        }

    def _on_step(self) -> bool:
        """
        Método invocado en cada paso del entrenamiento (requerido por BaseCallback).

        Returns
        -------
        bool
            True para continuar el entrenamiento.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        Captura las métricas del logger de SB3 al final de cada rollout.

        Se ejecuta automáticamente cuando PPO termina de recoger un batch
        de experiencia y antes de la actualización de la red.
        """
        logger = self.model.logger
        if not hasattr(logger, 'name_to_value'):
            return

        vals = logger.name_to_value
        self.metrics['timesteps'].append(self.num_timesteps)
        self.metrics['entropy'].append(vals.get('train/entropy_loss', np.nan))
        self.metrics['value_loss'].append(vals.get('train/value_loss', np.nan))
        self.metrics['explained_var'].append(vals.get('train/explained_variance', np.nan))
        self.metrics['approx_kl'].append(vals.get('train/approx_kl', np.nan))
        self.metrics['clip_fraction'].append(vals.get('train/clip_fraction', np.nan))

    def save_report(self, path: str = 'src/reports/training_diagnostics.png') -> None:
        """
        Genera y guarda el panel de diagnóstico académico del entrenamiento.

        Crea una figura con 6 paneles: entropía, value loss, explained variance,
        KL divergencia, clip fraction y un resumen diagnóstico textual.

        Parameters
        ----------
        path : str
            Ruta donde guardar la imagen PNG del diagnóstico.

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        steps = self.metrics['timesteps']
        if not steps:
            print("[AVISO] No hay métricas capturadas todavía.")
            return

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Diagnóstico Académico del Entrenamiento PPO',
                     fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        panels = [
            (0, 0, 'entropy', 'Entropía de Política',
             'Debe decrecer lentamente\n(colapso rápido = memorización)', 'steelblue'),
            (0, 1, 'value_loss',  'Value Function Loss',
             'Debe estabilizarse\n(divergencia = red de valor inestable)', 'tomato'),
            (0, 2, 'explained_var', 'Explained Variance',
             'Objetivo > 0.5\n(< 0 = peor que predecir la media)', 'mediumseagreen'),
            (1, 0, 'approx_kl', 'KL Divergencia Aproximada',
             'Objetivo < 0.05\n(> 0.05 = actualizaciones inestables)', 'darkorange'),
            (1, 1, 'clip_fraction', 'Fracción de Clipping PPO',
             'Objetivo 0.01-0.3\n(fuera de rango = ajustar clip_range)', 'mediumpurple'),
        ]

        for row, col, key, title, note, color in panels:
            ax = fig.add_subplot(gs[row, col])
            values = self.metrics[key]
            ax.plot(steps, values, color=color, linewidth=1.5, alpha=0.85)
            # Línea de media móvil para suavizar el ruido
            if len(values) >= 10:
                window = max(5, len(values) // 10)
                ma = pd.Series(values).rolling(window, min_periods=1).mean()
                ax.plot(steps, ma, color=color, linewidth=2.5, alpha=0.5,
                        linestyle='--')
            # Líneas de referencia
            if key == 'explained_var':
                ax.axhline(0.5, color='green', linestyle=':', alpha=0.6,
                           label='objetivo > 0.5')
                ax.axhline(0.0, color='red', linestyle=':', alpha=0.6,
                           label='umbral crítico')
            elif key == 'approx_kl':
                ax.axhline(0.05, color='red', linestyle=':', alpha=0.6,
                           label='umbral inestabilidad')
            elif key == 'clip_fraction':
                ax.axhline(0.3, color='orange', linestyle=':', alpha=0.5)
                ax.axhline(0.01, color='orange', linestyle=':', alpha=0.5)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Pasos de entrenamiento', fontsize=8)
            ax.text(0.98, 0.02, note, transform=ax.transAxes,
                    fontsize=7, ha='right', va='bottom', color='gray',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow',
                              alpha=0.7))
            ax.grid(True, alpha=0.3)

        # Panel 6: resumen diagnóstico en texto
        ax_text = fig.add_subplot(gs[1, 2])
        ax_text.axis('off')
        summary = self._generate_diagnostic()
        ax_text.text(0.05, 0.95, summary, transform=ax_text.transAxes,
                     fontsize=9, va='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Diagnóstico guardado en {path}")

    def _generate_diagnostic(self) -> str:
        """
        Genera texto de diagnóstico automático basado en las métricas finales.

        Evalúa explained variance, KL divergencia, clip fraction y evolución
        de la entropía para emitir un veredicto sobre la salud del entrenamiento.

        Returns
        -------
        str
            Texto formateado con el diagnóstico línea por línea.
        """
        lines = ["DIAGNÓSTICO AUTOMÁTICO\n" + "-" * 28]

        def last_valid(key):
            vals = [v for v in self.metrics[key] if not np.isnan(v)]
            return vals[-1] if vals else None

        ev = last_valid('explained_var')
        if ev is not None:
            status = "OK" if ev > 0.5 else ("REGULAR" if ev > 0 else "MAL")
            lines.append(f"Explained Var: {ev:.3f} {status}")

        kl = last_valid('approx_kl')
        if kl is not None:
            status = "ESTABLE" if kl < 0.05 else "INESTABLE"
            lines.append(f"KL Div: {kl:.4f} {status}")

        cf = last_valid('clip_fraction')
        if cf is not None:
            status = "NORMAL" if 0.01 <= cf <= 0.3 else "AJUSTAR clip_range"
            lines.append(f"Clip Frac: {cf:.3f} {status}")

        ent = self.metrics['entropy']
        if len(ent) >= 10:
            drop = (ent[0] - ent[-1]) / (abs(ent[0]) + 1e-8)
            if drop > 0.8:
                lines.append("Entropía colapsó rápido\n   (posible memorización, overfitting)")
            elif drop < 0.1:
                lines.append("Entropía sin cambio\n   (agente no aprendió)")
            else:
                lines.append("Entropía decrece gradualmente")

        return "\n".join(lines)

    #todo ver esto de la retro compaitbilidad
    # --- Compatibilidad hacia atrás ---
    guardar_reporte = save_report

    @property
    def metricas(self):
        """Alias de compatibilidad hacia atrás para self.metrics."""
        return self.metrics


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

# todo: no habría data leakage aquí?
def walk_forward_validation(features_path: str,
                            prices_path: str,
                            train_days: int = 504,
                            test_days: int  = 252,
                            total_timesteps: int = 100000) -> pd.DataFrame:
    """
    Validación Walk-Forward con ventanas de tamaño fijo.
    El número de ventanas lo determina el propio dataset, no es un parámetro.
    Avanza un periodo de test cada vez (rolling de 1 año).
    Hace un proceso circular: entrena con los datos de los años 1 y 2, y testea con el 3, etc.

    Criterios de tamaño de ventana:
      - train_days = 504 (2 años): mínimo para que PPO vea suficientes episodios completos
      - test_days  = 252 (1 año): mínimo para que el Sharpe anualizado sea estadísticamente
        fiable (con < 252 días el factor sqrt(252) distorsiona la métrica)

    Con el dataset actual (2018-2026, 8.2 años) genera 6 ventanas automáticamente.
    Si se amplía el rango histórico, el número de ventanas crece sin cambiar el código.

    Permite detectar:
      - Sharpe medio > 0 en todas las ventanas: política generalmente rentable
      - Alta varianza entre ventanas: política inestable, dependiente del régimen
      - Degradación monotónica en ventanas recientes: overfitting a datos viejos

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas.
    prices_path : str
        Ruta al CSV de precios originales.
    train_days : int
        Días de entrenamiento por ventana (por defecto 504, ~2 años).
    test_days : int
        Días de test por ventana (por defecto 252, ~1 año).
    total_timesteps : int
        Pasos de entrenamiento PPO por ventana.

    Returns
    -------
    pd.DataFrame
        DataFrame indexado por número de ventana con métricas de cada período
        de test (Sharpe Ratio, Retorno Total, Max Drawdown, etc.).

    References
    ----------
    Lopez de Prado (2018), "Advances in Financial Machine Learning", cap. 7.
    """
    df_f    = pd.read_csv(features_path, index_col=0)
    n_total = len(df_f)

    # -- Ventanas adaptativas --
    # Si el dataset no alcanza los valores académicos ideales (504+252=756 días),
    # se escalan proporcionalmente manteniendo la ratio 2:1 (train:test).
    # Mínimos absolutos: test >= 21 días (1 mes), train >= test * 2.
    MIN_TEST  = 21
    MIN_TRAIN = MIN_TEST * 2
    if n_total < MIN_TRAIN + MIN_TEST:
        raise ValueError(
            f"Dataset demasiado pequeño ({n_total} dias). "
            f"Necesita al menos {MIN_TRAIN + MIN_TEST} dias para walk-forward. "
            f"Amplía el rango de fechas en /fase1/preparar-datos."
        )

    if train_days + test_days > n_total:
        # Escalar: test = 20% del total, train = 40% del total (ratio 2:1)
        test_days  = max(MIN_TEST,  n_total // 5)
        train_days = max(MIN_TRAIN, n_total * 2 // 5)
        print(f"  [AVISO] Dataset corto ({n_total}d). "
              f"Ventanas adaptadas automaticamente: train={train_days}d, test={test_days}d")

    # Calcular ventanas: el número surge del dato, no del parámetro
    windows = []
    start = 0
    while start + train_days + test_days <= n_total:
        split = start + train_days
        end   = split + test_days
        windows.append((start, split, end))
        start += test_days   # avanza exactamente 1 periodo de test

    n_windows = len(windows)

    results = []
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"Dataset: {n_total} dias ({n_total/252:.1f} anios) | "
          f"Train: {train_days}d ({train_days/252:.1f}a) | "
          f"Test: {test_days}d ({test_days/252:.1f}a)")
    print(f"Ventanas calculadas automaticamente: {n_windows}")
    print(f"{'='*60}")

    for i, (start, split, end) in enumerate(windows):
        date_start = df_f.index[start][:10]
        date_split = df_f.index[split][:10]
        date_end   = df_f.index[end - 1][:10]
        print(f"\n[Ventana {i+1}/{n_windows}] "
              f"Train: {date_start} a {date_split} ({split-start}d) | "
              f"Test: {date_split} a {date_end} ({end-split}d)")

        # Entrenamiento dentro de la ventana
        train_env = PortfolioEnv(features_path, prices_path,
                                 start_idx=start, end_idx=split)
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=3e-4, n_steps=512, batch_size=64,
            clip_range=0.2, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0
        )
        model.learn(total_timesteps=total_timesteps)

        # Evaluación out-of-sample en la ventana de test
        test_env = PortfolioEnv(features_path, prices_path,
                                start_idx=split, end_idx=end)
        obs, _ = test_env.reset()
        done   = False
        values = [test_env.initial_balance]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = test_env.step(action)
            values.append(info['value'])

        series = pd.Series(values)
        window_metrics = calcular_metricas(series)
        window_metrics['ventana'] = i + 1
        window_metrics['dias_train'] = split - start
        window_metrics['dias_test']  = end - split
        results.append(window_metrics)

        print(f"  Sharpe: {window_metrics['Sharpe Ratio']:.3f} | "
              f"Retorno: {window_metrics['Retorno Total (%)']:.1f}% | "
              f"MDD: {window_metrics['Max Drawdown (%)']:.1f}%")

    df_wf = pd.DataFrame(results).set_index('ventana')

    print(f"\n{'='*60}")
    print("RESUMEN WALK-FORWARD")
    print(f"  Sharpe medio:    {df_wf['Sharpe Ratio'].mean():.3f}  "
          f"(+-{df_wf['Sharpe Ratio'].std():.3f})")
    print(f"  Retorno medio:   {df_wf['Retorno Total (%)'].mean():.1f}%")
    print(f"  MDD medio:       {df_wf['Max Drawdown (%)'].mean():.1f}%")
    print(f"  Ventanas con Sharpe > 0: "
          f"{(df_wf['Sharpe Ratio'] > 0).sum()} / {n_windows}")
    print(f"{'='*60}")

    # Guardar resultados
    os.makedirs('src/reports', exist_ok=True)
    df_wf.to_csv('src/reports/walk_forward_results.csv')
    _plot_walk_forward(df_wf)

    return df_wf


def _plot_walk_forward(df: pd.DataFrame,
                       path: str = 'src/reports/walk_forward_analysis.png') -> None:
    """
    Genera la visualización del análisis walk-forward con 3 paneles.

    Muestra barras de Sharpe Ratio, Retorno Total y Max Drawdown por ventana,
    junto con líneas de media para facilitar la interpretación visual.

    Parameters
    ----------
    df : pd.DataFrame
        Resultados del walk-forward indexados por número de ventana.
    path : str
        Ruta donde guardar la imagen PNG.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Walk-Forward Validation — Estabilidad de la Política',
                 fontsize=13, fontweight='bold')

    window_ids = df.index.tolist()

    for ax, col, title, color, threshold in [
        (axes[0], 'Sharpe Ratio',      'Sharpe Ratio por Ventana',
         'steelblue',  0.0),
        (axes[1], 'Retorno Total (%)', 'Retorno Total (%) por Ventana',
         'mediumseagreen', 0.0),
        (axes[2], 'Max Drawdown (%)',  'Max Drawdown (%) por Ventana',
         'tomato', None),
    ]:
        values = df[col].values
        ax.bar(window_ids, values, color=color, alpha=0.75, edgecolor='white')
        if threshold is not None:
            ax.axhline(threshold, color='black', linestyle='--', alpha=0.5,
                       linewidth=1)
        # Media
        mean_val = np.mean(values)
        ax.axhline(mean_val, color=color, linestyle='-', alpha=0.9,
                   linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Ventana de Validación')
        ax.set_xticks(window_ids)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica walk-forward guardada en {path}")


# ---------------------------------------------------------------------------
# Detección de sobreajuste: gap train vs eval
# ---------------------------------------------------------------------------

class OverfitDetectorCallback(BaseCallback):
    """
    Detecta sobreajuste comparando el reward en train y en evaluación.

    Señal de sobreajuste: reward_train >> reward_eval de forma sostenida.
    Implementa early stopping con paciencia configurable.

    Parameters
    ----------
    eval_env : gym.Env
        Entorno de validación (distinto del de entrenamiento).
    eval_freq : int
        Frecuencia de evaluación en pasos de entrenamiento.
    n_eval_ep : int
        Número de episodios de evaluación para estabilizar la media.
    patience : int
        Evaluaciones consecutivas sin mejora antes de detener el entrenamiento.
    min_improvement_pct : float
        Incremento mínimo relativo del reward de eval para contar como mejora.
    verbose : int
        Nivel de verbosidad (0 = silencioso, 1 = log por evaluación).
    """

    def __init__(self, eval_env, eval_freq=10000, n_eval_ep=3,
                 patience=5, min_improvement_pct=0.02, verbose=1):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eval_ep = n_eval_ep
        self.patience = patience
        # Mejora mínima en términos RELATIVOS al mejor valor visto (2% por defecto).
        # Usar porcentaje evita que la escala del reward (puede ser -50 a +50)
        # haga que min_mejora absoluta sea irrelevante.
        self.min_improvement_pct = min_improvement_pct

        self.train_history       = []
        self.eval_history        = []
        self.timesteps_log       = []
        self.best_eval           = -np.inf
        self.steps_without_improvement = 0

    def _on_step(self) -> bool:
        """
        Evaluación periódica del agente en el entorno de validación.

        Compara el reward medio por paso en train y eval, detecta
        sobreajuste por gap relativo > 50%, y activa early stopping
        si no hay mejora tras `patience` evaluaciones consecutivas.

        Returns
        -------
        bool
            False para detener el entrenamiento (early stopping), True para continuar.
        """
        if self.num_timesteps % self.eval_freq != 0:
            return True

        # Reward por step en train (ep_info_buffer contiene 'r'=total y 'l'=length).
        # Normalizar por longitud del episodio evita que datasets más largos produzcan
        # rewards acumulados mayores en magnitud y hagan incomparable la señal de mejora.
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            reward_train = np.mean([
                ep['r'] / max(ep['l'], 1) for ep in self.model.ep_info_buffer
            ])
        else:
            reward_train = np.nan

        # Reward por step en eval (N episodios deterministas)
        rewards_eval = []
        for _ in range(self.n_eval_ep):
            obs, _ = self.eval_env.reset()
            done   = False
            total  = 0.0
            n_steps = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, _, _ = self.eval_env.step(action)
                total += rew
                n_steps += 1
            rewards_eval.append(total / max(n_steps, 1))
        reward_eval = np.mean(rewards_eval)

        self.train_history.append(reward_train)
        self.eval_history.append(reward_eval)
        self.timesteps_log.append(self.num_timesteps)

        # Early stopping con paciencia.
        # Umbral de mejora: los rewards son por step (escala ~[-0.5, 0.5]),
        # así que usamos un umbral absoluto pequeño fijo (0.001) en lugar de
        # relativo al mejor valor, que con valores negativos grandes era demasiado exigente.
        improvement_threshold = 0.001
        if reward_eval > self.best_eval + improvement_threshold:
            self.best_eval = reward_eval
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        # Señal de sobreajuste: gap relativo > 50%. Gap entre entrenamiento y test
        gap = reward_train - reward_eval
        relative_gap = (gap / (abs(reward_train) + 1e-8)
                        if not np.isnan(reward_train) else 0)

        if self.verbose >= 1:
            print(f"  [paso {self.num_timesteps}] "
                  f"R_train={reward_train:.3f} | R_eval={reward_eval:.3f} | "
                  f"Gap={gap:.3f} | Paciencia="
                  f"{self.steps_without_improvement}/{self.patience}")

        if relative_gap > 0.5 and not np.isnan(reward_train):
            print(f"  GAP SOBREAJUSTE DETECTADO: train/eval = {relative_gap:.1%}")

        if self.steps_without_improvement >= self.patience:
            print(f"\n  EARLY STOPPING activado: {self.patience} evaluaciones sin mejora.")
            print(f"  Mejor R_eval alcanzado: {self.best_eval:.4f}")
            return False  # Detiene el entrenamiento

        return True

    def save_curves(self, path: str = 'src/reports/overfitting_analysis.png') -> None:
        """
        Genera y guarda la visualización del gap train vs eval para detectar sobreajuste.

        Panel izquierdo: curvas de reward en train y eval a lo largo del entrenamiento.
        Panel derecho: barras del gap (train - eval) coloreadas por signo.

        Parameters
        ----------
        path : str
            Ruta donde guardar la imagen PNG.

        Returns
        -------
        None
        """
        if not self.timesteps_log:
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Análisis de Sobreajuste — Train vs Evaluación',
                     fontsize=13, fontweight='bold')

        # Panel izquierdo: curvas train y eval
        ax1.plot(self.timesteps_log, self.train_history,
                 label='Reward Train', color='steelblue', linewidth=2)
        ax1.plot(self.timesteps_log, self.eval_history,
                 label='Reward Eval (OOS)', color='tomato', linewidth=2)
        ax1.fill_between(self.timesteps_log,
                         self.train_history, self.eval_history,
                         alpha=0.15, color='orange', label='Gap sobreajuste')
        ax1.set_title('Reward: Train vs Out-of-Sample',
                      fontsize=11, fontweight='bold')
        ax1.set_xlabel('Pasos de entrenamiento')
        ax1.set_ylabel('Reward acumulado por episodio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel derecho: gap relativo
        gaps = [t - e for t, e in zip(self.train_history, self.eval_history)
                if not np.isnan(t)]
        steps_gap = [s for s, t in zip(self.timesteps_log, self.train_history)
                     if not np.isnan(t)]
        colors = ['tomato' if g > 0 else 'mediumseagreen' for g in gaps]
        ax2.bar(steps_gap, gaps, color=colors, alpha=0.7,
                width=max(steps_gap) / len(steps_gap) * 0.8)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_title('Gap Train-Eval\n(positivo = posible sobreajuste)',
                      fontsize=11, fontweight='bold')
        ax2.set_xlabel('Pasos de entrenamiento')
        ax2.set_ylabel('Reward_train - Reward_eval')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Curvas de sobreajuste guardadas en {path}")

    # --- Compatibilidad hacia atrás ---
    guardar_curvas = save_curves

    @property
    def historial_train(self):
        """Alias de compatibilidad hacia atrás para self.train_history."""
        return self.train_history

    @property
    def historial_eval(self):
        """Alias de compatibilidad hacia atrás para self.eval_history."""
        return self.eval_history


# ---------------------------------------------------------------------------
# Función principal: entrenamiento académico completo
# ---------------------------------------------------------------------------

def train_academic(features_path: str = 'data/normalized_features.csv',
                   prices_path: str   = 'data/original_prices.csv',
                   total_timesteps: int = 500000,
                   split_pct: float= 0.8,
                   patience: int = 8) -> PPO:
    """
    Entrenamiento con monitorización académica completa.

    Incluye diagnóstico de entropía, value loss y explained variance,
    early stopping basado en reward out-of-sample con paciencia adaptativa,
    y detección automática de sobreajuste (gap train/eval).

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas.
    prices_path : str
        Ruta al CSV de precios originales.
    total_timesteps : int
        Pasos máximos de entrenamiento (puede parar antes por early stopping).
    split_pct : float
        Fracción de datos para entrenamiento (el resto se usa para evaluación).
    patience : int
        Evaluaciones base sin mejora antes de considerar early stopping.

    Returns
    -------
    PPO
        Modelo PPO entrenado (también guardado en models/best_model_academic/).
    """
    print("=" * 60)
    print("ENTRENAMIENTO ACADÉMICO CON VALIDACIÓN TEMPORAL")
    print("=" * 60)

    df_f = pd.read_csv(features_path, index_col=0)
    split_idx = int(len(df_f) * split_pct)

    train_env = PortfolioEnv(features_path, prices_path, end_idx=split_idx)
    eval_env  = PortfolioEnv(features_path, prices_path, start_idx=split_idx)

    # eval_freq adaptativo: evaluar cada ~20 episodios completos de entrenamiento.
    # Con dataset largo (split_idx=2000 steps/ep), eval_freq=10000 -> solo 5 episodios
    # entre evaluaciones -> señal de mejora ruidosa -> early stopping prematuro.
    # Con dataset corto (split_idx=100 steps/ep), eval_freq=10000 -> 100 episodios
    # entre evaluaciones -> umbral de mejora relativo se vuelve muy pequeño -> nunca para.
    # Solución: eval_freq = 20 episodios * longitud del episodio de train.
    ep_len_train = split_idx  # el entorno recorre todo el split en cada episodio
    eval_freq = max(5000, min(50000, ep_len_train * 20))

    # Patience adaptativa: más pasos totales -> más evaluaciones antes de rendir.
    # Con total_timesteps=500000 y eval_freq adaptativo, habrá ~total/eval_freq evaluaciones.
    # Patience = 30% de esas evaluaciones, con mínimo 5 y máximo 15.
    n_total_evals = total_timesteps // eval_freq
    effective_patience = max(5, min(15, n_total_evals // 3))

    print(f"  Longitud episodio train: {ep_len_train} steps")
    print(f"  eval_freq adaptativo:    {eval_freq} steps (~20 episodios)")
    print(f"  Evaluaciones previstas:  {n_total_evals}")
    print(f"  Patience efectiva:       {effective_patience} evaluaciones sin mejora")

    # Callbacks
    monitor_cb  = AcademicMonitorCallback(verbose=0)
    overfit_cb  = OverfitDetectorCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_ep=3,
        patience=effective_patience,
        verbose=1
    )
    # EvalCallback guarda el mejor modelo según reward medio
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_model_academic/',
        log_path='./logs/academic/',
        eval_freq=eval_freq,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/"
    )

    print(f"\nIniciando entrenamiento (máx. {total_timesteps:,} pasos, "
          f"early stopping con paciencia={patience})...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[monitor_cb, overfit_cb, eval_cb]
    )

    # Guardar reportes
    print("\nGenerando reportes de diagnóstico...")
    monitor_cb.save_report('src/reports/training_diagnostics.png')
    overfit_cb.save_curves('src/reports/overfitting_analysis.png')

    model.save("models/ppo_academic_final")
    print("\nModelo final guardado en models/ppo_academic_final.zip")
    print("Mejor modelo (por eval reward) en models/best_model_academic/")

    return model


# ---------------------------------------------------------------------------
# Compatibilidad hacia atrás: aliases de funciones renombradas
# ---------------------------------------------------------------------------
entrenar_academico = train_academic
