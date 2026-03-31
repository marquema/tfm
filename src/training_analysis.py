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
    from src.environment_trading import PortfolioEnv
    from src.benchmarking.baselines import calcular_metricas
except ImportError:
    from environment_trading import PortfolioEnv
    from benchmarking.baselines import calcular_metricas


# ─────────────────────────────────────────────────────────────────────────────
# Callback: monitorización académica durante el entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

class AcademicMonitorCallback(BaseCallback):
    """
    Captura métricas internas de SB3 en cada update de PPO para diagnóstico académico:

      - entropy_loss   : entropía de la política. Debe decrecer lentamente.
                         Colapso rápido → agente se ha "memorizado" el train.
      - value_loss     : error cuadrático del value function. Debe estabilizarse.
                         Si diverge → la red de valor no aprende la función de retorno.
      - explained_var  : fracción de varianza de los returns explicada por el value function.
                         Valores < 0 → el value function es peor que predecir la media.
                         Objetivo: > 0.5 para una política bien calibrada.
      - approx_kl      : divergencia KL aproximada entre política vieja y nueva.
                         Valores > 0.05 indican actualizaciones demasiado grandes (inestabilidad).
      - clip_fraction  : fracción de actualizaciones recortadas por PPO clip_range.
                         > 0.3 → clip_range demasiado pequeño; < 0.01 → innecesariamente conservador.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metricas = {
            'timesteps':     [],
            'entropy':       [],
            'value_loss':    [],
            'explained_var': [],
            'approx_kl':     [],
            'clip_fraction': [],
        }

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Captura las métricas del logger de SB3 al final de cada rollout."""
        logger = self.model.logger
        if not hasattr(logger, 'name_to_value'):
            return

        vals = logger.name_to_value
        self.metricas['timesteps'].append(self.num_timesteps)
        self.metricas['entropy'].append(vals.get('train/entropy_loss', np.nan))
        self.metricas['value_loss'].append(vals.get('train/value_loss', np.nan))
        self.metricas['explained_var'].append(vals.get('train/explained_variance', np.nan))
        self.metricas['approx_kl'].append(vals.get('train/approx_kl', np.nan))
        self.metricas['clip_fraction'].append(vals.get('train/clip_fraction', np.nan))

    def guardar_reporte(self, ruta: str = 'src/reports/training_diagnostics.png') -> None:
        """Genera el panel de diagnóstico académico del entrenamiento."""
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        steps = self.metricas['timesteps']
        if not steps:
            print("[AVISO] No hay métricas capturadas todavía.")
            return

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Diagnóstico Académico del Entrenamiento PPO', fontsize=14, fontweight='bold')
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        paneles = [
            (0, 0, 'entropy',       'Entropía de Política',         'Debe decrecer lentamente\n(colapso rápido = memorización)', 'steelblue'),
            (0, 1, 'value_loss',    'Value Function Loss',           'Debe estabilizarse\n(divergencia = red de valor inestable)', 'tomato'),
            (0, 2, 'explained_var', 'Explained Variance',            'Objetivo > 0.5\n(< 0 = peor que predecir la media)', 'mediumseagreen'),
            (1, 0, 'approx_kl',     'KL Divergencia Aproximada',     'Objetivo < 0.05\n(> 0.05 = actualizaciones inestables)', 'darkorange'),
            (1, 1, 'clip_fraction', 'Fracción de Clipping PPO',      'Objetivo 0.01–0.3\n(fuera de rango = ajustar clip_range)', 'mediumpurple'),
        ]

        for fila, col, clave, titulo, nota, color in paneles:
            ax = fig.add_subplot(gs[fila, col])
            valores = self.metricas[clave]
            ax.plot(steps, valores, color=color, linewidth=1.5, alpha=0.85)
            # Línea de media móvil para suavizar el ruido
            if len(valores) >= 10:
                ventana = max(5, len(valores) // 10)
                ma = pd.Series(valores).rolling(ventana, min_periods=1).mean()
                ax.plot(steps, ma, color=color, linewidth=2.5, alpha=0.5, linestyle='--')
            # Líneas de referencia
            if clave == 'explained_var':
                ax.axhline(0.5, color='green', linestyle=':', alpha=0.6, label='objetivo > 0.5')
                ax.axhline(0.0, color='red',   linestyle=':', alpha=0.6, label='umbral crítico')
            elif clave == 'approx_kl':
                ax.axhline(0.05, color='red', linestyle=':', alpha=0.6, label='umbral inestabilidad')
            elif clave == 'clip_fraction':
                ax.axhline(0.3, color='orange', linestyle=':', alpha=0.5)
                ax.axhline(0.01, color='orange', linestyle=':', alpha=0.5)
            ax.set_title(titulo, fontsize=10, fontweight='bold')
            ax.set_xlabel('Pasos de entrenamiento', fontsize=8)
            ax.text(0.98, 0.02, nota, transform=ax.transAxes,
                    fontsize=7, ha='right', va='bottom', color='gray',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
            ax.grid(True, alpha=0.3)

        # Panel 6: resumen diagnóstico en texto
        ax_text = fig.add_subplot(gs[1, 2])
        ax_text.axis('off')
        resumen = self._generar_diagnostico()
        ax_text.text(0.05, 0.95, resumen, transform=ax_text.transAxes,
                     fontsize=9, va='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

        plt.savefig(ruta, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Diagnóstico guardado en {ruta}")

    def _generar_diagnostico(self) -> str:
        """Texto de diagnóstico automático basado en las métricas finales."""
        lineas = ["DIAGNÓSTICO AUTOMÁTICO\n" + "─" * 28]

        def ultimo_valido(clave):
            vals = [v for v in self.metricas[clave] if not np.isnan(v)]
            return vals[-1] if vals else None

        ev = ultimo_valido('explained_var')
        if ev is not None:
            estado = "✓ BIEN" if ev > 0.5 else ("⚠ REGULAR" if ev > 0 else "✗ MAL")
            lineas.append(f"Explained Var: {ev:.3f} {estado}")

        kl = ultimo_valido('approx_kl')
        if kl is not None:
            estado = "✓ ESTABLE" if kl < 0.05 else "⚠ INESTABLE"
            lineas.append(f"KL Div: {kl:.4f} {estado}")

        cf = ultimo_valido('clip_fraction')
        if cf is not None:
            estado = "✓ NORMAL" if 0.01 <= cf <= 0.3 else "⚠ AJUSTAR clip_range"
            lineas.append(f"Clip Frac: {cf:.3f} {estado}")

        ent = self.metricas['entropy']
        if len(ent) >= 10:
            caida = (ent[0] - ent[-1]) / (abs(ent[0]) + 1e-8)
            if caida > 0.8:
                lineas.append("⚠ Entropía colapsó rápido\n   (posible memorización)")
            elif caida < 0.1:
                lineas.append("⚠ Entropía sin cambio\n   (agente no aprendió)")
            else:
                lineas.append("✓ Entropía decrece gradualmente")

        return "\n".join(lineas)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_validation(features_path: str,
                             prices_path: str,
                             dias_train: int = 504,
                             dias_test: int  = 252,
                             total_timesteps: int = 100000) -> pd.DataFrame:
    """
    Validación Walk-Forward con ventanas de tamaño fijo.

    El número de ventanas lo determina el propio dataset — no es un parámetro.
    Avanza un periodo de test cada vez (rolling de 1 año).

    Criterios de tamaño de ventana:
      - dias_train = 504 (2 años): mínimo para que PPO vea suficientes episodios completos
      - dias_test  = 252 (1 año): mínimo para que el Sharpe anualizado sea estadísticamente fiable
        (con < 252 días el factor sqrt(252) distorsiona la métrica)

    Con el dataset actual (2018-2026, 8.2 años) genera 6 ventanas automáticamente.
    Si se amplía el rango histórico, el número de ventanas crece sin cambiar el código.

    Permite detectar:
      - Sharpe medio > 0 en todas las ventanas: política generalmente rentable
      - Alta varianza entre ventanas: política inestable, dependiente del régimen
      - Degradación monotónica en ventanas recientes: overfitting a datos viejos

    Referencia
    ----------
    Lopez de Prado (2018), "Advances in Financial Machine Learning", cap. 7.
    """
    df_f    = pd.read_csv(features_path, index_col=0)
    n_total = len(df_f)

    # ── Ventanas adaptativas ───────────────────────────────────────────────────
    # Si el dataset no alcanza los valores académicos ideales (504+252=756 días),
    # se escalan proporcionalmente manteniendo la ratio 2:1 (train:test).
    # Mínimos absolutos: test ≥ 21 días (1 mes), train ≥ test × 2.
    MIN_TEST  = 21
    MIN_TRAIN = MIN_TEST * 2
    if n_total < MIN_TRAIN + MIN_TEST:
        raise ValueError(
            f"Dataset demasiado pequeño ({n_total} dias). "
            f"Necesita al menos {MIN_TRAIN + MIN_TEST} dias para walk-forward. "
            f"Amplía el rango de fechas en /fase1/preparar-datos."
        )

    if dias_train + dias_test > n_total:
        # Escalar: test = 20% del total, train = 40% del total (ratio 2:1)
        dias_test  = max(MIN_TEST,  n_total // 5)
        dias_train = max(MIN_TRAIN, n_total * 2 // 5)
        print(f"  [AVISO] Dataset corto ({n_total}d). "
              f"Ventanas adaptadas automaticamente: train={dias_train}d, test={dias_test}d")

    # Calcular ventanas: el número surge del dato, no del parámetro
    ventanas = []
    inicio = 0
    while inicio + dias_train + dias_test <= n_total:
        split = inicio + dias_train
        fin   = split + dias_test
        ventanas.append((inicio, split, fin))
        inicio += dias_test   # avanza exactamente 1 periodo de test

    n_ventanas = len(ventanas)

    resultados = []
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION")
    print(f"Dataset: {n_total} dias ({n_total/252:.1f} anios) | "
          f"Train: {dias_train}d ({dias_train/252:.1f}a) | "
          f"Test: {dias_test}d ({dias_test/252:.1f}a)")
    print(f"Ventanas calculadas automaticamente: {n_ventanas}")
    print(f"{'='*60}")

    for i, (inicio, split, fin) in enumerate(ventanas):
        fecha_ini   = df_f.index[inicio][:10]
        fecha_split = df_f.index[split][:10]
        fecha_fin   = df_f.index[fin - 1][:10]
        print(f"\n[Ventana {i+1}/{n_ventanas}] "
              f"Train: {fecha_ini} a {fecha_split} ({split-inicio}d) | "
              f"Test: {fecha_split} a {fecha_fin} ({fin-split}d)")

        # Entrenamiento dentro de la ventana
        train_env = PortfolioEnv(features_path, prices_path,
                                 start_idx=inicio, end_idx=split)
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
                                start_idx=split, end_idx=fin)
        obs, _ = test_env.reset()
        done   = False
        valor  = [test_env.initial_balance]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = test_env.step(action)
            valor.append(info['value'])

        serie = pd.Series(valor)
        metricas = calcular_metricas(serie)
        metricas['ventana'] = i + 1
        metricas['dias_train'] = split - inicio
        metricas['dias_test']  = fin - split
        resultados.append(metricas)

        print(f"  Sharpe: {metricas['Sharpe Ratio']:.3f} | "
              f"Retorno: {metricas['Retorno Total (%)']:.1f}% | "
              f"MDD: {metricas['Max Drawdown (%)']:.1f}%")

    df_wf = pd.DataFrame(resultados).set_index('ventana')

    print(f"\n{'='*60}")
    print("RESUMEN WALK-FORWARD")
    print(f"  Sharpe medio:    {df_wf['Sharpe Ratio'].mean():.3f}  "
          f"(±{df_wf['Sharpe Ratio'].std():.3f})")
    print(f"  Retorno medio:   {df_wf['Retorno Total (%)'].mean():.1f}%")
    print(f"  MDD medio:       {df_wf['Max Drawdown (%)'].mean():.1f}%")
    print(f"  Ventanas con Sharpe > 0: "
          f"{(df_wf['Sharpe Ratio'] > 0).sum()} / {n_ventanas}")
    print(f"{'='*60}")

    # Guardar resultados
    os.makedirs('src/reports', exist_ok=True)
    df_wf.to_csv('src/reports/walk_forward_results.csv')
    _plot_walk_forward(df_wf)

    return df_wf


def _plot_walk_forward(df: pd.DataFrame,
                       ruta: str = 'src/reports/walk_forward_analysis.png') -> None:
    """Visualización del análisis walk-forward."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Walk-Forward Validation — Estabilidad de la Política',
                 fontsize=13, fontweight='bold')

    ventanas = df.index.tolist()

    for ax, col, titulo, color, umbral in [
        (axes[0], 'Sharpe Ratio',      'Sharpe Ratio por Ventana', 'steelblue',  0.0),
        (axes[1], 'Retorno Total (%)', 'Retorno Total (%) por Ventana', 'mediumseagreen', 0.0),
        (axes[2], 'Max Drawdown (%)',  'Max Drawdown (%) por Ventana', 'tomato', None),
    ]:
        valores = df[col].values
        barras  = ax.bar(ventanas, valores, color=color, alpha=0.75, edgecolor='white')
        if umbral is not None:
            ax.axhline(umbral, color='black', linestyle='--', alpha=0.5, linewidth=1)
        # Media
        media = np.mean(valores)
        ax.axhline(media, color=color, linestyle='-', alpha=0.9, linewidth=2,
                   label=f'Media: {media:.2f}')
        ax.set_title(titulo, fontsize=11, fontweight='bold')
        ax.set_xlabel('Ventana de Validación')
        ax.set_xticks(ventanas)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(ruta, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica walk-forward guardada en {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# Detección de sobreajuste: gap train vs eval
# ─────────────────────────────────────────────────────────────────────────────

class OverfitDetectorCallback(BaseCallback):
    """
    Detecta sobreajuste comparando el reward en train y en evaluación.

    Señal de sobreajuste: reward_train >> reward_eval de forma sostenida.
    Implementa early stopping con paciencia configurable.

    Parámetros
    ----------
    eval_env      : entorno de validación (distinto del de entrenamiento)
    eval_freq     : frecuencia de evaluación en pasos
    n_eval_ep     : episodios de evaluación para estabilizar la media
    patience      : paradas consecutivas sin mejora antes de detener
    min_mejora    : incremento mínimo del reward de eval para contar como mejora
    """

    def __init__(self, eval_env, eval_freq=10000, n_eval_ep=3,
                 patience=5, min_mejora_pct=0.02, verbose=1):
        super().__init__(verbose)
        self.eval_env       = eval_env
        self.eval_freq      = eval_freq
        self.n_eval_ep      = n_eval_ep
        self.patience       = patience
        # Mejora mínima en términos RELATIVOS al mejor valor visto (2% por defecto).
        # Usar porcentaje evita que la escala del reward (puede ser -50 a +50)
        # haga que min_mejora absoluta sea irrelevante.
        self.min_mejora_pct = min_mejora_pct

        self.historial_train = []
        self.historial_eval  = []
        self.timesteps_log   = []
        self.mejor_eval      = -np.inf
        self.pasos_sin_mejora = 0

    def _on_step(self) -> bool:
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
            pasos  = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rew, done, _, _ = self.eval_env.step(action)
                total += rew
                pasos += 1
            rewards_eval.append(total / max(pasos, 1))
        reward_eval = np.mean(rewards_eval)

        self.historial_train.append(reward_train)
        self.historial_eval.append(reward_eval)
        self.timesteps_log.append(self.num_timesteps)

        # Early stopping con paciencia.
        # Umbral de mejora: ahora los rewards son por step (escala ~[-0.5, 0.5]),
        # así que usamos un umbral absoluto pequeño fijo (0.001) en lugar de
        # relativo al mejor valor, que con valores negativos grandes era demasiado exigente.
        umbral_mejora = 0.001
        if reward_eval > self.mejor_eval + umbral_mejora:
            self.mejor_eval       = reward_eval
            self.pasos_sin_mejora = 0
        else:
            self.pasos_sin_mejora += 1

        # Señal de sobreajuste: gap relativo > 50%
        gap = reward_train - reward_eval
        gap_relativo = gap / (abs(reward_train) + 1e-8) if not np.isnan(reward_train) else 0

        if self.verbose >= 1:
            print(f"  [paso {self.num_timesteps}] "
                  f"R_train={reward_train:.3f} | R_eval={reward_eval:.3f} | "
                  f"Gap={gap:.3f} | Paciencia={self.pasos_sin_mejora}/{self.patience}")

        if gap_relativo > 0.5 and not np.isnan(reward_train):
            print(f"  ⚠ GAP SOBREAJUSTE DETECTADO: train/eval = {gap_relativo:.1%}")

        if self.pasos_sin_mejora >= self.patience:
            print(f"\n  EARLY STOPPING activado: {self.patience} evaluaciones sin mejora.")
            print(f"  Mejor R_eval alcanzado: {self.mejor_eval:.4f}")
            return False  # Detiene el entrenamiento

        return True

    def guardar_curvas(self, ruta: str = 'src/reports/overfitting_analysis.png') -> None:
        """Visualiza el gap train vs eval para detectar sobreajuste."""
        if not self.timesteps_log:
            return

        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Análisis de Sobreajuste — Train vs Evaluación',
                     fontsize=13, fontweight='bold')

        # Panel izquierdo: curvas train y eval
        ax1.plot(self.timesteps_log, self.historial_train,
                 label='Reward Train', color='steelblue', linewidth=2)
        ax1.plot(self.timesteps_log, self.historial_eval,
                 label='Reward Eval (OOS)', color='tomato', linewidth=2)
        ax1.fill_between(self.timesteps_log,
                         self.historial_train, self.historial_eval,
                         alpha=0.15, color='orange', label='Gap sobreajuste')
        ax1.set_title('Reward: Train vs Out-of-Sample', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Pasos de entrenamiento')
        ax1.set_ylabel('Reward acumulado por episodio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel derecho: gap relativo
        gaps = [t - e for t, e in zip(self.historial_train, self.historial_eval)
                if not np.isnan(t)]
        steps_gap = [s for s, t in zip(self.timesteps_log, self.historial_train)
                     if not np.isnan(t)]
        colores = ['tomato' if g > 0 else 'mediumseagreen' for g in gaps]
        ax2.bar(steps_gap, gaps, color=colores, alpha=0.7, width=max(steps_gap)/len(steps_gap)*0.8)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_title('Gap Train–Eval\n(positivo = posible sobreajuste)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Pasos de entrenamiento')
        ax2.set_ylabel('Reward_train − Reward_eval')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(ruta, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Curvas de sobreajuste guardadas en {ruta}")


# ─────────────────────────────────────────────────────────────────────────────
# Función principal: entrenamiento académico completo
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_academico(features_path: str = 'data/normalized_features.csv',
                        prices_path: str   = 'data/original_prices.csv',
                        total_timesteps: int = 500000,
                        split_pct: float     = 0.8,
                        patience: int        = 8) -> PPO:
    """
    Entrenamiento con monitorización académica completa:
      - Diagnóstico de entropía, value loss y explained variance
      - Early stopping basado en reward out-of-sample con paciencia
      - Detección automática de sobreajuste (gap train/eval)

    Parámetros
    ----------
    total_timesteps : pasos máximos de entrenamiento (puede parar antes por early stopping)
    split_pct       : fracción de datos para entrenamiento
    patience        : evaluaciones sin mejora antes de early stopping

    Retorna
    -------
    Modelo PPO entrenado (también guardado en models/best_model_academic/)
    """
    print("=" * 60)
    print("ENTRENAMIENTO ACADÉMICO CON VALIDACIÓN TEMPORAL")
    print("=" * 60)

    df_f      = pd.read_csv(features_path, index_col=0)
    split_idx = int(len(df_f) * split_pct)

    train_env = PortfolioEnv(features_path, prices_path, end_idx=split_idx)
    eval_env  = PortfolioEnv(features_path, prices_path, start_idx=split_idx)

    # eval_freq adaptativo: evaluar cada ~20 episodios completos de entrenamiento.
    # Con dataset largo (split_idx=2000 steps/ep), eval_freq=10000 → solo 5 episodios
    # entre evaluaciones → señal de mejora ruidosa → early stopping prematuro.
    # Con dataset corto (split_idx=100 steps/ep), eval_freq=10000 → 100 episodios
    # entre evaluaciones → umbral de mejora relativo se vuelve muy pequeño → nunca para.
    # Solución: eval_freq = 20 episodios × longitud del episodio de train.
    ep_len_train = split_idx  # el entorno recorre todo el split en cada episodio
    eval_freq = max(5000, min(50000, ep_len_train * 20))

    # Patience adaptativa: más pasos totales → más evaluaciones antes de rendir.
    # Con total_timesteps=500000 y eval_freq adaptativo, habrá ~total/eval_freq evaluaciones.
    # Patience = 30% de esas evaluaciones, con mínimo 5 y máximo 15.
    n_evaluaciones_total = total_timesteps // eval_freq
    patience_efectiva = max(5, min(15, n_evaluaciones_total // 3))

    print(f"  Longitud episodio train: {ep_len_train} steps")
    print(f"  eval_freq adaptativo:    {eval_freq} steps (~20 episodios)")
    print(f"  Evaluaciones previstas:  {n_evaluaciones_total}")
    print(f"  Patience efectiva:       {patience_efectiva} evaluaciones sin mejora")

    # Callbacks
    monitor_cb  = AcademicMonitorCallback(verbose=0)
    overfit_cb  = OverfitDetectorCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_ep=3,
        patience=patience_efectiva,
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
    monitor_cb.guardar_reporte('src/reports/training_diagnostics.png')
    overfit_cb.guardar_curvas('src/reports/overfitting_analysis.png')

    model.save("models/ppo_academic_final")
    print("\nModelo final guardado en models/ppo_academic_final.zip")
    print("Mejor modelo (por eval reward) en models/best_model_academic/")

    return model
