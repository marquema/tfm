"""
Análisis académico del entrenamiento del agente PPO.

Este módulo es el "laboratorio" del TFM: contiene las herramientas para
entrenar el agente con monitorización de calidad, validar que su política
generaliza fuera de los datos vistos, y producir las gráficas que sustentan
las conclusiones de la memoria.

Cuatro bloques principales:

  1. Walk-Forward Validation — equivalente temporal de la validación cruzada.
     Entrena con un tramo histórico, valida en el siguiente, y desliza la
     ventana hacia adelante. Detecta si la política funciona bien solo en
     un periodo concreto (sobreajuste a un régimen de mercado).

  2. Expanding Window — variante donde el train empieza siempre en el día 0
     y crece. Simula mejor el caso real ("uso todo lo que sé hasta hoy para
     decidir mañana") a costa de mayor coste computacional.

  3. Detección de sobreajuste — compara reward en train vs evaluación
     out-of-sample paso a paso, y para el entrenamiento si el agente deja
     de mejorar (early stopping con paciencia).

  4. Diagnóstico de la dinámica de PPO — captura entropía, value loss,
     explained variance, KL aproximada y clip fraction durante el entrenamiento.
     Permite verificar que la red está aprendiendo y no colapsa ni oscila.

Nota sobre métricas clásicas de ML:
  - K-fold cross-validation y ROC/AUC no aplican a DRL sobre series temporales.
    K-fold mezclaría futuro y pasado (data leakage); ROC es para clasificación binaria.
  - Walk-forward es el estándar académico para modelos financieros secuenciales.
    Documentacion: Ver: López de Prado (2018), "Advances in Financial Machine Learning", cap. 7.

─────────────────────────────────────────────────────────────────────────────
Decisiones de diseño y experimentación pendiente (para la memoria del TFM)
─────────────────────────────────────────────────────────────────────────────

Función de recompensa actual:
    reward = Sharpe_rolling_20d - phi·MDD - gamma·Turnover

  Justificación: el Sharpe rolling de 20 días aporta una señal de calidad
  ajustada por riesgo (no solo retorno bruto), mientras que las penalizaciones
  por MDD y turnover desincentivan caídas grandes y operativa excesiva.
  phi y gamma se parametrizan vía perfil de riesgo (risk_profiles.py).

Alternativas estudiadas (no implementadas, candidatas a sección de trabajo
futuro en la memoria):

  - Ventana de Sharpe a 40 días en lugar de 20: la señal sería más suave
    (cada día pesa 1/40 en lugar de 1/20), lo que reduciría el clip fraction
    observado (~66 % → ~30-45 %) y bajaría el turnover, a cambio de mayor
    latencia para detectar cambios de régimen (10-15 días vs 5).

  - Reward = log_return en vez de Sharpe rolling: señal puramente diaria,
    sin oscilación por datos entrando/saliendo de la ventana. Esperaría
    clip fraction más bajo y explicación más facil, pero peor rendimiento
    ("retorno logarítmico penalizado por drawdown y turnover"). Coste:
    pierde la penalización implícita de la volatilidad que ofrece el
    Sharpe; habría que compensar con phi mayor.

# todo para la memoria.
Estado actual del modelo: R_eval = 0.617, explained variance = 0.979,
KL aproximada controlada (~0.049). Resultados sólidos; el clip fraction
elevado es justificable en la memoria. Cambiar la recompensa exige
reentrenar y re-validar todo el pipeline, por lo que se documenta como
experimento comparativo, no como sustitución del modelo de referencia.
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
    Captura métricas internas de PPO durante el entrenamiento para diagnóstico.

    SB3 (Stable-Baselines3) emite cada cierto número de pasos un resumen
    interno con varias señales con la evolución de la red. Este callback las
    recopila para luego pintarlas y poder verificar que el agente aprende
    de forma estable. Monitorización del entrenamiento.

    Métricas capturadas y cómo interpretarlas (todas se grafican en un
    panel en src/reports/training_diagnostics.png):

      - entropy_loss
            Mide cuánta "exploración aleatoria" sigue haciendo la política.
            Al principio es alta (el agente prueba acciones); debe decrecer
            poco a poco según se va decantando por las buenas. Un colapso
            brusco indica que el agente ha memorizado y deja de explorar
            (riesgo de sobreajuste). Sin descenso = no está aprendiendo.

      - value_loss
            Error de la "red de valor" — la parte del modelo que predice
            cuánto reward obtendrá el agente desde el estado actual hasta
            el final. Debe estabilizarse en un rango bajo. Si oscila o
            crece, la red de valor no consigue aprender la dinámica del
            entorno y el agente carece de una buena guía.

      - explained_var
            Qué fracción de la variación real de los retornos consigue
            anticipar la red de valor (R² del modelo de valor). Va de
            −∞ a 1. Objetivo > 0.5 ("entiende a grandes rasgos cómo
            evoluciona el valor de la cartera"). Valores < 0 significan
            que predecir la media constante sería mejor: red no funciona.

      - approx_kl
            Cuánto cambia la política en cada actualización (dirvergencia
            KL entre la política antigua y la nueva). PPO está diseñado
            para hacer cambios pequeños: > 0.05 indica saltos demasiado
            grandes, lo que puede dar entrenamientos inestables o que
            "olvidan" lo aprendido.

      - clip_fraction
            Porcentaje de actualizaciones que PPO recorta porque exceden
            su clip_range. Rango sano: 0.01-0.3. Si supera 0.3, el modelo
            quiere cambiar mucho y PPO está frenándolo casi siempre →
            considerar bajar learning_rate o subir clip_range. Si es < 0.01,
            la configuración es innecesariamente conservadora.

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
        Texto de diagnóstico automático basado en las métricas finales.

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

LEARNING_RATE=1e-4
CLIP_RANGE=0.1
# TODO (rigor académico): los CSVs de entrada llegan ya normalizados con z-score
# calculado sobre el split train global del dataset (~2018-2024). 
# z-score es una forma de "estandarizar" un número. La fórmula es: z = (valor - media) / desviación
# Para que el todas las features (RSI, MACD, volumen...) estén en una escala parecida y la red neuronal pueda compararlas.
# Para que walk-forward sea académicamente puro, cada ventana debería recalcular sus
# propios stats con el train de esa ventana — si no, hay un leakage sutil
# porque la escala de los features de cada ventana refleja info futura.
# En la práctica el efecto es pequeño (z-score en mercados financieros estables
# es bastante robusto) pero conviene mencionarlo en la memoria.

# Opción A (correcta): usar solo los datos de train de cada ventana. Si V3 entrena con 2020-2022, calculo media/desviación con esos años exclusivamente.
# TODO: alerta de dataleakage
# Opción B (lo que hacemos hoy): media/desviación calculadas una sola vez al principio, sobre el split train global del dataset entero (~2018-2024).
# Conclusión que saco para la memoria:
# En el TFM, el z-score se calcula sobre el split train global, no por ventana. Esto introduce un leakage sutil pero conviene reconocerlo. En la práctica el efecto es pequeño porque las medias/desviaciones de un activo financiero son razonablemente estables a lo largo de varios años (no cambian drásticamente entre 2018 y 2024). Para una versión académicamente impecable habría que recalcular stats por ventana — queda como mejora futura."
def walk_forward_validation(features_path: str,
                            prices_path: str,
                            train_days: int = 504,
                            test_days: int  = 252,
                            total_timesteps: int = 100000) -> pd.DataFrame:
    """
    Validación Walk-Forward con ventanas deslizantes de tamaño fijo.

    Idea intuitiva: como en finanzas el orden temporal importa, no podemos
    barajar los datos al azar (eso mezclaría futuro con pasado y se llama
    data leakage). En su lugar, dividimos el histórico en bloques
    consecutivos: entrenamos con dos años, validamos en el siguiente,
    deslizamos la ventana hacia adelante un año y repetimos. Cada ventana
    es un experimento independiente y completo.

    Pregunta del TFM que responde: "¿es el agente PPO igual de bueno en
    cualquier periodo de mercado, o solo casualmente acertó en el tramo
    de test del split 80/20?". Si el Sharpe es bueno en todas las
    ventanas, la política generaliza; si solo en una concreta, no.

    El número de ventanas lo determina el dataset, no es un parámetro:
    cuantos más años de histórico, más ventanas se generan automáticamente
    sin tocar el código.

    Criterios de tamaño de ventana:
      - train_days = 504 (2 años): mínimo para que PPO vea suficientes
        episodios y aprenda regímenes alcistas y bajistas.
      - test_days  = 252 (1 año): mínimo para que el Sharpe anualizado
        sea estadísticamente fiable (con < 252 días el factor √252
        amplifica el ruido y la métrica deja de ser comparable).

    Con el dataset actual (2018-2026, 8.2 años) genera 6 ventanas.

    Lectura de los resultados (df devuelto):
      - Sharpe medio > 0 en todas las ventanas → política generalmente rentable.
      - Alta varianza del Sharpe entre ventanas → política inestable,
        depende del régimen de mercado.
      - Degradación monotónica, que siempre va en al misma dirección, en las ventanas recientes → posible
        sobreajuste a la dinámica antigua del mercado.
        Ejemplo: El Sharpe baja sin parar según avanzamos en el tiempo: 1.4 → 1.3 → 1.1 → 0.8 → 0.5 → 0.2.
        ¿Qué significa? El agente aprendió patrones que funcionaban bien en mercados antiguos pero que ya no se dan hoy. 
        Está sobreajustado al pasado lejano. Una alarma para el TFM.
        Si en cambio el Sharpe oscilara (1.4, 0.8, 1.5, 0.7, 1.3, 0.9), eso sería simplemente "varianza alta"
        → política inestable, dependiente del régimen, pero no hay un patrón sistemático de 
        empeoramiento.

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
    # Borrar reportes anteriores para evitar mostrar datos incoherentes
    # si el walk-forward falla a mitad de ejecución.
    for old_file in ['src/reports/walk_forward_results.csv',
                     'src/reports/walk_forward_analysis.png']:
        if os.path.exists(old_file):
            os.remove(old_file)

    df_f = pd.read_csv(features_path, index_col=0)
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

    # Calcular ventanas: el número depende del dato, no del parámetro
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

        # Entrenamiento dentro de la ventana.
        # Hiperparámetros PPO unificados con la configuración de producción y
        # del análisis de sensibilidad (n_steps=2048, batch_size=128,
        # vf_coef=0.5, max_grad_norm=0.5) para garantizar coherencia entre
        # validación y modelo final entregado.
        train_env = PortfolioEnv(features_path, prices_path,
                                 start_idx=start, end_idx=split)
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=LEARNING_RATE,
            n_steps=2048,
            batch_size=128,
            clip_range=CLIP_RANGE,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
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
        window_metrics['train_start'] = date_start
        window_metrics['train_end']   = date_split
        window_metrics['test_start']  = date_split
        window_metrics['test_end']    = date_end
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
    n_windows = len(df)
    has_dates = 'train_start' in df.columns and 'test_end' in df.columns

    # Layout: 3 gráficas arriba + tabla de ventanas debajo
    # La altura crece con el número de ventanas para que la tabla sea legible
    table_height_ratio = max(1.5, n_windows * 0.3)
    fig_height = 5 + table_height_ratio
    fig = plt.figure(figsize=(max(15, n_windows * 1.8), fig_height))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, table_height_ratio],
                          hspace=0.4)

    x_pos = list(range(n_windows))
    x_labels = [f"V{i+1}" for i in range(n_windows)]

    # ─── Fila 1: 3 gráficas de barras ────────────────────────────────────────
    for col_idx, (col, title, color, threshold) in enumerate([
        ('Sharpe Ratio',      'Sharpe Ratio',       'steelblue',      0.0),
        ('Retorno Total (%)', 'Retorno Total (%)',   'mediumseagreen', 0.0),
        ('Max Drawdown (%)',  'Max Drawdown (%)',    'tomato',         None),
    ]):
        ax = fig.add_subplot(gs[0, col_idx])
        values = df[col].values
        bars = ax.bar(x_pos, values, color=color, alpha=0.75, edgecolor='white')

        for bar, val in zip(bars, values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y,
                    f'{val:.2f}', ha='center', va='bottom' if y >= 0 else 'top',
                    fontsize=7 if n_windows > 8 else 8, fontweight='bold', color=color)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8 if n_windows <= 10 else 6)
        if threshold is not None:
            ax.axhline(threshold, color='black', linestyle='--', alpha=0.5, linewidth=1)
        mean_val = np.mean(values)
        ax.axhline(mean_val, color=color, linestyle='-', alpha=0.9,
                   linewidth=2, label=f'Media: {mean_val:.2f}')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    # ─── Fila 2: tabla de ventanas con períodos ──────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')

    if has_dates:
        table_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            table_data.append([
                f"V{i+1}",
                f"{row.get('train_start', '?')}",
                f"{row.get('train_end', '?')}",
                f"{int(row.get('dias_train', 0))}d",
                f"{row.get('test_start', '?')}",
                f"{row.get('test_end', '?')}",
                f"{int(row.get('dias_test', 0))}d",
                f"{row.get('Sharpe Ratio', 0):.3f}",
                f"{row.get('Retorno Total (%)', 0):.1f}%",
                f"{row.get('Max Drawdown (%)', 0):.1f}%",
            ])

        col_labels = ['', 'Train inicio', 'Train fin', 'Dias',
                       'Test inicio', 'Test fin', 'Dias',
                       'Sharpe', 'Retorno', 'MDD']

        table = ax_table.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8 if n_windows <= 8 else 7)
        table.scale(1, 1.3)

        # Estilo de la tabla
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor('#cccccc')
            if row_idx == 0:  # Header
                cell.set_facecolor('#4a4a6a')
                cell.set_text_props(color='white', fontweight='bold')
            else:
                cell.set_facecolor('#f8f8f8' if row_idx % 2 == 0 else '#ffffff')

        ax_table.set_title('Detalle de ventanas', fontsize=10, fontweight='bold', pad=10)
    else:
        ax_table.text(0.5, 0.5, 'Sin información de fechas disponible',
                      ha='center', va='center', fontsize=10, color='gray')

    fig.suptitle('Walk-Forward Validation — Estabilidad de la Política',
                 fontsize=14, fontweight='bold')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Gráfica walk-forward guardada en {path}")


# ---------------------------------------------------------------------------
# Expanding Window Validation
# ---------------------------------------------------------------------------

def expanding_window_validation(features_path: str,
                                 prices_path: str,
                                 min_train_days: int = 504,
                                 test_days: int = 63,
                                 total_timesteps: int = 100000) -> pd.DataFrame:
    """
    Validación Expanding Window: el conjunto de entrenamiento crece en cada ventana.

    Idea intuitiva: a diferencia del walk-forward rolling (donde la ventana
    de train tiene tamaño fijo y se desliza), aquí el train empieza siempre
    en el día 0 y va acumulando historia. Cada nueva ventana entrena con
    TODO lo conocido hasta el momento y prueba en los siguientes meses.

    Pregunta del TFM que responde: "¿el agente mejora cuando le doy más
    historia, o llega un punto en que más datos no ayudan?". Es la
    metodología que más se parece a un sistema en producción real, donde
    cada noche reentrenaría con todo el histórico que tengo.

    Ejemplo con min_train=2 años, test=3 meses:
      V1: Train [2019-01 → 2021-01] (2.0 años)  → Test [2021-01 → 2021-04]
      V2: Train [2019-01 → 2021-04] (2.25 años) → Test [2021-04 → 2021-07]
      V3: Train [2019-01 → 2021-07] (2.5 años)  → Test [2021-07 → 2021-10]
      ...

    Ventajas sobre rolling window:
      - Simula producción real ("uso todo lo que sé hasta hoy para predecir mañana").
      - El modelo ve más regímenes de mercado conforme avanza.
      - Genera más ventanas (~12 con 5 años vs ~3-4 con rolling de 1 año),
        lo que da más puntos de evaluación y robustez estadística.

    Desventajas:
      - Las últimas ventanas tardan más en entrenar (más datos por procesar).
      - Asume implícitamente que los datos antiguos siguen siendo relevantes
        — si el régimen de mercado cambia mucho, el rolling puede ser preferible.

    Parameters
    ----------
    features_path: ruta al CSV de features normalizadas
    prices_path: ruta al CSV de precios originales
    min_train_days  : días mínimos de train para la primera ventana (504 = 2 años)
    test_days: días de test por ventana (63 = 3 meses, como sugiere Rubén)
    total_timesteps : pasos de entrenamiento PPO por ventana

    Returns
    -------
    pd.DataFrame con métricas por ventana (Sharpe, Retorno, MDD, fechas de train/test)

    References
    ----------
    López de Prado (2018), "Advances in Financial Machine Learning", cap. 7.
    Recomendación del tutor del TFM: 2 años iniciales + test de 3 meses expandiendo.
    """
    # Borrar reportes anteriores
    for old_file in ['src/reports/expanding_window_results.csv',
                     'src/reports/expanding_window_analysis.png']:
        if os.path.exists(old_file):
            os.remove(old_file)

    df_f    = pd.read_csv(features_path, index_col=0)
    n_total = len(df_f)

    if n_total < min_train_days + test_days:
        raise ValueError(
            f"Dataset demasiado pequeño ({n_total} dias). "
            f"Necesita al menos {min_train_days + test_days} dias para expanding window."
        )

    # Calcular ventanas: train empieza siempre en 0 y crece
    windows = []
    split = min_train_days
    while split + test_days <= n_total:
        windows.append((0, split, split + test_days))
        split += test_days  # avanza el split por cada período de test

    n_windows = len(windows)

    results = []
    print(f"\n{'='*60}")
    print(f"EXPANDING WINDOW VALIDATION")
    print(f"Dataset: {n_total} dias ({n_total/252:.1f} anios) | "
          f"Train minimo: {min_train_days}d ({min_train_days/252:.1f}a) | "
          f"Test: {test_days}d ({test_days/252:.1f}a)")
    print(f"Ventanas calculadas: {n_windows}")
    print(f"{'='*60}")

    for i, (start, split_idx, end) in enumerate(windows):
        date_start = df_f.index[start][:10]
        date_split = df_f.index[split_idx][:10]
        date_end   = df_f.index[end - 1][:10]
        train_size = split_idx - start

        print(f"\n[Ventana {i+1}/{n_windows}] "
              f"Train: {date_start} -> {date_split} ({train_size}d, {train_size/252:.1f}a) | "
              f"Test: {date_split} -> {date_end} ({end-split_idx}d)")

        # Entrenamiento con toda la historia hasta split_idx.
        # Hiperparámetros PPO unificados con la configuración de producción y
        # del análisis de sensibilidad (n_steps=2048, batch_size=128,
        # vf_coef=0.5, max_grad_norm=0.5) para garantizar coherencia entre
        # validación y modelo final entregado.
        train_env = PortfolioEnv(features_path, prices_path,
                                 start_idx=start, end_idx=split_idx)
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=LEARNING_RATE,
            n_steps=2048,
            batch_size=128,
            clip_range=CLIP_RANGE,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0
        )
        model.learn(total_timesteps=total_timesteps)

        # Evaluación out-of-sample
        test_env = PortfolioEnv(features_path, prices_path,
                                start_idx=split_idx, end_idx=end)
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
        window_metrics['dias_train'] = train_size
        window_metrics['dias_test']  = end - split_idx
        window_metrics['train_start'] = date_start
        window_metrics['train_end']   = date_split
        window_metrics['test_start']  = date_split
        window_metrics['test_end']    = date_end
        results.append(window_metrics)

        print(f"  Sharpe: {window_metrics['Sharpe Ratio']:.3f} | "
              f"Retorno: {window_metrics['Retorno Total (%)']:.1f}% | "
              f"MDD: {window_metrics['Max Drawdown (%)']:.1f}%")

    df_ew = pd.DataFrame(results).set_index('ventana')

    print(f"\n{'='*60}")
    print("RESUMEN EXPANDING WINDOW")
    print(f"  Sharpe medio:    {df_ew['Sharpe Ratio'].mean():.3f}  "
          f"(+-{df_ew['Sharpe Ratio'].std():.3f})")
    print(f"  Retorno medio:   {df_ew['Retorno Total (%)'].mean():.1f}%")
    print(f"  MDD medio:       {df_ew['Max Drawdown (%)'].mean():.1f}%")
    print(f"  Ventanas con Sharpe > 0: "
          f"{(df_ew['Sharpe Ratio'] > 0).sum()} / {n_windows}")
    print(f"{'='*60}")

    # Guardar resultados
    os.makedirs('src/reports', exist_ok=True)
    df_ew.to_csv('src/reports/expanding_window_results.csv')
    _plot_walk_forward(df_ew, path='src/reports/expanding_window_analysis.png')

    return df_ew


##############################################
# Detección de sobreajuste: gap train vs eval
# ############################################

class OverfitDetectorCallback(BaseCallback):
    """
    Detecta sobreajuste y aplica early stopping comparando train vs evaluación.

    Idea intuitiva: durante el entrenamiento, el reward en datos de train
    siempre tiende a subir (la red puede acabar memorizando). El que importa
    es el reward en datos out-of-sample (eval), que el agente nunca ve durante
    el aprendizaje. Si el de eval deja de mejorar mientras el de train sigue
    subiendo → el agente está sobreajustando: aprende patrones que solo
    funcionan en el pasado conocido.

    Acciones que toma este callback:
      1. Cada `eval_freq` pasos, ejecuta backtests sobre el conjunto de eval
         y calcula su reward medio por step.
      2. Si reward_eval no mejora en `patience` evaluaciones consecutivas,
         detiene el entrenamiento (early stopping).
      3. Si el gap relativo (train - eval) supera el 50%, lo logea como
         alerta de sobreajuste — el modelo aprende cosas que no generalizan.
      4. Almacena el historial completo para pintar luego las curvas
         train/eval en src/reports/overfitting_analysis.png.

    Parameters
    ----------
    eval_env : gym.Env
        Entorno de validación (distinto del de entrenamiento, debe usar el
        20 % de datos out-of-sample que el agente nunca verá durante el train).
    eval_freq : int
        Frecuencia de evaluación en pasos del modelo (no en episodios).
    n_eval_ep : int
        Número de episodios de evaluación que se promedian. Con política
        determinista y entorno sin aleatoriedad, valores > 1 son redundantes
        en este TFM, pero se mantiene la convención de SB3 por robustez ante futuras introducciones de ruido.
        
        Ampliación: simulamos 3 veces el mismo backtest y promediamos. Hoy las 3 ejecuciones dan 
        resultado idéntico porque:
            La política es determinista: ante la misma observación, siempre devuelve la misma acción.
            El entorno es determinista: los precios reales del histórico no cambian. El día 15 enero 2024 IVV cerró a $480, 
                sea la primera vez que lo simulemos o la décima. → Las 3 ejecuciones son fotocopias entre sí. 
                Hacer 1 sería suficiente y triplicaría la velocidad.

        ¿Por qué entonces lo dejamos en 3? Porque podríamos meter aleatoriedad en el futuro. 
        Por ejemplo:   Dropout en la red: técnica que apaga neuronas al azar para evitar sobreajuste. 
            Si lo activamos, la política deja de ser determinista — la misma observación da acciones ligeramente distintas cada vez.
            Slippage estocástico: simular que cuando compras IVV no consigo exactamente el precio teórico, 
            sino uno con un pequeño desliz aleatorio (lo realista en un broker real). El entorno deja de ser determinista.
        
        Si introducimos cualquiera de esas dos cosas, las 3 ejecuciones empezarían a dar resultados ligeramente distintos, 
        y promediar 3 sería estadísticamente más fiable que confiar en una sola.

        Conclusión: dejamos N=3 por si las moscas por si mañana metemos aleatoriedad. 
        Hoy es redundante pero no rompe nada.
        todo: debería contemplar esta posibilidad no determinista? Verlo con Rubén.

    patience : int
        Evaluaciones consecutivas sin mejora antes de aplicar early stopping.
    min_improvement_pct : float
        OBSOLETO — se mantiene por compatibilidad con la firma antigua, pero
        actualmente el callback usa un umbral absoluto fijo (0.001) en lugar
        de uno relativo. Ver justificación en _on_step().
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
        # Conservado por compatibilidad de firma; el código no lo usa porque
        # el umbral relativo daba problemas con rewards cercanos a cero.
        # El umbral efectivo (absoluto, 0.001) está en _on_step().
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

        # ── Reward por step en train ────────────────────────────────────────
        # SB3 guarda en ep_info_buffer un resumen de los últimos episodios
        # ejecutados durante el entrenamiento (campo 'r' = reward total del
        # episodio, 'l' = longitud en pasos).
        #
        # Dividimos reward total entre longitud para obtener "reward medio
        # por paso". Esto es importante porque un dataset largo da episodios
        # más largos, que acumularían más reward sin que el agente sea
        # mejor — al normalizar por step, podemos comparar directamente con
        # reward_eval, que también se calcula por step.
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            reward_train = np.mean([
                ep['r'] / max(ep['l'], 1) for ep in self.model.ep_info_buffer
            ])
        else:
            reward_train = np.nan

        # ── Reward por step en eval (out-of-sample) 
        # Ejecuta `n_eval_ep` episodios completos sobre el conjunto de
        # evaluación (el 20% de datos que el agente NUNCA vio durante el
        # entrenamiento) y promedia el reward medio por paso de los N.
        #
        # Un "episodio" aquí = un backtest entero del periodo de eval, desde
        # el primer día hasta el último (`done = True`).
        #
        # Por qué N=3 con `deterministic=True`: en este setup, política
        # determinista + entorno sin aleatoriedad → los 3 episodios dan el
        # mismo resultado. Mantenemos la convención de SB3 (más robusta si en
        # el futuro se introduce ruido en política/entorno: dropout, slippage
        # estocástico, etc.).
        #
        # Esta es la métrica real de generalización: si crece aquí (no solo
        # en train), el agente está aprendiendo patrones, no memorizando.
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

        # ── Early stopping con paciencia 
        # Si tras `patience` evaluaciones consecutivas no hay mejora en eval,
        # paramos el entrenamiento (el modelo dejó de aprender; seguir solo
        # agrava el sobreajuste y consume tiempo).
        #
        # Decisión de diseño — umbral ABSOLUTO (0.001) vs RELATIVO (% del mejor):
        #   Un umbral relativo (ej. "mejora ≥ 1% del best_eval") tiene un fallo
        #   conocido cuando el mejor valor cruza cero o es negativo. Si
        #   `best_eval = -0.4`, un 1% es −0.404 (razonable); pero si
        #   `best_eval = -0.001`, casi cualquier ruido cuenta como mejora y el
        #   entrenamiento no para nunca. Con valores positivos altos, exige
        #   mejoras irreales.
        #
        #   El umbral absoluto resuelve esto: 0.001 representa siempre la misma
        #   cantidad de progreso real. Como el reward está en una escala
        #   estable ~[-0.5, 0.5] (es reward por step, no acumulado), 0.001
        #   equivale a ~0.2% del rango total — suficiente para filtrar
        #   fluctuación estadística sin perder mejoras reales.
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
        ax1.set_ylabel('Reward medio por step')
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


# # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Función principal: entrenamiento académico completo
# # # # # # # # # # # # # # # # # # # # # # # # # # # 

def train_academic(features_path: str = 'data/normalized_features.csv',
                   prices_path: str   = 'data/original_prices.csv',
                   total_timesteps: int = 500000,
                   split_pct: float = 0.8,
                   patience: int = 8,
                   risk_profile: str = 'low_turnover') -> PPO:
    """
    Entrenamiento académico completo del agente PPO con todos los controles
    de calidad activos. Es la función "entrar a producir el modelo del TFM".

    Qué hace, paso a paso:
      1. Resuelve el perfil de riesgo elegido (phi, gamma) y crea dos
         entornos: uno de train (80 % de los datos) y otro de eval (20 %
         restante, out-of-sample).
      2. Calcula automáticamente eval_freq y patience adaptativos para que
         las evaluaciones lleguen con la frecuencia adecuada al tamaño del
         dataset (ver explicación más abajo).
      3. Engancha tres callbacks: monitor académico (diagnóstico interno
         PPO), detector de sobreajuste (gap train vs eval + early stopping)
         y EvalCallback (guarda automáticamente el mejor modelo según el
         reward de eval, en models/best_model_academic/).
      4. Entrena PPO con hiperparámetros calibrados para series financieras
         (lr=1e-4, n_steps=2048, batch_size=128, clip_range=0.1, red 256×256).
      5. Genera dos PNG de diagnóstico: uno con la dinámica interna PPO
         (entropía, value loss, KL...) y otro con las curvas train/eval
         que evidencian si hubo sobreajuste.

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas (entrada a la red).
    prices_path : str
        Ruta al CSV de precios originales (para calcular el reward real).
    total_timesteps : int
        Pasos máximos de entrenamiento. El early stopping puede detenerlo
        antes si el agente deja de mejorar en eval.
    split_pct : float
        Fracción de datos para train. El resto queda como out-of-sample
        para evaluar el modelo durante y después del entrenamiento.
    patience : int
        Valor base de paciencia para early stopping. La función calcula
        además una `effective_patience` adaptativa al número total de
        evaluaciones previstas (entre 5 y 15).
    risk_profile : str
        Perfil que determina phi y gamma de la recompensa:
          - 'low_turnover' → DEFAULT y PERFIL PRINCIPAL DEL TFM. Mejor
                             Sharpe en el análisis de sensibilidad.
          - 'balanced'     → equilibrado (alternativa conservadora)
          - 'conservative' → mayor penalización por drawdown
          - 'aggressive'   → mínimas penalizaciones, máxima libertad
        Ver risk_profiles.py para los valores exactos de cada uno.

    Returns
    -------
    PPO
        Modelo PPO entrenado. Se guardan dos copias en disco:
          - models/best_model_academic/best_model.zip → el mejor modelo
            visto en eval durante el entrenamiento (el que se usa en
            simulaciones).
          - models/ppo_academic_final.zip → el modelo en el último step
            (no necesariamente el mejor, útil para depuración).
    """
    from src.training_drl.risk_profiles import get_profile, get_phi_gamma

    # Resolver perfil de riesgo
    profile = get_profile(risk_profile)
    phi, gamma = get_phi_gamma(risk_profile)

    # Borrar reportes anteriores para evitar mostrar datos incoherentes
    for old_file in ['src/reports/training_diagnostics.png',
                     'src/reports/overfitting_analysis.png']:
        if os.path.exists(old_file):
            os.remove(old_file)

    print("=" * 60)
    print("ENTRENAMIENTO ACADÉMICO CON VALIDACIÓN TEMPORAL")
    print(f"  Perfil de riesgo: {profile['name']} ({risk_profile})")
    print(f"  phi={phi}, gamma={gamma}")
    print("=" * 60)

    df_f = pd.read_csv(features_path, index_col=0)
    split_idx = int(len(df_f) * split_pct)

    train_env = PortfolioEnv(features_path, prices_path,
                             end_idx=split_idx, phi=phi, gamma=gamma)
    eval_env  = PortfolioEnv(features_path, prices_path,
                             start_idx=split_idx, phi=phi, gamma=gamma)

    # ── eval_freq adaptativo ──────────────────────────────────────────────
    # ¿Cada cuántos pasos del modelo evaluamos en out-of-sample? Si
    # evaluamos demasiado seguido, la señal "estoy mejorando" es ruidosa
    # (no le ha dado tiempo a aprender nada nuevo entre dos evaluaciones)
    # y el early stopping dispara antes de tiempo. Si evaluamos muy poco,
    # el modelo se sobreajusta antes de que lo detectemos.
    #
    # Regla: una evaluación cada ~20 episodios completos de train. Como en
    # este entorno cada episodio recorre todo el split, ep_len_train = split_idx.
    # Con un dataset pequeñajo (split=100) → eval cada 5 000 pasos; con un dataset
    # grande (split=2 000) → eval cada 40 000 pasos. Acotado entre 5 k y 50 k.
    ep_len_train = split_idx  # el entorno recorre todo el split en cada episodio
    eval_freq = max(5000, min(50000, ep_len_train * 20))

    # ── Patience adaptativa ───────────────────────────────────────────────
    # Cuántas evaluaciones consecutivas sin mejora aceptamos antes de parar.
    # Si entrenamos pocos pasos en total, no podemos exigir 15 evaluaciones
    # de paciencia (solo va a haber 8 evaluaciones en total). Si entrenamos
    # 500 k pasos, sí podemos permitirnos esperar más antes de rendirnos.
    #
    # Regla: 30 % del número total de evaluaciones previstas, acotado entre
    # 5 (paciencia mínima razonable) y 15 (más sería ineficiente).
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

    # ── Hiperparámetros del PPO ───────────────────────────────────────────
    # Calibrados para el entorno financiero del TFM tras experimentación:
    # los valores por defecto de SB3 (lr=3e-4, n_steps=1024, batch=64,
    # clip=0.2) producían entrenamientos inestables (KL alta, gradientes
    # ruidosos) en series temporales con mucho ruido como las de mercado.
    # Los valores actuales priorizan estabilidad sobre velocidad.
    
    # explicación detallada. Va a ir de cabeza al tfm
    # Agente PPO con red MLP (perceptrón multicapa) sobre el entorno de entrenamiento.
    # learning_rate=1e-4
        #Cuánto se mueve la red en cada actualización. Es como caminar hacia la cima de una montaña: 
        # pasos grandes (3e-4 = lr alto) llegas rápido pero te puedes saltar la cima; pasos pequeños 
        # (1e-4) tardas más pero aciertas mejor. Bajamos de 3e-4 a 1e-4 porque las series financieras 
        # tienen mucho ruido y los pasos grandes hacían oscilar el modelo.
    #n_steps=2048
        #Cuánta experiencia recoge antes de aprender. PPO funciona en ciclos: vive 2048 pasos en el 
        # entorno (toma decisiones, ve consecuencias) y solo entonces actualiza la red con todo lo 
        # aprendido. Subido de 1024 a 2048 = "vive el doble antes de reflexionar" → reflexión más 
        # informada.
    # batch_size=128
        #De cuántos ejemplos toma nota a la vez al actualizar. Si los 2048 pasos son la "experiencia 
        # total", el batch_size es el "trozo que le da a la red de cada vez". Subido de 64 a 128 
        # = gradientes más promediados, menos ruidosos.

    # clip_range=0.1
        # Cuánto puede cambiar la política como máximo en una actualización. PPO previene saltos 
        # bruscos: si la nueva política se quiere alejar más del 10% de la antigua, recorta. Bajado de 
        # 0.2 a 0.1 = más conservador → entrenamiento más estable, menos riesgo de "olvidar" 
        # lo aprendido.
    # ent_coef=0.01
        #Premio por explorar. Penaliza que el agente se vuelva demasiado seguro (siempre la misma acción)
        # . 0.01 = pequeño empujón a probar acciones distintas. Si fuera 0, el agente se cerraría 
        # enseguida en una estrategia y dejaría de buscar mejoras.

    # vf_coef=0.5
        # Cuánto pesa aprender el "modelo de valor" (la red que predice "cuánto reward voy a sacar 
        # desde aquí") respecto a aprender la política directamente. 0.5 = mitad y mitad, balance 
        # estándar.

    # max_grad_norm=0.5
        # Tope al tamaño de los gradientes. Si en una actualización los gradientes (las correcciones a 
        # la red) son enormes — porque hubo un día catastrófico — los recortamos a 0.5. Evita que un 
        # día anómalo destruya lo aprendido en semanas.

    # policy_kwargs=dict(net_arch=[256, 256])
        # Forma de la red neuronal: 2 capas ocultas con 256 neuronas cada una. Suficiente capacidad para
        # aprender patrones financieros sin caer en sobreajuste por exceso de parámetros.

    # verbose=1
        # Verbosidad = "cuánto habla". 0 = silencioso (no imprime nada en consola). 1 = imprime un 
        # resumen en cada actualización (rewards, losses, KL, etc.). 2 = aún más detallado. Como 
        # queremos ver el progreso del entrenamiento en consola, usamos 1.


    #tensorboard_log="./logs/"
        # Dónde guardar logs de entrenamiento para visualizarlos con TensorBoard (una herramienta 
        # gráfica de TensorFlow/PyTorch que pinta curvas de entrenamiento en tiempo real). No es 
        # imprescindible, pero útil para diagnóstico durante el desarrollo.



    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,  # 1e-4 — actualizaciones más graduales (defecto: 3e-4)
        n_steps=2048,# más experiencia entre updates (defecto: 1024)
        batch_size=128,# gradientes más estables (defecto: 64)
        clip_range=CLIP_RANGE, # 0.1 — frena cambios bruscos de política (defecto: 0.2)
        ent_coef=0.01,  # exploración: peso de la entropía en la pérdida
        vf_coef=0.5,# peso de la value loss en la pérdida total
        max_grad_norm=0.5,  # clipping de gradientes para evitar explosiones
        policy_kwargs=dict(net_arch=[256, 256]),  # MLP, perceptron multicapa, de 2 capas ocultas
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
