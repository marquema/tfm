"""
Dashboard interactivo del TFM — AI-Driven Portfolio Management.

Muestra:
  - Backtest comparativo: PPO vs Equal Weight, 60/40, Buy & Hold, Markowitz
  - Métricas académicas: Sharpe, Sortino, MDD, CAGR, Volatilidad
  - Asset allocation de la IA (pie chart interactivo)
  - Diagnóstico del entrenamiento con explicaciones académicas
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from stable_baselines3 import PPO
from src.environment_trading import PortfolioEnv

try:
    from src.benchmarking.baselines import ejecutar_baselines, calcular_metricas, tabla_comparativa
except ImportError:
    from src.benchmarking.baselines import ejecutar_baselines, calcular_metricas, tabla_comparativa


# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="TFM — AI Portfolio Manager",
    page_icon="📈",
    layout="wide"
)

st.title("AI-Driven Portfolio Management — TFM Dashboard")
st.markdown(
    "Evaluación del agente de Deep Reinforcement Learning (PPO) frente a estrategias "
    "clásicas de gestión de carteras. Todos los resultados se calculan sobre el **período "
    "de test out-of-sample** — datos que el agente nunca vio durante el entrenamiento."
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Configuración")

modelo_path = st.sidebar.selectbox(
    "Modelo a evaluar",
    options=[
        "models/best_model_academic/best_model.zip",
        "models/best_model/best_model.zip",
        "models/ppo_academic_final.zip",
    ],
    index=0
)
commission  = st.sidebar.slider("Comisión por operación (%)", 0.0, 0.5, 0.1) / 100
initial_bal = st.sidebar.number_input("Capital inicial ($)", value=10000, step=1000)
split_pct   = st.sidebar.slider("Split train/test (%)", 60, 90, 80) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado del sistema**")
features_ok = os.path.exists('data/normalized_features.csv')
prices_ok   = os.path.exists('data/original_prices.csv')
modelo_ok   = os.path.exists(modelo_path)
st.sidebar.markdown(f"{'✅' if features_ok else '❌'} Features CSV")
st.sidebar.markdown(f"{'✅' if prices_ok   else '❌'} Precios CSV")
st.sidebar.markdown(f"{'✅' if modelo_ok   else '❌'} Modelo PPO")


# ─── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv',     index_col=0)
    return df_f, df_p

if not (features_ok and prices_ok):
    st.error("Faltan archivos de datos. Ejecuta primero POST /fase1/preparar-datos.")
    st.stop()

df_f, df_p = load_data()
split_idx  = int(len(df_f) * split_pct)
df_p_test  = df_p.iloc[split_idx:].copy()
tickers    = df_p.columns.tolist()

st.info(
    f"Período de test: **{df_p_test.index[0][:10]}** → **{df_p_test.index[-1][:10]}**"
    f" | {len(df_p_test)} días de trading | {len(tickers)} activos"
)


# ─── Constantes visuales ──────────────────────────────────────────────────────
COLORES = {
    'IA_PPO':               '#00d4ff',
    'Equal_Weight_Mensual': '#f0a500',
    'Buy_and_Hold':         '#7ed957',
    'Cartera_60_40':        '#ff6b6b',
    'Markowitz_MV':         '#c77dff',
}

NOMBRES = {
    'IA_PPO':               'IA PPO',
    'Equal_Weight_Mensual': 'Equal Weight',
    'Buy_and_Hold':         'Buy & Hold',
    'Cartera_60_40':        'Cartera 60/40',
    'Markowitz_MV':         'Markowitz MV',
}

DESCRIPCIONES_METRICAS = {
    'Retorno Total (%)':          'Ganancia o pérdida total sobre el capital inicial durante todo el período de test.',
    'CAGR (%)':                   'Tasa de crecimiento anual compuesta. Normaliza el retorno por el tiempo transcurrido.',
    'Volatilidad Anualizada (%)': 'Desviación estándar de los retornos diarios, anualizada con √252. Mide el riesgo total.',
    'Sharpe Ratio':               'Retorno ajustado por riesgo total: (retorno − tasa libre de riesgo) / volatilidad. Cuanto mayor, mejor.',
    'Sortino Ratio':              'Como el Sharpe pero solo penaliza la volatilidad negativa (caídas). Más justo con estrategias asimétricas.',
    'Max Drawdown (%)':           'Caída máxima desde un máximo histórico hasta el siguiente mínimo. Mide el peor escenario vivido.',
    'Valor Final ($)':            'Capital resultante al final del período de test partiendo del capital inicial configurado.',
}

LAYOUT_OSCURO = dict(template='plotly_dark', hovermode='x unified',
                     legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))


# ─── Backtest ────────────────────────────────────────────────────────────────
if st.button("▶  Ejecutar Backtest Completo", type="primary", use_container_width=True):

    if not modelo_ok:
        st.error(f"Modelo no encontrado en {modelo_path}. Entrena con POST /fase3/entrenar-academico")
        st.stop()

    with st.spinner("Simulando estrategias en datos de test..."):

        # ── Agente PPO ────────────────────────────────────────────────────────
        env_test = PortfolioEnv(
            'data/normalized_features.csv',
            'data/original_prices.csv',
            start_idx=split_idx,
            commission=commission,
            initial_balance=initial_bal
        )
        model = PPO.load(modelo_path)
        obs, _ = env_test.reset()
        done   = False
        equity_ppo      = [initial_bal]
        weights_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env_test.step(action)
            equity_ppo.append(info['value'])
            w = np.clip(action, 0, 1)
            w = w / (w.sum() + 1e-6)
            weights_history.append(w)

        serie_ppo    = pd.Series(equity_ppo, name='IA_PPO')

        # ── Baselines ─────────────────────────────────────────────────────────
        resultados_bl = ejecutar_baselines(
            df_p_test,
            initial_balance=initial_bal,
            commission=commission,
            ticker_rv='IVV_Close',
            ticker_rf='BND_Close'
        )

        todas_series = {'IA_PPO': serie_ppo, **resultados_bl}
        df_metricas  = tabla_comparativa(todas_series)

    fechas = list(range(max(len(s) for s in todas_series.values())))

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 1: Métricas comparativas
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 1. Métricas de Rendimiento")
    st.markdown(
        "Resumen cuantitativo de cada estrategia calculado sobre el período de test "
        "out-of-sample. Las métricas más relevantes para comparar gestores son el "
        "**Sharpe Ratio** (rentabilidad por unidad de riesgo total) y el "
        "**Max Drawdown** (peor caída posible desde un máximo). "
        "Un buen gestor tiene Sharpe > 1 y MDD controlado."
    )

    cols = st.columns(len(df_metricas))
    for col, nombre in zip(cols, df_metricas.index):
        sharpe  = df_metricas.loc[nombre, 'Sharpe Ratio']
        retorno = df_metricas.loc[nombre, 'Retorno Total (%)']
        mdd     = df_metricas.loc[nombre, 'Max Drawdown (%)']
        col.metric(
            label=NOMBRES.get(nombre, nombre),
            value=f"Sharpe {sharpe:.2f}",
            delta=f"Ret {retorno:.1f}%  |  MDD {mdd:.1f}%",
            delta_color="normal" if retorno >= 0 else "inverse"
        )

    st.markdown("### Tabla completa")
    with st.expander("ℹ️  Qué significa cada métrica"):
        for metrica, desc in DESCRIPCIONES_METRICAS.items():
            st.markdown(f"- **{metrica}**: {desc}")

    def highlight_ppo(row):
        return ['background-color: #1e3a5f; color: white; font-weight: bold'
                if row.name == 'IA_PPO' else '' for _ in row]

    st.dataframe(
        df_metricas.style.apply(highlight_ppo, axis=1).format("{:.2f}"),
        use_container_width=True
    )

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 2: Equity Curves
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 2. Evolución del Capital (Equity Curves)")
    st.markdown(
        "Muestra cómo crece o decrece el capital invertido en cada estrategia "
        "a lo largo del período de test. La línea discontinua marca el capital "
        "inicial — todo lo que quede por encima es ganancia. "
        "**Haz clic en la leyenda** para activar o desactivar estrategias. "
        "**Doble clic** sobre una para aislarla. Puedes hacer zoom arrastrando."
    )

    fig_eq = go.Figure()
    for nombre, serie in todas_series.items():
        fig_eq.add_trace(go.Scatter(
            x=fechas[:len(serie)],
            y=serie.values,
            name=NOMBRES.get(nombre, nombre),
            line=dict(color=COLORES.get(nombre, '#aaa'), width=3 if nombre == 'IA_PPO' else 1.5),
            hovertemplate=f"<b>{NOMBRES.get(nombre, nombre)}</b><br>Día %{{x}}: $%{{y:,.2f}}<extra></extra>"
        ))
    fig_eq.add_hline(y=initial_bal, line_dash="dash", line_color="white",
                     opacity=0.3, annotation_text="Capital inicial", annotation_font_color="white")
    fig_eq.update_layout(**LAYOUT_OSCURO, xaxis_title="Días de trading", yaxis_title="Valor ($)", height=450)
    st.plotly_chart(fig_eq, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3: Drawdown
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 3. Drawdown")
    st.markdown(
        "El **drawdown** mide la caída desde el máximo histórico alcanzado hasta ese momento. "
        "Un valor de −20% significa que la cartera vale un 20% menos que en su mejor punto "
        "anterior. Es la métrica que más impacta al inversor real: representa el peor "
        "escenario que habrías vivido si hubieras invertido justo en el pico. "
        "Cuanto más cerca de 0 se mantenga la línea, más estable es la estrategia. "
        "**Haz clic en la leyenda** para comparar pares de estrategias."
    )

    fig_dd = go.Figure()
    for nombre, serie in todas_series.items():
        rolling_max = serie.cummax()
        dd = (serie - rolling_max) / (rolling_max + 1e-8) * 100
        fig_dd.add_trace(go.Scatter(
            x=fechas[:len(dd)],
            y=dd.values,
            name=NOMBRES.get(nombre, nombre),
            line=dict(color=COLORES.get(nombre, '#aaa'), width=1.5),
            fill='tozeroy',
            hovertemplate=f"<b>{NOMBRES.get(nombre, nombre)}</b><br>Día %{{x}}: %{{y:.2f}}%<extra></extra>"
        ))
    fig_dd.update_layout(**LAYOUT_OSCURO, xaxis_title="Días de trading",
                         yaxis_title="Drawdown (%)", yaxis_ticksuffix="%", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4: Asset Allocation
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 4. Asset Allocation del Agente PPO")

    col_pie, col_evol = st.columns(2)

    with col_pie:
        st.markdown("### Pesos finales")
        st.markdown(
            "Distribución de la cartera en el **último paso del período de test**. "
            "Representa qué compraría el agente si operase hoy con los datos disponibles. "
            "**Haz clic** en un activo de la leyenda para ocultarlo y ver el resto con más detalle."
        )
        last_w = np.array(weights_history[-1]).flatten()
        if len(last_w) == len(tickers):
            fig_pie = go.Figure(go.Pie(
                labels=tickers, values=last_w, hole=0.35,
                hovertemplate="<b>%{label}</b><br>Peso: %{percent}<extra></extra>"
            ))
            fig_pie.update_layout(template='plotly_dark', height=380)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning(f"Dimensiones: modelo={len(last_w)}, CSV={len(tickers)}. Reentrenar.")

    with col_evol:
        st.markdown("### Evolución de pesos")
        st.markdown(
            "Cómo varía la asignación a cada activo a lo largo del test. "
            "Mucha variación entre pasos indica **alto turnover** (rotación de cartera), "
            "lo que genera costes de transacción. Una línea estable indica que el agente "
            "mantiene posiciones y opera con eficiencia. **Haz clic en la leyenda** para aislar activos."
        )
        if weights_history and len(weights_history[0]) == len(tickers):
            df_w = pd.DataFrame(weights_history, columns=tickers)
            fig_w = go.Figure()
            for ticker in tickers:
                fig_w.add_trace(go.Scatter(
                    x=list(range(len(df_w))), y=df_w[ticker].values,
                    name=ticker, stackgroup='one',
                    hovertemplate=f"<b>{ticker}</b>: %{{y:.1%}}<extra></extra>"
                ))
            fig_w.update_layout(**LAYOUT_OSCURO, xaxis_title="Días de trading",
                                yaxis_title="Peso", yaxis_tickformat=".0%", height=350)
            st.plotly_chart(fig_w, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 5: Diagnóstico del Entrenamiento
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 5. Diagnóstico del Entrenamiento")
    st.markdown(
        "Antes de confiar en los resultados del backtest, hay que verificar que el "
        "entrenamiento fue **correcto y sin sobreajuste**. Las tres gráficas siguientes "
        "son la evidencia académica de que el modelo aprendió de forma robusta."
    )

    with st.expander("ℹ️  Cómo interpretar cada gráfica"):
        st.markdown("""
**Gráfica 1 — Métricas internas del entrenamiento PPO**

Indicadores que miden la salud del proceso de aprendizaje paso a paso:

| Métrica | Qué mide | Objetivo |
|---|---|---|
| **Entropía de política** | Cuánta exploración mantiene el agente | Debe bajar gradualmente. Colapso rápido = memorización |
| **Value Function Loss** | Error al predecir retornos futuros | Debe estabilizarse. Si sube, la red es inestable |
| **Explained Variance** | Fracción de retornos futuros que la red predice | > 0.5. Si es negativa, el modelo es peor que la media |
| **KL Divergencia** | Cambio de política por actualización | < 0.05. Si supera, las actualizaciones son demasiado bruscas |
| **Clip Fraction** | Actualizaciones recortadas por PPO | Entre 0.01 y 0.3. Fuera de rango: ajustar clip_range |

---

**Gráfica 2 — Detección de sobreajuste**

Compara el reward en datos de **entrenamiento** vs datos de **evaluación** (nunca vistos):

- Ambas curvas suben juntas → el agente generaliza correctamente.
- Train sube pero eval no → **sobreajuste**: memorizó el pasado, no aprendió estructura.
- El Early Stopping guarda el modelo en el punto de máxima generalización antes de que sobreajuste.

---

**Gráfica 3 — Walk-Forward Validation**

Equivalente temporal del cross-validation. Se divide la historia en **6 ventanas**
(2 años de entrenamiento + 1 año de test cada una). El agente se reentrena desde cero
en cada ventana y se evalúa en el período siguiente que nunca vio:

- Sharpe estable entre ventanas → la política funciona en distintos regímenes de mercado.
- Alta varianza entre ventanas → rendimiento dependiente del período concreto (inestable).
- Sharpe decreciente en ventanas recientes → el modelo está sesgado hacia el pasado.
        """)

    col_r1, col_r2, col_r3 = st.columns(3)
    for col, ruta, titulo, desc in [
        (col_r1, 'src/reports/training_diagnostics.png',
         "Métricas internas PPO",
         "Entropía, value loss, explained variance, KL y clip fraction durante el entrenamiento."),
        (col_r2, 'src/reports/overfitting_analysis.png',
         "Sobreajuste Train vs Eval",
         "Gap entre reward en datos de train y datos de evaluación. Detecta memorización."),
        (col_r3, 'src/reports/walk_forward_analysis.png',
         "Walk-Forward Validation",
         "Sharpe, retorno y MDD del agente en 6 ventanas temporales independientes."),
    ]:
        col.markdown(f"**{titulo}**")
        col.caption(desc)
        if os.path.exists(ruta):
            col.image(ruta, use_container_width=True)
        else:
            col.info(f"Pendiente de generar.\n\n`{ruta}`")
