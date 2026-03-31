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

    # Construir eje de fechas: las series tienen 1 punto extra al inicio (balance inicial)
    # Se añade un día hábil anterior al test como "día 0" para ese punto.
    fechas_test = pd.to_datetime(df_p_test.index)
    from pandas.tseries.offsets import BDay
    fecha_d0 = fechas_test[0] - BDay(1)
    fechas   = [fecha_d0] + fechas_test.tolist()

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
        "Muestra el valor de la cartera día a día durante el período de test. "
        "Todas las estrategias parten del mismo capital inicial (línea discontinua). "
        "La estrategia que acaba más arriba obtuvo mayor rentabilidad total, pero fíjate "
        "también en el **camino**: una curva con grandes bajadas intermedias es mucho más "
        "difícil de seguir en la práctica que una curva suave aunque acabe en el mismo punto."
    )
    st.caption("Interacción: clic en la leyenda para mostrar/ocultar estrategias · doble clic para aislar una · arrastra para hacer zoom")

    fig_eq = go.Figure()
    for nombre, serie in todas_series.items():
        fig_eq.add_trace(go.Scatter(
            x=fechas[:len(serie)],
            y=serie.values,
            name=NOMBRES.get(nombre, nombre),
            line=dict(color=COLORES.get(nombre, '#aaa'), width=3 if nombre == 'IA_PPO' else 1.5),
            hovertemplate=f"<b>{NOMBRES.get(nombre, nombre)}</b><br>%{{x|%d %b %Y}}: $%{{y:,.2f}}<extra></extra>"
        ))
    fig_eq.add_hline(y=initial_bal, line_dash="dash", line_color="white",
                     opacity=0.3, annotation_text="Capital inicial", annotation_font_color="white")
    fig_eq.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha", yaxis_title="Valor ($)", height=450)
    st.plotly_chart(fig_eq, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 3: Drawdown
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 3. Drawdown — Peor Caída en Cada Momento")
    st.markdown(
        "Esta gráfica responde a la pregunta: **¿cuánto habría perdido un inversor si hubiese "
        "comprado en el peor momento?** Para cada día, muestra qué porcentaje ha caído la "
        "cartera respecto a su máximo histórico anterior."
    )
    st.markdown(
        "- Un valor de **−20%** significa que en ese momento la cartera valía un 20% menos que "
        "en su mejor punto previo.  \n"
        "- Las zonas sombreadas más profundas son las **crisis**: cuanto más tiempo permanece "
        "la curva alejada de 0, más tarda la estrategia en recuperarse.  \n"
        "- El **máximo drawdown** de la tabla anterior es el punto más bajo de esta gráfica."
    )
    st.caption("Interacción: clic en la leyenda para comparar dos estrategias en detalle")

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
            hovertemplate=f"<b>{NOMBRES.get(nombre, nombre)}</b><br>%{{x|%d %b %Y}}: %{{y:.2f}}%<extra></extra>"
        ))
    fig_dd.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                         yaxis_title="Drawdown (%)", yaxis_ticksuffix="%", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 4: Asset Allocation
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 4. Qué Compra el Agente — Asset Allocation")

    col_pie, col_evol = st.columns(2)

    with col_pie:
        st.markdown("### Cartera final")
        st.markdown(
            "Qué proporción del capital asigna el agente a cada activo al **final del período de test**. "
            "Si el agente ha aprendido bien, debería concentrarse en los activos que mejor se "
            "comportaron en ese período y reducir exposición a los volátiles o con peor rendimiento."
        )
        st.caption("Clic en la leyenda para ocultar activos y ver los demás con más detalle")
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
        st.markdown("### Cómo cambia la cartera en el tiempo")
        st.markdown(
            "Cada franja de color representa el porcentaje asignado a un activo en cada día del test. "
            "La suma siempre es 100%. Fíjate en dos cosas:  \n"
            "- **Franjas estables** = el agente mantiene posiciones (bajo coste de transacción)  \n"
            "- **Franjas muy cambiantes** = alta rotación, lo que erosiona el rendimiento con comisiones"
        )
        st.caption("Clic en la leyenda para aislar un activo concreto")
        if weights_history and len(weights_history[0]) == len(tickers):
            df_w = pd.DataFrame(weights_history, columns=tickers,
                                index=pd.to_datetime(df_p_test.index[:len(weights_history)]))
            fig_w = go.Figure()
            for ticker in tickers:
                fig_w.add_trace(go.Scatter(
                    x=df_w.index, y=df_w[ticker].values,
                    name=ticker, stackgroup='one',
                    hovertemplate=f"<b>{ticker}</b> %{{x|%d %b %Y}}: %{{y:.1%}}<extra></extra>"
                ))
            fig_w.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                                yaxis_title="Peso", yaxis_tickformat=".0%", height=350)
            st.plotly_chart(fig_w, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECCIÓN 5: Diagnóstico del Entrenamiento
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 5. ¿Aprendió Bien el Agente? — Diagnóstico del Entrenamiento")
    st.markdown(
        "Un backtest con buenos resultados no es suficiente: también hay que demostrar que el "
        "agente **aprendió de verdad** y no simplemente memorizó los datos de entrenamiento. "
        "Las tres gráficas siguientes son esa evidencia."
    )

    with st.expander("ℹ️  Guía para interpretar cada gráfica"):
        st.markdown("""
### Gráfica 1 — ¿Cómo fue el proceso de aprendizaje?

Muestra cinco indicadores internos del algoritmo PPO durante el entrenamiento.
Son el equivalente a un electrocardiograma: permiten saber si el entrenamiento
fue saludable o si hubo algún problema.

| Indicador | En palabras simples | Señal buena | Señal de problema |
|---|---|---|---|
| **Entropía** | ¿Cuánto explora el agente vs cuánto explota lo que ya sabe? | Baja poco a poco: primero explora, luego se especializa | Cae a cero muy rápido: el agente dejó de explorar antes de aprender |
| **Value Loss** | ¿Cuán equivocado está el agente al predecir sus recompensas futuras? | Baja y se estabiliza | Sube sin parar: la red es inestable |
| **Explained Variance** | ¿Qué fracción del futuro acierta a predecir? | Por encima de 0.5 | Negativa: predice peor que la media |
| **KL Divergence** | ¿Cuánto cambia su estrategia en cada actualización? | Por debajo de 0.05 | Supera 0.05: cambios demasiado bruscos, riesgo de colapso |
| **Clip Fraction** | ¿Cuántas actualizaciones frena PPO por ser demasiado grandes? | Entre 1% y 30% | Fuera de ese rango: clip_range mal calibrado |

---

### Gráfica 2 — ¿Memorizó los datos o aprendió de verdad?

Esta es la prueba más importante. Compara la recompensa en dos conjuntos:
- **Train** (azul): datos que el agente usó para aprender
- **Eval** (rojo): datos que el agente **nunca vio** durante el entrenamiento

**¿Qué buscar?**
- ✅ Ambas curvas suben juntas → el agente generalizó, funcionará en datos nuevos
- ⚠️ Train sube pero Eval no → **sobreajuste**: memorizó el pasado pero no aprendió nada transferible
- El sistema guarda automáticamente el modelo en el momento donde Eval es máximo, antes de que empiece a degradarse

---

### Gráfica 3 — ¿Funciona en distintos períodos de mercado?

Esta gráfica responde a: "¿los buenos resultados del backtest son casualidad o son robustos?"

Se divide toda la historia disponible en varias ventanas temporales. En cada una, el agente
se entrena desde cero con datos anteriores y se evalúa en el período siguiente que nunca vio.
Es el equivalente financiero del **k-fold cross-validation** en machine learning.

**¿Qué buscar?**
- ✅ Sharpe positivo y consistente en la mayoría de ventanas → la estrategia funciona en distintos regímenes (crisis, rally, consolidación)
- ⚠️ Alta varianza entre ventanas → el rendimiento depende de qué período toque: suerte, no habilidad
- ⚠️ Sharpe decreciente en las ventanas más recientes → el modelo está sesgado hacia el pasado lejano
        """)

    col_r1, col_r2, col_r3 = st.columns(3)
    for col, ruta, titulo, desc in [
        (col_r1, 'src/reports/training_diagnostics.png',
         "1. Salud del entrenamiento",
         "Entropía, value loss, explained variance, KL y clip fraction. "
         "Confirman que el algoritmo PPO convergió de forma estable."),
        (col_r2, 'src/reports/overfitting_analysis.png',
         "2. ¿Memorizó o aprendió?",
         "Reward en datos de train vs datos de evaluación nunca vistos. "
         "Si las dos curvas van juntas, el agente generalizó correctamente."),
        (col_r3, 'src/reports/walk_forward_analysis.png',
         "3. ¿Funciona en el tiempo?",
         "Sharpe, retorno y drawdown en ventanas temporales independientes. "
         "Resultados consistentes = estrategia robusta, no suerte de un período."),
    ]:
        col.markdown(f"**{titulo}**")
        col.caption(desc)
        if os.path.exists(ruta):
            col.image(ruta, use_container_width=True)
        else:
            col.info(f"Pendiente de generar.\n\n`{ruta}`")
