import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from src.trading_env import PortfolioEnv
from reports.academic_evaluation import (
    run_three_benchmark_backtest,
    load_multiseed_results,
    load_wfv_results,
)

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .section-title {
        color: #c9d1e0; font-size: 1rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase;
        border-left: 3px solid #3b82f6; padding-left: 10px; margin: 24px 0 12px;
    }
    div[data-testid="stMetric"] { background: #1e2130; border-radius: 8px; padding: 12px; }
    .insight {
        background: #1a1f2e;
        border-left: 3px solid #22c55e;
        border-radius: 0 6px 6px 0;
        padding: 8px 14px;
        color: #9ca3b0;
        font-size: 0.85rem;
        margin: 4px 0 12px 0;
    }
    .insight strong { color: #e2e8f0; }
    .formula {
        background: #0f1520;
        border: 1px solid #2e3555;
        border-radius: 6px;
        padding: 10px 16px;
        font-family: monospace;
        color: #93c5fd;
        font-size: 0.9rem;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper: bloques de explicación
# ---------------------------------------------------------------------------
def insight(texto):
    """Cápsula de insight siempre visible bajo una gráfica."""
    st.markdown(f'<div class="insight">{texto}</div>', unsafe_allow_html=True)

def formula(texto):
    """Bloque de fórmula con formato monospace."""
    st.markdown(f'<div class="formula">{texto}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## AI Portfolio Manager")
    st.markdown("---")
    st.markdown("#### Configuración")
    model_path    = st.text_input("Modelo principal", "models/best_model/best_model.zip")
    ablation_path = st.text_input("Modelo ablation (opcional)", "models/ablation_log_return/best_model.zip")
    commission    = st.slider("Comisión por operación (%)", 0.0, 0.5, 0.1, 0.01) / 100
    split_pct     = st.slider("Split train/test", 0.6, 0.9, 0.8, 0.05)
    reward_mode   = st.selectbox("Reward mode", ["rolling_sharpe", "sharpe_drawdown", "log_return"])
    capital       = st.number_input("Capital inicial ($)", min_value=1000, value=10000, step=1000)
    st.markdown("---")
    run_btn = st.button("Lanzar simulación", width="stretch", type="primary")
    st.markdown("---")
    st.caption("TFM · AI-Driven Portfolio Management · 2026")

# ---------------------------------------------------------------------------
# Utilidades de cálculo
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df_f = pd.read_csv('data/features_normalizadas.csv', index_col=0)
    df_p = pd.read_csv('data/precios_originales.csv', index_col=0)
    return df_f, df_p

def calcular_sharpe(rets, rf=0.0):
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return (rets.mean() - rf) / rets.std() * np.sqrt(252)

def calcular_max_drawdown(valores):
    s = pd.Series(valores)
    return float((s / s.cummax() - 1).min())

def calcular_drawdown_serie(valores):
    s = pd.Series(valores)
    return (s / s.cummax() - 1) * 100

def calcular_sortino(rets, rf=0.0):
    downside = rets[rets < 0].std()
    if downside == 0:
        return 0.0
    return (rets.mean() - rf) / downside * np.sqrt(252)

def calcular_calmar(valores):
    rets = pd.Series(valores).pct_change().dropna()
    ann  = rets.mean() * 252
    dd   = abs(calcular_max_drawdown(valores))
    return ann / dd if dd > 0 else 0.0

def simular(model_path_local, env):
    model = PPO.load(model_path_local)
    obs, _ = env.reset()
    done = False
    valores, drawdowns, pesos = [], [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        valores.append(info['value'])
        drawdowns.append(info.get('drawdown', 0.0))
        pesos.append(info.get('weights', np.zeros(env.n_assets)).copy())
    return valores, drawdowns, pesos

def benchmark_bh(env, capital_ini):
    rets = env.df_precios.pct_change().dropna()
    cum  = capital_ini * (1 + rets.mean(axis=1)).cumprod()
    return np.insert(cum.values, 0, capital_ini)

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.markdown("# AI Portfolio Manager")
st.markdown(
    "**Reinforcement Learning · PPO · Gymnasium** — "
    "Simulación de gestión activa de cartera sobre datos de mercado reales"
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
try:
    df_f, df_p = load_data()
    tickers = [c.replace("_Close", "") for c in df_p.columns]
except FileNotFoundError:
    st.error("No se encontraron los ficheros de datos en `data/`. Ejecuta primero `/fase1/preparar-datos`.")
    st.stop()

# ---------------------------------------------------------------------------
# Ejecución de la simulación
# ---------------------------------------------------------------------------
if run_btn:
    import os
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo en `{model_path}`. Entrena primero con `/fase3/entrenar`.")
        st.stop()

    split_idx = int(len(df_f) * split_pct)

    with st.spinner("Simulando cartera IA en datos de test..."):
        env_ia = PortfolioEnv(
            'data/features_normalizadas.csv', 'data/precios_originales.csv',
            initial_balance=capital, commission=commission,
            start_idx=split_idx, reward_mode=reward_mode,
            episode_length=None, random_start=False,
        )
        valores_ia, drawdowns_ia, pesos_hist = simular(model_path, env_ia)

    valores_bh = benchmark_bh(env_ia, capital)[:len(valores_ia)]
    dias        = list(range(len(valores_ia)))

    rets_ia = pd.Series(valores_ia).pct_change().dropna()
    rets_bh = pd.Series(valores_bh).pct_change().dropna()

    final_ia   = valores_ia[-1]
    final_bh   = valores_bh[-1]
    ret_ia     = (final_ia / capital - 1) * 100
    ret_bh     = (final_bh / capital - 1) * 100
    sharpe_ia  = calcular_sharpe(rets_ia)
    sharpe_bh  = calcular_sharpe(rets_bh)
    dd_ia      = calcular_max_drawdown(valores_ia) * 100
    dd_bh      = calcular_max_drawdown(valores_bh) * 100
    sortino_ia = calcular_sortino(rets_ia)
    calmar_ia  = calcular_calmar(valores_ia)
    vol_ia     = rets_ia.std() * np.sqrt(252) * 100

    # -----------------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "📉 Equity & Drawdown", "🥧 Asset Allocation",
        "⚠️ Risk Analytics", "🎓 Validación Académica"
    ])

    # =======================================================================
    # TAB 1 — OVERVIEW
    # =======================================================================
    with tab1:

        # --- KPIs ---
        st.markdown('<div class="section-title">Performance vs Benchmark</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Valor Final IA",    f"${final_ia:,.2f}",  f"{ret_ia:+.2f}%")
        c2.metric("Valor Final B&H",   f"${final_bh:,.2f}",  f"{ret_bh:+.2f}%")
        c3.metric("Sharpe IA",         f"{sharpe_ia:.3f}",   f"vs B&H {sharpe_bh:.3f}")
        c4.metric("Max Drawdown IA",   f"{dd_ia:.2f}%",      f"B&H {dd_bh:.2f}%")
        c5.metric("Volatilidad Anual", f"{vol_ia:.2f}%")

        with st.expander("📖 Cómo interpretar estos KPIs"):
            st.markdown("""
**Valor Final** es el resultado más directo: con cuánto dinero terminó el agente ese periodo.
Pero el valor final solo no es suficiente para evaluar un gestor — un resultado bueno obtenido
asumiendo un riesgo enorme no es un resultado bueno.

**Sharpe Ratio** responde a la pregunta clave: *¿cuánto retorno extra obtienes por cada unidad de riesgo que asumes?*
""")
            formula("Sharpe = (R_cartera − R_libre_riesgo) / σ_cartera  ×  √252")
            st.markdown("""
Multiplicamos por √252 para anualizarlo (252 días de mercado al año).
- **< 0**: el modelo pierde dinero ajustado por riesgo — peor que no invertir.
- **0 – 1**: rentable, pero el riesgo asumido es alto en proporción al retorno.
- **1 – 2**: zona de gestores activos competentes.
- **> 2**: excelente. Muy difícil de mantener sostenidamente.

**Max Drawdown** mide la *peor pesadilla real* del inversor: la mayor caída desde un máximo hasta
el mínimo posterior. Un drawdown de −30% significa que si hubieras invertido justo en el máximo,
habrías llegado a ver tu capital reducido un 30% en algún momento.

**Volatilidad Anual** es la desviación estándar de los retornos diarios, anualizada.
Representa la "agitación" de la cartera. Un fondo con baja volatilidad y buen retorno
es lo que busca cualquier inversor institucional.
""")

        # --- Equity Curve ---
        st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
        # Calcular MVP para mostrarlo también en la equity curve del Overview
        from reports.academic_evaluation import compute_mvp_weights, simulate_mvp
        df_p_full   = pd.read_csv('data/precios_originales.csv', index_col=0)
        pesos_mvp_  = compute_mvp_weights(df_p_full.iloc[:split_idx])
        valores_mvp = simulate_mvp(df_p_full.iloc[split_idx:], pesos_mvp_, capital)[:len(valores_ia)]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=dias, y=valores_ia, name="IA PPO",
            line=dict(color="#3b82f6", width=2),
            fill='tonexty', fillcolor='rgba(59,130,246,0.05)'
        ))
        fig_eq.add_trace(go.Scatter(
            x=dias, y=valores_bh, name="Buy & Hold 1/N",
            line=dict(color="#f97316", width=2, dash="dash")
        ))
        fig_eq.add_trace(go.Scatter(
            x=dias, y=valores_mvp, name="MVP (Markowitz)",
            line=dict(color="#22c55e", width=2, dash="dashdot")
        ))
        fig_eq.add_hline(y=capital, line_dash="dot", line_color="#6b7280", annotation_text="Capital inicial")
        fig_eq.update_layout(
            template="plotly_dark", height=380,
            xaxis_title="Días de negociación", yaxis_title="Valor ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=10, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig_eq, width="stretch")

        insight(
            f"<strong>Qué ver:</strong> ¿La línea azul (IA) termina por encima de la naranja (B&H)? "
            f"Si sí, el agente generó <em>alpha</em> — retorno por encima del mercado. "
            f"Diferencia final: <strong>${final_ia - final_bh:+,.2f}</strong>"
        )
        with st.expander("📖 Cómo leer la Equity Curve"):
            st.markdown("""
La **equity curve** (curva de capital) es la representación visual más directa del desempeño de
cualquier estrategia de inversión. Muestra el valor acumulado de la cartera día a día, partiendo
del capital inicial.
""")
            formula("V_t = V₀ × ∏ᵢ (1 + rᵢ)    donde rᵢ = retorno del día i")
            st.markdown("""
**La línea azul** representa al agente de IA (PPO). Cada punto es el valor de la cartera
ese día, después de que el agente haya decidido cómo distribuir el capital entre los 7 activos.

**La línea naranja punteada** es el *benchmark*: la estrategia Buy & Hold 1/N, que simplemente
invierte el mismo porcentaje en cada activo y no toca nada. Esta estrategia es difícil de batir
de forma consistente — es el listón mínimo que debe superar cualquier gestor activo para
justificar su existencia.

**La línea de puntos horizontal** es el capital inicial. Cuando la curva está por encima,
el inversor está en positivo; por debajo, en pérdidas.

**Qué buscar:**
- ¿La IA bate al benchmark de forma sostenida o solo al final?
- ¿Las caídas de la IA son menos profundas que las del benchmark?
- ¿La IA recupera las caídas más rápido?
""")

        # --- Tabla de métricas ---
        st.markdown('<div class="section-title">Tabla de Métricas</div>', unsafe_allow_html=True)
        rets_mvp   = pd.Series(valores_mvp).pct_change().dropna()
        ret_mvp    = (valores_mvp[-1] / capital - 1) * 100
        sharpe_mvp = calcular_sharpe(rets_mvp)
        dd_mvp     = calcular_max_drawdown(valores_mvp) * 100

        df_metrics = pd.DataFrame({
            "Métrica":           ["Retorno Total", "Sharpe Ratio", "Sortino Ratio",
                                  "Max Drawdown", "Calmar Ratio", "Volatilidad Anual"],
            "IA PPO":            [f"{ret_ia:+.2f}%", f"{sharpe_ia:.3f}", f"{sortino_ia:.3f}",
                                  f"{dd_ia:.2f}%",   f"{calmar_ia:.3f}", f"{vol_ia:.2f}%"],
            "Buy & Hold 1/N":    [f"{ret_bh:+.2f}%", f"{sharpe_bh:.3f}", f"{calcular_sortino(rets_bh):.3f}",
                                  f"{dd_bh:.2f}%",   f"{calcular_calmar(valores_bh):.3f}",
                                  f"{rets_bh.std()*np.sqrt(252)*100:.2f}%"],
            "MVP (Markowitz)":   [f"{ret_mvp:+.2f}%", f"{sharpe_mvp:.3f}", f"{calcular_sortino(rets_mvp):.3f}",
                                  f"{dd_mvp:.2f}%",   f"{calcular_calmar(valores_mvp):.3f}",
                                  f"{rets_mvp.std()*np.sqrt(252)*100:.2f}%"],
        })
        st.dataframe(df_metrics.set_index("Métrica"), width="stretch")

        with st.expander("📖 Qué significa cada fila de la tabla"):
            st.markdown("""
| Métrica | Qué mide | Fórmula simplificada | Referencia |
|---------|----------|----------------------|------------|
| **Retorno Total** | Ganancia o pérdida acumulada sobre el capital inicial | (V_final / V_inicial − 1) × 100 | Depende del periodo |
| **Sharpe Ratio** | Retorno extra por unidad de riesgo total | (R̄ − Rᶠ) / σ × √252 | > 1 aceptable, > 2 excelente |
| **Sortino Ratio** | Como el Sharpe, pero solo penaliza la volatilidad bajista | (R̄ − Rᶠ) / σ_negativa × √252 | Más justo: no castiga las subidas |
| **Max Drawdown** | La peor caída posible desde un máximo histórico | min(V_t / max(V₀..V_t) − 1) | Cuanto más cercano a 0%, mejor |
| **Calmar Ratio** | Retorno anualizado por cada punto de drawdown máximo | R_anual / |Max DD| | > 1 indica buena eficiencia de riesgo |
| **Volatilidad Anual** | Dispersión de los retornos diarios, escalada a 1 año | σ_diaria × √252 | Menor = más estable |

El **Sortino** es especialmente relevante en este contexto porque el agente IA fue entrenado con
una función de recompensa que *penaliza específicamente las caídas* (φ·drawdown). Si el modelo
aprendió bien, su Sortino debería ser notablemente mejor que el del benchmark.
""")

        # --- Ablation study ---
        import os
        if os.path.exists(ablation_path):
            st.markdown('<div class="section-title">Ablation Study: impacto del risk-shaping</div>', unsafe_allow_html=True)
            with st.spinner("Cargando modelo ablation..."):
                env_ab = PortfolioEnv(
                    'data/features_normalizadas.csv', 'data/precios_originales.csv',
                    initial_balance=capital, commission=commission,
                    start_idx=split_idx, reward_mode='log_return',
                    episode_length=None, random_start=False,
                )
                valores_ab, _, _ = simular(ablation_path, env_ab)
            rets_ab   = pd.Series(valores_ab).pct_change().dropna()
            ret_ab    = (valores_ab[-1] / capital - 1) * 100
            sharpe_ab = calcular_sharpe(rets_ab)
            dd_ab     = calcular_max_drawdown(valores_ab) * 100

            fig_ab = go.Figure()
            fig_ab.add_trace(go.Scatter(x=dias[:len(valores_ia)], y=valores_ia,
                name=f"sharpe_drawdown (Sharpe {sharpe_ia:.2f})", line=dict(color="#3b82f6", width=2)))
            fig_ab.add_trace(go.Scatter(x=dias[:len(valores_ab)], y=valores_ab,
                name=f"log_return baseline (Sharpe {sharpe_ab:.2f})", line=dict(color="#22c55e", width=2, dash="dashdot")))
            fig_ab.add_trace(go.Scatter(x=dias[:len(valores_bh)], y=valores_bh,
                name="Buy & Hold 1/N", line=dict(color="#f97316", width=1.5, dash="dash")))
            fig_ab.update_layout(
                template="plotly_dark", height=340,
                xaxis_title="Días", yaxis_title="Valor ($)",
                hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_ab, width="stretch")

            ca, cb, cc = st.columns(3)
            ca.metric("Retorno IA principal",   f"{ret_ia:+.2f}%")
            cb.metric("Retorno IA baseline",    f"{ret_ab:+.2f}%")
            cc.metric("ΔSharpe (SD − LR)",      f"{sharpe_ia - sharpe_ab:+.3f}")

            delta_interpretacion = (
                "el risk-shaping **mejoró** la calidad de las decisiones"
                if sharpe_ia > sharpe_ab else
                "el risk-shaping **no aportó mejora** — revisar φ o las features de riesgo"
            )
            insight(
                f"<strong>Conclusión del ablation:</strong> ΔSharpe = {sharpe_ia - sharpe_ab:+.3f} → {delta_interpretacion}."
            )

            with st.expander("📖 Qué es un ablation study y por qué importa"):
                st.markdown("""
Un **ablation study** (estudio de ablación) es una técnica estándar en investigación de IA
para medir la contribución real de cada componente del sistema. El término viene de la neurociencia:
se "extirpa" una parte del sistema y se observa qué deja de funcionar.

En este caso, comparamos **dos versiones del mismo agente PPO**:

- **Modelo principal** (`sharpe_drawdown`): entrenado con R_t = log_return − φ·drawdown.
  El agente fue *incentivado explícitamente* a evitar caídas grandes.
- **Modelo baseline** (`log_return`): entrenado con R_t = log_return únicamente.
  El agente solo maximiza retorno, sin ningún coste por el riesgo asumido.

**¿Qué demuestra el ΔSharpe?**

Si el modelo principal tiene un Sharpe *significativamente mayor* que el baseline,
concluimos que la penalización por drawdown aportó valor real: el agente aprendió
a usar las features de riesgo (kurtosis, div_volatility, corr_IBIT_IVV) para
*anticipar* y *evitar* las caídas, en lugar de simplemente reaccionar a ellas.

Si el ΔSharpe es ≈ 0 o negativo, indica que el parámetro φ=0.5 podría necesitar ajuste,
o que las features de riesgo no están siendo correctamente aprovechadas.
""")
                formula("ΔSharpe = Sharpe(sharpe_drawdown) − Sharpe(log_return)")

    # =======================================================================
    # TAB 2 — EQUITY & DRAWDOWN
    # =======================================================================
    with tab2:

        st.markdown('<div class="section-title">Equity Curve + Drawdown Combinados</div>', unsafe_allow_html=True)
        dd_serie_ia = calcular_drawdown_serie(valores_ia)
        dd_serie_bh = calcular_drawdown_serie(valores_bh)

        fig_dd = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.04,
            subplot_titles=("Valor de la Cartera ($)", "Drawdown desde máximo histórico (%)")
        )
        fig_dd.add_trace(go.Scatter(x=dias, y=valores_ia, name="IA PPO",
            line=dict(color="#3b82f6", width=2)), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=dias, y=valores_bh, name="B&H 1/N",
            line=dict(color="#f97316", width=1.5, dash="dash")), row=1, col=1)
        fig_dd.add_trace(go.Scatter(x=dias, y=dd_serie_ia, name="Drawdown IA",
            fill='tozeroy', fillcolor='rgba(239,68,68,0.2)',
            line=dict(color="#ef4444", width=1)), row=2, col=1)
        fig_dd.add_trace(go.Scatter(x=dias, y=dd_serie_bh, name="Drawdown B&H",
            line=dict(color="#f97316", width=1, dash="dot")), row=2, col=1)
        fig_dd.update_layout(
            template="plotly_dark", height=520,
            hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_dd, width="stretch")

        insight(
            f"<strong>Max Drawdown IA: {dd_ia:.2f}%</strong> vs B&H: {dd_bh:.2f}%. "
            "El área roja representa el 'dolor' del inversor en cada momento — cuánto dinero "
            "habría perdido en papel respecto a su mejor momento anterior."
        )
        with st.expander("📖 Cómo leer el gráfico Equity + Drawdown"):
            st.markdown("""
Este gráfico de dos paneles es el estándar en análisis cuantitativo de fondos de inversión.
Muestra la historia del valor y del riesgo de forma sincronizada.

**Panel superior — Equity Curve:**
La historia del capital. Las subidas y bajadas reflejan las decisiones de rebalanceo de la IA
y los movimientos del mercado. Las dos líneas se comparan directamente:
cuando la azul cae más lentamente que la naranja durante una crisis, el risk-shaping está funcionando.

**Panel inferior — Drawdown:**
En cada día t, el drawdown es:
""")
            formula("DD(t) = (máximo(V₀, V₁, ..., V_t) − V_t) / máximo(V₀, V₁, ..., V_t)  × 100")
            st.markdown("""
Es decir: *¿cuánto ha caído la cartera desde su mejor momento anterior?*

El área roja rellena la "herida" del inversor. Una crisis profunda y larga (área grande y extendida)
es mucho más difícil psicológicamente que muchas caídas pequeñas y cortas.

**Qué buscar:**
- La **profundidad** del drawdown: ¿cuánto llegó a caer como máximo?
- La **duración** del drawdown: ¿cuántos días tardó en recuperar el máximo anterior?
- La **forma**: drawdowns en V (caída brusca y recuperación rápida) vs. drawdowns en U o L
  (caídas lentas y prolongadas, características de mercados bajistas estructurales).

Un buen gestor activo debería tener drawdowns más cortos y menos profundos que su benchmark,
especialmente si fue entrenado explícitamente para minimizarlos.
""")

        # --- Rolling Sharpe ---
        st.markdown('<div class="section-title">Rolling Sharpe Ratio (ventana 60 días)</div>', unsafe_allow_html=True)
        rolling_sharpe_ia = rets_ia.rolling(60).apply(calcular_sharpe, raw=True)
        rolling_sharpe_bh = rets_bh.rolling(60).apply(calcular_sharpe, raw=True)

        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_dash="dot", line_color="#6b7280",
                         annotation_text="Sharpe = 0 (umbral de valor)", annotation_position="right")
        fig_rs.add_hline(y=1, line_dash="dot", line_color="#374151",
                         annotation_text="Sharpe = 1 (objetivo mínimo)", annotation_position="right")
        fig_rs.add_trace(go.Scatter(y=rolling_sharpe_ia.values, name="IA PPO",
            line=dict(color="#3b82f6", width=2)))
        fig_rs.add_trace(go.Scatter(y=rolling_sharpe_bh.values, name="B&H 1/N",
            line=dict(color="#f97316", width=1.5, dash="dash")))
        fig_rs.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Días", yaxis_title="Sharpe anualizado (60d)",
            hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_rs, width="stretch")

        insight(
            "<strong>Qué ver:</strong> Un Sharpe rodante que se mantiene por encima de 0 "
            "de forma consistente indica un modelo robusto. Si cae menos que el benchmark "
            "durante las crisis de mercado, el risk-shaping está cumpliendo su función."
        )
        with st.expander("📖 Por qué el Sharpe rodante es más honesto que el Sharpe global"):
            st.markdown("""
El Sharpe Ratio que aparece en la tabla de métricas es un **único número** que resume todo el
periodo de test. El problema es que ese número puede esconder periodos de muy buen desempeño
y periodos desastrosos que se compensan entre sí.

El **Rolling Sharpe** calcula el Sharpe en una ventana móvil de 60 días (~3 meses de mercado).
Esto permite responder preguntas que el Sharpe global no puede:

- *¿El modelo funciona bien en todos los regímenes de mercado?*
  Si el Rolling Sharpe cae a valores muy negativos durante mercados bajistas, el modelo
  no está generalizando bien.

- *¿El modelo mejora respecto al benchmark precisamente cuando más importa?*
  Un modelo valioso debería sufrir *menos* que el B&H durante las crisis
  (su Rolling Sharpe debería caer menos o recuperarse antes).

- *¿Hay consistencia?* Un Sharpe global de 1.5 con mucha varianza mensual
  es menos deseable que un Sharpe global de 1.2 muy estable.

**Umbral de referencia Sharpe = 0:** cuando la línea está por debajo de 0, la cartera
está destruyendo valor ajustado por riesgo en esa ventana — es peor que estar en cash.
""")

    # =======================================================================
    # TAB 3 — ASSET ALLOCATION
    # =======================================================================
    with tab3:

        last_weights = np.array(pesos_hist[-1]).flatten()
        pesos_array  = np.array(pesos_hist)
        step_sample  = max(1, len(pesos_array) // 100)
        pesos_sample = pesos_array[::step_sample]
        dias_sample  = dias[::step_sample]

        # --- Donut + Bar ---
        st.markdown('<div class="section-title">Allocación Final de la IA (último día de test)</div>', unsafe_allow_html=True)
        col_pie, col_bar = st.columns([1, 1])

        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=tickers, values=last_weights, hole=0.45,
                textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Bold, line=dict(color='#0e1117', width=2))
            ))
            fig_pie.update_layout(
                template="plotly_dark", height=360, showlegend=False,
                annotations=[dict(text="Último<br>rebalanceo", x=0.5, y=0.5, font_size=12, showarrow=False)],
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_pie, width="stretch")

        with col_bar:
            sorted_idx = np.argsort(last_weights)[::-1]
            fig_bar = go.Figure(go.Bar(
                x=[tickers[i] for i in sorted_idx],
                y=[last_weights[i] * 100 for i in sorted_idx],
                marker_color=px.colors.qualitative.Bold[:len(tickers)],
                text=[f"{last_weights[i]*100:.1f}%" for i in sorted_idx],
                textposition='outside'
            ))
            fig_bar.update_layout(
                template="plotly_dark", height=360,
                yaxis_title="Peso (%)", xaxis_title="Activo",
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_bar, width="stretch")

        top_activo = tickers[int(np.argmax(last_weights))]
        insight(
            f"<strong>Convicción principal:</strong> el agente concentró su mayor posición "
            f"en <strong>{top_activo}</strong> ({last_weights.max()*100:.1f}%) "
            f"y su menor posición en <strong>{tickers[int(np.argmin(last_weights))]}</strong> "
            f"({last_weights.min()*100:.1f}%). Una cartera más equitativa indica más diversificación."
        )
        with st.expander("📖 Cómo interpretar la allocación de la IA"):
            st.markdown("""
La **allocación de cartera** es la decisión central de cualquier gestor de inversiones:
¿qué porcentaje del capital total asignamos a cada activo?

En teoría moderna de carteras (Markowitz, 1952), la allocación óptima maximiza el retorno
esperado para un nivel de riesgo dado, o equivalentemente, minimiza el riesgo para un retorno
dado. El resultado es la **frontera eficiente**.

El agente IA no conoce explícitamente la teoría de Markowitz — aprende empíricamente a través
del refuerzo qué allocaciones han funcionado históricamente dado el contexto de mercado.

**Qué nos dice la allocación final:**
- **Alta concentración en un activo** (> 40%): el agente tiene alta "convicción" — ha aprendido
  que ese activo domina la rentabilidad en el contexto actual del mercado.
- **Distribución muy uniforme** (todos ~1/N): el agente está en modo defensivo o no tiene señal clara.
- **Cero o casi cero en algún activo**: el agente lo considera negativo esperado en ese momento
  (equivalente a un *short* limitado a cero en estrategias long-only).

La acción del agente es un vector de N valores reales que se proyecta al **simplex** (pesos
que suman 1, todos ≥ 0) mediante normalización. Esto garantiza que nunca haya apalancamiento
ni posiciones cortas.
""")
            formula("w_i = a_i / Σⱼ aⱼ    donde a_i ∈ [0,1]  es la salida de la red")

        # --- Heatmap ---
        st.markdown('<div class="section-title">Evolución de la Allocación — Mapa de Calor</div>', unsafe_allow_html=True)
        fig_heat = go.Figure(go.Heatmap(
            z=pesos_sample.T * 100, x=dias_sample, y=tickers,
            colorscale="Blues", colorbar=dict(title="Peso (%)"), zmin=0, zmax=100
        ))
        fig_heat.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Días de negociación", yaxis_title="Activo",
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_heat, width="stretch")

        insight(
            "<strong>Franjas oscuras continuas</strong> = el agente mantiene alta convicción en ese activo durante días. "
            "<strong>Franjas que cambian rápido</strong> = rotación activa. "
            "Mucho ruido sin patrón = el modelo podría estar sobrereaccionando a señales de corto plazo."
        )
        with st.expander("📖 Cómo leer el mapa de calor de allocación"):
            st.markdown("""
El mapa de calor muestra la **historia completa de las decisiones de la IA** de forma compacta.
Cada fila es un activo y cada columna es un día de simulación. El color representa el peso
asignado: azul oscuro = mucho peso, casi blanco = peso mínimo.

Este gráfico revela la **estrategia implícita** que el agente aprendió:

- **Franjas horizontales oscuras y prolongadas:** el agente mantiene una posición alta en ese
  activo durante muchos días seguidos. Indica que ha aprendido una señal persistente
  (p. ej.: BND en periodos de alta volatilidad del mercado → activo refugio).

- **Transiciones bruscas de color:** rebalanceos importantes. El agente detectó un cambio de
  régimen de mercado y ajustó la cartera rápidamente. Esto tiene un coste de transacción
  que el modelo ya internalizó durante el entrenamiento.

- **Patrón muy ruidoso sin estructura:** puede indicar que el modelo está sobreajustando
  a fluctuaciones de corto plazo en lugar de aprender tendencias. Un buen indicador de
  que los hiperparámetros o las features podrían mejorarse.

- **Actividad coordinada entre activos:** si cuando IVV cae, BND sube en el mapa →
  el agente está implementando rotación defensiva, exactamente lo que se esperaría de
  un gestor humano inteligente.
""")

        # --- Área apilada ---
        st.markdown('<div class="section-title">Rotación Sectorial — Pesos Apilados</div>', unsafe_allow_html=True)
        fig_area = go.Figure()
        colors = px.colors.qualitative.Bold
        for i, ticker in enumerate(tickers):
            fig_area.add_trace(go.Scatter(
                x=dias_sample, y=pesos_sample[:, i] * 100,
                name=ticker, stackgroup='one',
                line=dict(width=0), fillcolor=colors[i % len(colors)],
            ))
        fig_area.update_layout(
            template="plotly_dark", height=300,
            xaxis_title="Días de negociación", yaxis_title="Peso (%)",
            hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_area, width="stretch")

        insight(
            "<strong>Qué ver:</strong> ¿El agente rota hacia activos defensivos (BND, CB) "
            "cuando el mercado cae y hacia activos de crecimiento (IVV, IBIT) cuando sube? "
            "Si sí, la IA está aprendiendo gestión táctica de activos."
        )
        with st.expander("📖 Rotación táctica de activos: qué nos revela este gráfico"):
            st.markdown("""
El gráfico de áreas apiladas siempre suma el 100% — muestra en todo momento cómo se distribuye
el capital total entre los activos. Es la forma más clara de observar la **rotación sectorial**
que hace el agente a lo largo del tiempo.

**Rotación táctica de activos (TAA)** es la estrategia por la que un gestor ajusta los pesos
de la cartera en función del ciclo económico y los regímenes de mercado. Los gestores humanos
lo hacen basándose en análisis macroeconómico; el agente IA lo aprende de las features.

**Patrones que indican aprendizaje genuino:**

- **Expansión de IVV / IBIT en mercados alcistas:** los activos de mayor retorno esperado
  ganan peso cuando el contexto es favorable.

- **Expansión de BND / CB en mercados de estrés:** BND (bonos del Tesoro) y CB (seguros)
  son activos defensivos clásicos. Si el agente los sobrepondera cuando la kurtosis sube
  o la corr_IBIT_IVV aumenta, está usando exactamente las features que le enseñamos.

- **Peso relevante en MO o JNJ en entornos de alta incertidumbre:** los aristócratas del
  dividendo actúan como colchón de retorno en periodos laterales.

La riqueza de este gráfico está en correlacionarlo mentalmente con la equity curve del
panel anterior: ¿los cambios de allocación preceden a las caídas o reaccionan a ellas?
Si los preceden, el agente está anticipando. Si reaccionan, está siguiendo precio.
""")

    # =======================================================================
    # TAB 4 — RISK ANALYTICS
    # =======================================================================
    with tab4:

        # --- Distribución de retornos ---
        st.markdown('<div class="section-title">Distribución de Retornos Diarios</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rets_ia * 100, name="IA PPO", nbinsx=60, opacity=0.7, marker_color="#3b82f6"
        ))
        fig_hist.add_trace(go.Histogram(
            x=rets_bh * 100, name="B&H 1/N", nbinsx=60, opacity=0.5, marker_color="#f97316"
        ))
        fig_hist.add_vline(x=0, line_dash="dot", line_color="#6b7280")
        fig_hist.update_layout(
            template="plotly_dark", height=320, barmode='overlay',
            xaxis_title="Retorno diario (%)", yaxis_title="Nº de días",
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_hist, width="stretch")

        skew_ia = float(rets_ia.skew())
        kurt_ia = float(rets_ia.kurt())
        insight(
            f"<strong>Forma de la distribución IA:</strong> "
            f"Skewness = {skew_ia:.3f} "
            f"({'cola a la izquierda → más riesgo de pérdidas bruscas' if skew_ia < 0 else 'cola a la derecha → sorpresas positivas más frecuentes'}). "
            f"Kurtosis = {kurt_ia:.3f} "
            f"({'colas pesadas → eventos extremos más probables de lo normal' if kurt_ia > 3 else 'distribución relativamente normal'})."
        )
        with st.expander("📖 Por qué la forma de la distribución importa tanto como la media"):
            st.markdown("""
La **distribución de retornos diarios** muestra cuántos días la cartera subió X%, bajó Y%, etc.
En finanzas, la distribución normal (campana simétrica) es el supuesto estándar — pero los
mercados reales casi nunca la cumplen.

**Dos métricas clave de la forma:**
""")
            formula("Skewness = E[(X − μ)³] / σ³     |     Kurtosis = E[(X − μ)⁴] / σ⁴")
            st.markdown("""
**Skewness (asimetría):**
- = 0: distribución simétrica (ideal teórico).
- < 0: cola izquierda larga → hay más días malos que lo esperado. Los desastres son más frecuentes
  que las bonanzas del mismo tamaño. Esto es lo habitual en renta variable.
- > 0: cola derecha larga → las sorpresas positivas son mayores que las negativas. Deseable.

**Kurtosis (curtosis):**
- = 3 en una distribución normal (o 0 en "kurtosis excess").
- > 3: distribución **leptocúrtica** (colas pesadas). Los días de movimientos extremos — tanto
  positivos como negativos — son más frecuentes de lo que predice la campana de Gauss.
  Esto es característico de activos financieros, especialmente en periodos de crisis.
- El agente fue entrenado con features de kurtosis rolling precisamente para detectar cuando
  entramos en un régimen de colas pesadas y reducir la exposición a tiempo.

**Comparando IA vs B&H:**
- Si la distribución de la IA tiene una cola izquierda *menos extensa* que la del B&H,
  el modelo está truncando los peores días.
- Si ambas distribuciones tienen el mismo rango de retornos negativos, el risk-shaping
  no está aportando protección real.
""")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            # --- VaR & CVaR ---
            st.markdown('<div class="section-title">VaR & CVaR (nivel de confianza 95%)</div>', unsafe_allow_html=True)
            var_95  = float(np.percentile(rets_ia, 5) * 100)
            cvar_95 = float(rets_ia[rets_ia <= np.percentile(rets_ia, 5)].mean() * 100)
            var_bh  = float(np.percentile(rets_bh, 5) * 100)
            cvar_bh = float(rets_bh[rets_bh <= np.percentile(rets_bh, 5)].mean() * 100)

            df_var = pd.DataFrame({
                "Métrica":        ["VaR 95%", "CVaR 95%"],
                "IA PPO":         [f"{var_95:.3f}%",  f"{cvar_95:.3f}%"],
                "Buy & Hold 1/N": [f"{var_bh:.3f}%",  f"{cvar_bh:.3f}%"],
            })
            st.dataframe(df_var.set_index("Métrica"), width="stretch")

            insight(
                f"<strong>VaR IA {var_95:.2f}%</strong>: en el 5% de los peores días, "
                f"perderás más de {abs(var_95):.2f}% del capital ese día. "
                f"El <strong>CVaR {cvar_95:.2f}%</strong> te dice cuánto perderás <em>en promedio</em> "
                "en esos peores días — siempre más severo que el VaR."
            )
            with st.expander("📖 VaR y CVaR: la medida de riesgo de los reguladores"):
                st.markdown("""
El **Value at Risk (VaR)** al 95% responde a: *"¿cuánto puedo perder como máximo el 95% de los días?"*
O equivalentemente: *"El 5% de los días puedo perder más de X — ¿cuánto es ese X?"*
""")
                formula("VaR₉₅ = −percentil(retornos, 5%)")
                st.markdown("""
Es el estándar regulatorio internacional (Basilea III, ESMA) para medir el riesgo de mercado.
Los bancos están obligados a calcular su VaR diario.

**El problema del VaR:** no dice *cuánto* se pierde más allá del umbral — solo que se supera.
Ahí entra el **CVaR** (también llamado *Expected Shortfall*):
""")
                formula("CVaR₉₅ = E[retorno | retorno < VaR₉₅]")
                st.markdown("""
El CVaR es el *promedio* de los retornos en el 5% peor de los días.
Siempre es más severo que el VaR y captura el "riesgo de la cola" que el VaR ignora.

Desde 2016, el CVaR ha reemplazado al VaR en los marcos regulatorios avanzados
precisamente porque es más informativo cuando los mercados se comportan de forma extrema.

**Para este proyecto:** si el CVaR de la IA es *menos negativo* que el del B&H,
significa que en los peores días de mercado la IA pierde *menos en promedio* —
lo que indica que el risk-shaping está protegiendo efectivamente el capital en las crisis.
""")

            # --- Métricas de riesgo-retorno ---
            st.markdown('<div class="section-title">Ratios de Riesgo-Retorno</div>', unsafe_allow_html=True)
            df_risk = pd.DataFrame({
                "Métrica": ["Sortino Ratio", "Calmar Ratio", "Skewness", "Kurtosis (excess)"],
                "IA PPO":  [f"{sortino_ia:.3f}", f"{calmar_ia:.3f}",
                            f"{float(rets_ia.skew()):.3f}", f"{float(rets_ia.kurt()):.3f}"],
                "Buy & Hold 1/N": [f"{calcular_sortino(rets_bh):.3f}", f"{calcular_calmar(valores_bh):.3f}",
                                   f"{float(rets_bh.skew()):.3f}", f"{float(rets_bh.kurt()):.3f}"],
            })
            st.dataframe(df_risk.set_index("Métrica"), width="stretch")

            with st.expander("📖 Sortino y Calmar: los ratios que el Sharpe no cuenta"):
                st.markdown("""
**Sortino Ratio** — *la versión justa del Sharpe*

El Sharpe penaliza *toda* la volatilidad, incluyendo los días en que la cartera sube mucho.
Pero al inversor no le molesta que la cartera suba — solo le molesta que baje.
El Sortino solo penaliza la **volatilidad bajista**:
""")
                formula("Sortino = (R̄ − Rᶠ) / σ_negativa  ×  √252")
                st.markdown("""
donde σ_negativa es la desviación estándar calculada *solo sobre los retornos negativos*.

Un modelo que tiene muchos días de grandes subidas tendrá un Sortino *mayor* que su Sharpe.
Si el Sortino de la IA es significativamente mayor que el del B&H, significa que el modelo
ha aprendido a eliminar asimetrías — más días positivos, caídas más moderadas.

---

**Calmar Ratio** — *eficiencia del capital frente al peor escenario*
""")
                formula("Calmar = Retorno Anualizado / |Max Drawdown|")
                st.markdown("""
Responde a: *"Por cada 1% de drawdown máximo que acepto, ¿cuánto retorno anualizado obtengo?"*

Un Calmar de 0.5 significa que por soportar una caída máxima del 20%, obtengo un 10% anual.
Un Calmar de 2.0 significa que por soportar esa misma caída del 20%, obtengo un 40% anual.

Es la métrica favorita en fondos de cobertura con mandato de preservación de capital.
""")

        with col_r2:
            # --- Retornos mensuales ---
            st.markdown('<div class="section-title">Retornos Mensuales — Calendario de Rendimiento</div>', unsafe_allow_html=True)
            fechas = df_f.index[split_idx: split_idx + len(valores_ia)]
            try:
                serie_ia = pd.Series(valores_ia, index=pd.to_datetime(fechas))
                monthly  = serie_ia.resample('ME').last().pct_change().dropna() * 100
                colors_monthly = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly.values]
                fig_monthly = go.Figure(go.Bar(
                    x=[d.strftime("%b %Y") for d in monthly.index],
                    y=monthly.values,
                    marker_color=colors_monthly,
                    text=[f"{v:+.1f}%" for v in monthly.values],
                    textposition='outside'
                ))
                fig_monthly.update_layout(
                    template="plotly_dark", height=340,
                    xaxis_title="Mes", yaxis_title="Retorno (%)",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_monthly, width="stretch")

                meses_pos = (monthly > 0).sum()
                meses_neg = (monthly <= 0).sum()
                ret_medio_pos = monthly[monthly > 0].mean() if meses_pos > 0 else 0
                ret_medio_neg = monthly[monthly <= 0].mean() if meses_neg > 0 else 0
                insight(
                    f"<strong>{meses_pos} meses positivos</strong> (media +{ret_medio_pos:.1f}%) vs "
                    f"<strong>{meses_neg} meses negativos</strong> (media {ret_medio_neg:.1f}%). "
                    "Un ratio de rentabilidad positiva/negativa > 1 y pérdidas medias menores que "
                    "ganancias medias define un perfil de retornos asimétrico deseable."
                )
                with st.expander("📖 El calendario mensual: consistencia por encima de los picos"):
                    st.markdown("""
El **calendario de rendimiento mensual** es una de las primeras cosas que mira cualquier
inversor institucional antes de comprometer capital en un fondo. Un retorno total alto
puede venir de unos pocos meses extraordinarios que compensan muchos meses mediocres.

**Qué buscar:**

- **Ratio meses positivos / negativos:** más de 2:1 (dos meses buenos por cada malo) es
  el listón típico para un fondo competente.

- **Asimetría entre ganancias y pérdidas medias:** un buen perfil tiene pérdidas medias
  menores en valor absoluto que las ganancias medias. Matemáticamente, esto produce
  retornos compuestos superiores aunque el retorno aritmético sea el mismo.
  Por ejemplo, +5% y −4% compuesto = +0.8% neto vs +3% y −3% = 0% neto.

- **Ausencia de meses de pérdidas extremas:** una barra roja de −15% en un mes
  puede destruir meses de trabajo acumulado. Si el agente entrenado con penalización
  de drawdown evita esos meses catastróficos, la función de recompensa está haciendo
  exactamente lo que se le pidió.

- **Consistencia estacional:** ¿hay meses del año donde la IA sistemáticamente lo hace
  mejor o peor? Esto podría revelar patrones de mercado (efecto enero, rally navideño)
  que el modelo ha capturado o que no ha capturado.
""")
            except Exception:
                st.info("No hay suficientes datos para el gráfico mensual.")

    # =======================================================================
    # TAB 5 — VALIDACIÓN ACADÉMICA
    # =======================================================================
    with tab5:

        # --- MVP: pesos ---
        st.markdown('<div class="section-title">Minimum Variance Portfolio — Pesos Óptimos</div>', unsafe_allow_html=True)

        tickers_mvp = [c.replace("_Close", "") for c in pd.read_csv('data/precios_originales.csv', index_col=0).columns]
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            fig_mvp_pie = go.Figure(go.Pie(
                labels=tickers_mvp, values=pesos_mvp_,
                hole=0.4, textinfo='label+percent',
                marker=dict(colors=px.colors.qualitative.Safe, line=dict(color='#0e1117', width=2))
            ))
            fig_mvp_pie.update_layout(
                template="plotly_dark", height=320, showlegend=False,
                annotations=[dict(text="MVP<br>Markowitz", x=0.5, y=0.5, font_size=12, showarrow=False)],
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig_mvp_pie, width="stretch")

        with col_m2:
            df_mvp_w = pd.DataFrame({
                'Activo': tickers_mvp,
                'Peso MVP': [f"{w*100:.1f}%" for w in pesos_mvp_],
                'Peso B&H 1/N': [f"{100/len(tickers_mvp):.1f}%"] * len(tickers_mvp),
            })
            st.dataframe(df_mvp_w.set_index('Activo'), width="stretch")

        dominante_mvp = tickers_mvp[int(np.argmax(pesos_mvp_))]
        insight(
            f"<strong>El MVP concentra el mayor peso en {dominante_mvp} ({pesos_mvp_.max()*100:.1f}%)</strong>. "
            "El MVP premia activos con baja varianza individual y baja correlación con el resto — "
            "generalmente bonos (BND) y utilities (AWK, CB) en lugar de renta variable pura."
        )
        with st.expander("📖 Qué es el Minimum Variance Portfolio y por qué es un benchmark más exigente que B&H 1/N"):
            st.markdown("""
El **Minimum Variance Portfolio** (MVP) es la solución al problema de optimización de Markowitz (1952)
que minimiza la varianza de la cartera sujeto a que los pesos sumen 1 y sean no-negativos:
""")
            formula("min  w' Σ w     s.t.  Σwᵢ = 1,  wᵢ ≥ 0")
            st.markdown("""
donde **Σ** es la matriz de covarianza de los retornos diarios de los activos.

La solución analítica (sin restricción de positividad) es:
""")
            formula("w* = (Σ⁻¹ · 1) / (1' · Σ⁻¹ · 1)")
            st.markdown("""
Con la restricción long-only (wᵢ ≥ 0), se resuelve numéricamente con SLSQP
(Sequential Least Squares Programming).

**¿Por qué el MVP es un benchmark más justo que B&H 1/N?**

- El B&H 1/N no usa ninguna información — es el benchmark más fácil de batir.
- El MVP usa la *estructura histórica del mercado* (covarianzas del periodo de train)
  para construir la cartera más eficiente en varianza. Si la IA no supera al MVP,
  significa que no está aportando valor más allá de la diversificación estática óptima.

**Importante:** los pesos MVP se calculan en el periodo de **entrenamiento** y se
aplican **fijos** durante todo el test. Esto garantiza la evaluación out-of-sample:
el MVP no hace trampa usando datos futuros. La diferencia con la IA es que el MVP
usa pesos estáticos mientras que la IA los ajusta diariamente según las features.
""")

        st.markdown("---")

        # --- Multi-seed ---
        st.markdown('<div class="section-title">Robustez Estadística — Múltiples Semillas</div>', unsafe_allow_html=True)
        resultado_ms = load_multiseed_results()
        if resultado_ms is not None:
            df_ms, resumen_ms = resultado_ms
            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                fig_ms = go.Figure()
                fig_ms.add_trace(go.Bar(
                    x=[f"Seed {s}" for s in df_ms['seed']],
                    y=df_ms['sharpe'],
                    marker_color=["#3b82f6", "#22c55e", "#f97316"][:len(df_ms)],
                    text=[f"{v:.3f}" for v in df_ms['sharpe']],
                    textposition='outside',
                    name='Sharpe por semilla'
                ))
                media = df_ms['sharpe'].mean()
                std   = df_ms['sharpe'].std()
                fig_ms.add_hline(y=media, line_dash="dash", line_color="#e2e8f0",
                                 annotation_text=f"Media {media:.3f}")
                fig_ms.add_hrect(y0=media - std, y1=media + std,
                                 fillcolor="rgba(255,255,255,0.05)", line_width=0,
                                 annotation_text=f"±1σ", annotation_position="right")
                fig_ms.update_layout(
                    template="plotly_dark", height=300,
                    yaxis_title="Sharpe Ratio", margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_ms, width="stretch")

            with col_s2:
                st.dataframe(resumen_ms.set_index('Metric'), width="stretch")
                st.dataframe(df_ms[['seed', 'sharpe', 'max_drawdown']].set_index('seed'),
                             width="stretch")

            insight(
                f"<strong>Sharpe: {media:.3f} ± {std:.3f}</strong>. "
                "Un std bajo respecto a la media indica que el resultado es robusto "
                "y no depende de una inicialización favorable de la red neuronal."
            )
        else:
            st.info("Ejecuta `entrenar_multisemilla()` desde train.py o el endpoint `/fase3/multisemilla` "
                    "para generar estos resultados.")

        with st.expander("📖 Por qué las múltiples semillas son obligatorias en IA para finanzas"):
            st.markdown("""
Los algoritmos de deep reinforcement learning como PPO tienen **alta varianza estocástica**:
los mismos hiperparámetros con distintas semillas aleatorias pueden producir modelos con
Sharpe Ratios que difieren en ±0.3 o más. Esto ocurre por tres fuentes de aleatoriedad:

1. **Inicialización de pesos** de la red neuronal (Xavier/Glorot inicialización aleatoria).
2. **Orden de los mini-batches** durante las actualizaciones del gradiente.
3. **Muestreo de acciones** durante la fase de exploración (la política es estocástica).

Un resultado publicado con una sola semilla es científicamente sospechoso: podría ser
un *cherrypick* accidental. El estándar en investigación de RL aplicado a finanzas
(Zhang et al., 2020; Liu et al., 2022) es reportar resultados sobre al menos 5 semillas.

**Cómo interpretar el intervalo de confianza:**
""")
            formula("Sharpe reportado = μ ± σ   (media y desviación típica sobre N semillas)")
            st.markdown("""
- Si **σ < 0.1·μ**: el resultado es muy robusto (variabilidad < 10%).
- Si **σ ≈ 0.2·μ** o más: hay alta sensibilidad al seed — considerar más timesteps
  o ajustar la tasa de aprendizaje para reducir la varianza.
""")

        st.markdown("---")

        # --- Walk-forward ---
        st.markdown('<div class="section-title">Walk-Forward Validation — Generalización por Régimen</div>', unsafe_allow_html=True)
        df_wfv = load_wfv_results()
        if df_wfv is not None:
            fig_wfv = go.Figure()
            colors_wfv = ["#3b82f6", "#22c55e", "#f97316"]
            fig_wfv.add_trace(go.Bar(
                x=df_wfv['descripcion'],
                y=df_wfv['sharpe'],
                marker_color=colors_wfv[:len(df_wfv)],
                text=[f"{v:.3f}" for v in df_wfv['sharpe']],
                textposition='outside',
                name='Sharpe por ventana'
            ))
            fig_wfv.add_hline(y=0, line_dash="dot", line_color="#6b7280",
                              annotation_text="Sharpe = 0 (umbral de valor)")
            fig_wfv.update_layout(
                template="plotly_dark", height=320,
                yaxis_title="Sharpe Ratio (test set de cada ventana)",
                xaxis_tickangle=-15, margin=dict(l=0, r=0, t=10, b=60)
            )
            st.plotly_chart(fig_wfv, width="stretch")

            st.dataframe(
                df_wfv[['descripcion', 'train_dias', 'test_dias', 'sharpe', 'max_drawdown_pct']]
                .rename(columns={'descripcion': 'Ventana', 'train_dias': 'Días Train',
                                 'test_dias': 'Días Test', 'sharpe': 'Sharpe',
                                 'max_drawdown_pct': 'MaxDD (%)'})
                .set_index('Ventana'),
                width="stretch"
            )

            sharpes_pos = (df_wfv['sharpe'] > 0).sum()
            insight(
                f"<strong>{sharpes_pos}/{len(df_wfv)} ventanas con Sharpe positivo.</strong> "
                "Si las 3 ventanas son positivas, el modelo generaliza en los tres regímenes "
                "de mercado del periodo estudiado."
            )
        else:
            st.info("Ejecuta `walk_forward_validation()` desde train.py o el endpoint "
                    "`/fase3/walk-forward` para generar estos resultados.")

        with st.expander("📖 Walk-forward validation: la prueba más exigente de generalización"):
            st.markdown("""
La validación walk-forward (WFV) es el estándar de facto en backtesting cuantitativo
profesional (Pardo, 2008). Consiste en repetir el ciclo **train → test** múltiples veces,
avanzando la ventana cronológicamente para cubrir distintos regímenes de mercado.
""")
            formula("""
Ventana 1: Train [2014, 2020) → Test [2020, 2022)   ← Volatilidad COVID y recuperación
Ventana 2: Train [2014, 2022) → Test [2022, 2024)   ← Bear market, subidas de tipos Fed
Ventana 3: Train [2014, 2024) → Test [2024, 2026)   ← Normalización macro, rally IA
""")
            st.markdown("""
**Por qué un único split 80/20 no es suficiente:**

Con un único split, el modelo podría obtener un buen Sharpe simplemente porque el 20% de
test resultó ser un periodo alcista fácil. La WFV obliga al modelo a demostrar que funciona
en *al menos tres contextos macroeconómicos distintos*.

**Los tres regímenes que cubre:**
- **2020–2022**: volatilidad extrema del COVID (marzo 2020: −34% en 33 días) seguida de
  la mayor recuperación en V de la historia, luego las primeras subidas de tipos.
- **2022–2024**: bear market real en renta variable y bonos simultáneamente (algo
  rarísimo históricamente), con IBIT colapsando un −75%.
- **2024–2026**: normalización, rally de mercado liderado por IA, Bitcoin superando máximos.

Un modelo que obtiene Sharpe > 0 en las tres ventanas ha demostrado adaptabilidad real.
Un modelo que falla en 2022–2024 (el año más difícil) tiene una debilidad identificada.
""")

else:
    # Estado inicial — sin simulación
    st.info("Configura los parámetros en el panel lateral y pulsa **Lanzar simulación**.")

    st.markdown('<div class="section-title">Dataset cargado</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Activos", len(tickers))
    c2.metric("Features", df_f.shape[1])
    c3.metric("Días históricos", df_f.shape[0])

    st.markdown('<div class="section-title">Activos en cartera</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame({"Ticker": tickers, "Columna precio": df_p.columns.tolist()}),
        width="stretch", hide_index=True
    )
