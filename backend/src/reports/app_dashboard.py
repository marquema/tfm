"""
Dashboard interactivo (Streamlit) del TFM — AI-Driven Portfolio Management.

Qué es este módulo en una frase:
    Es el "panel de control académico" del TFM. Una aplicación web
    Streamlit que permite explorar de forma interactiva los resultados
    del agente PPO contra las baselines, ajustar parámetros (split,
    comisión, capital) y ver el impacto en tiempo real. Diseñado para
    la defensa del TFM — ideal para mostrar al tribunal "en vivo"
    cualquier comparación que pidan.

Diferencia con el frontend Angular:
    El TFM tiene DOS interfaces visuales que coexisten deliberadamente:
      - Streamlit (este módulo): orientado a EXPLORACIÓN académica.
        Permite tocar parámetros y re-evaluar. Pensado para uso del
        admin/investigador y la defensa.
      - Angular (frontend-angular/): orientado a USUARIO FINAL inversor.
        UI pulida, autenticación JWT, consume API REST. Pensado para
        demostrar viabilidad de producto.
    Ambas leen los mismos artefactos (CSVs, modelo PPO, BD) — la
    diferencia es de público objetivo y nivel de profundidad técnica.

Cómo se ejecuta:
    streamlit run src/reports/app_dashboard.py

    Streamlit levanta un servidor local en http://localhost:8501.
    Cualquier widget (slider, selectbox) que el usuario toque dispara
    una re-ejecución completa del script — el patrón de programación
    es declarativo (describes la UI; Streamlit gestiona el ciclo).

Secciones del dashboard (8 paneles):
  1. Métricas comparativas: tabla Sharpe/Sortino/MDD/CAGR/Vol/Retorno
     para todas las estrategias.
  2. Equity curves: evolución del capital de cada estrategia (Plotly).
  3. Drawdown over time: caídas desde máximos por estrategia.
  4. Asset allocation: pie chart de los pesos finales del PPO.
  5. Diagnóstico de entrenamiento: KL, clip fraction, value loss, etc.
  6. Distribución de retornos diarios.
  7. Volatilidad rolling.
  8. Análisis de regímenes (vía regime_analysis.py).

Carga datos directamente de:
  - data/normalized_features.csv y data/original_prices.csv
  - models/best_model_academic/best_model.zip (modelo PPO entrenado)
  - BD (universe_repository) para metadata del modelo (perfil de riesgo).
"""
import os
import sys
# Añadir raíz del proyecto al path para que 'src' sea importable
# cuando se ejecuta con: streamlit run src/reports/app_dashboard.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO, A2C, SAC

ALGO_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
from src.training_drl.environment_trading import PortfolioEnv
from src.pipeline_getdata.asset_registry import get_asset_info, get_display_name, get_tickers

try:
    from src.benchmarking.baselines import ejecutar_baselines, calcular_metricas, tabla_comparativa
except ImportError:
    from src.benchmarking.baselines import ejecutar_baselines, calcular_metricas, tabla_comparativa

from src.training_drl.risk_profiles import RISK_PROFILES


@st.cache_data(ttl=60)
def get_trained_model_info(algo: str = 'ppo'):
    """
    Consulta la BD para recuperar el perfil de riesgo del último modelo entrenado
    del algoritmo solicitado (PPO, A2C o SAC).

    Sobre el cache: `@st.cache_data(ttl=60)` significa que Streamlit
    cachea el resultado durante 60 segundos. Sin esto, cada interacción
    del usuario (mover un slider) dispararía una consulta nueva a la BD
    — innecesario porque la metadata del modelo no cambia segundo a
    segundo. TTL bajo (60s) garantiza que tras reentrenar un modelo, el
    dashboard refleje los nuevos valores casi inmediatamente sin
    requerir restart manual.

    Devuelve dict con: risk_profile, phi, gamma, steps, trained_at.
    Si no hay modelo en BD (o falla la conexión), devuelve None — el
    dashboard sigue funcionando, solo pierde el banner de perfil.
    """
    try:
        from src.auth.models import SessionLocal
        from src.auth import universe_repository as universe_repo
    except Exception:
        return None

    db = SessionLocal()
    try:
        model = universe_repo.get_latest_model(db, model_type=algo.lower())
        if model is None:
            return None
        metrics = model.train_metrics or {}
        profile_id = metrics.get('risk_profile', 'low_turnover')
        profile = RISK_PROFILES.get(profile_id, RISK_PROFILES['low_turnover'])
        return {
            'risk_profile': profile_id,
            'name': profile['name'],
            'phi': profile['phi'],
            'gamma': profile['gamma'],
            'description': profile['description'],
            'steps': model.steps,
            'trained_at': str(model.created_at)[:19] if model.created_at else None,
        }
    except Exception:
        return None
    finally:
        db.close()


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

MODEL_OPTIONS = {
    # ════ PPO honest (universo n=17 post data leakage fix) ═════════════════
    "PPO — LT manual honest (Sharpe 0.469)": {
        "algo": "PPO",
        "path": "models/best_model_academic_low_turnover_sharpe_HONEST_20260511_2230/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "PPO — dual manual honest (Sharpe 0.936)": {
        "algo": "PPO",
        "path": "models/best_model_academic_dual_HONEST_20260511_2331/best_model.zip",
        "profile": "low_turnover",
        "reward": "dual",
    },
    "PPO — Optuna best seed 0 (Sharpe 0.458)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_seed0/best_model.zip",
        "profile": "low_turnover",  # phi=0.0094 from Optuna best
        "reward": "sharpe",
    },
    "PPO — Optuna best seed 1 (Sharpe 0.607)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_seed1/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "PPO — Optuna best seed 2 (Sharpe 0.307)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_seed2/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "PPO — Optuna best seed 3 (Sharpe 0.553)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_seed3/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "PPO — Optuna best seed 4 (Sharpe 0.443)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_seed4/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "PPO — Optuna best + dual reward (Sharpe 0.475)": {
        "algo": "PPO",
        "path": "models/best_model_academic_OPTUNA_dual/best_model.zip",
        "profile": "low_turnover",
        "reward": "dual",
    },
    # ════ A2C honest multi-seed (universo n=17) ════════════════════════════
    "A2C — LT seed 0 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_low_turnover_seed0/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "A2C — LT seed 1 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_low_turnover_seed1/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "A2C — LT seed 2 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_low_turnover_seed2/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "A2C — LT seed 3 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_low_turnover_seed3/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "A2C — LT seed 4 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_low_turnover_seed4/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "A2C — aggressive seed 0 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_aggressive_seed0/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "A2C — aggressive seed 1 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_aggressive_seed1/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "A2C — aggressive seed 2 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_aggressive_seed2/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "A2C — aggressive seed 3 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_aggressive_seed3/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "A2C — aggressive seed 4 (honest)": {
        "algo": "A2C",
        "path": "models/best_model_academic_a2c_aggressive_seed4/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    # ════ SAC honest multi-seed (universo n=17) ════════════════════════════
    "SAC — LT seed 0 (Sharpe 0.532)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_low_turnover_seed0/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "SAC — LT seed 1 (Sharpe 1.191)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_low_turnover_seed1/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "SAC — LT seed 2 (Sharpe 0.712)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_low_turnover_seed2/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "SAC — LT seed 3 (Sharpe 0.701)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_low_turnover_seed3/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "SAC — LT seed 4 (Sharpe 1.280) ★ top": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_low_turnover_seed4/best_model.zip",
        "profile": "low_turnover",
        "reward": "sharpe",
    },
    "SAC — aggressive seed 0 (Sharpe 1.097)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_aggressive_seed0/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "SAC — aggressive seed 1 (Sharpe 0.643)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_aggressive_seed1/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "SAC — aggressive seed 2 (Sharpe 1.000)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_aggressive_seed2/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "SAC — aggressive seed 3 (Sharpe 0.798)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_aggressive_seed3/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    "SAC — aggressive seed 4 (Sharpe 0.805)": {
        "algo": "SAC",
        "path": "models/best_model_academic_sac_aggressive_seed4/best_model.zip",
        "profile": "aggressive",
        "reward": "sharpe",
    },
    # ════ Modelos viejos n=15 data leakage (NO compatibles env actual) ══════
    # Mantengo apuntadores comentados como referencia histórica:
    # - models/best_model_academic_low_turnover_sharpe/  (Sharpe 1.875 leakage)
    # - models/best_model_academic_aggressive/           (Sharpe 1.650 leakage)
    # - models/best_model_academic_dual/                 (Sharpe 1.624 leakage)
    # - models/best_model_academic_a2c_calibrated/       (n=15)
    # - models/best_model_academic_sac_BACKUP_TFM/       (n=15)
    # Para usarlos: revertir backend/data/*.csv a backups _PRE_RETRAIN_*.csv
}
model_label = st.sidebar.selectbox(
    "Modelo a evaluar",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
)
_sel = MODEL_OPTIONS[model_label]
model_algo    = _sel["algo"]
model_path    = _sel["path"]
model_profile = _sel["profile"]
model_reward  = _sel["reward"]
commission  = st.sidebar.slider("Comisión por operación (%)", 0.0, 0.5, 0.1) / 100
initial_bal = st.sidebar.number_input("Capital inicial ($)", value=10000, step=1000)
split_pct   = st.sidebar.slider("Split train/test (%)", 60, 90, 80) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**Universo de activos**")


def _real_universe_tickers() -> list:
    """
    Lee los tickers reales del CSV de precios actual. Refleja exactamente
    el universo con el que se entreno el modelo PPO actualmente desplegado,
    en lugar de la lista estatica CORE_UNIVERSE (que es solo fallback).
    """
    prices_path = 'data/original_prices.csv'
    if not os.path.exists(prices_path):
        return get_tickers('core')  # fallback si aun no se ha preparado data
    try:
        cols = pd.read_csv(prices_path, index_col=0, nrows=0).columns.tolist()
        return [c.replace('_Close', '') for c in cols]
    except Exception:
        return get_tickers('core')


_real_tickers = _real_universe_tickers()
with st.sidebar.expander(f"Ver activos del portfolio ({len(_real_tickers)})"):
    for t in _real_tickers:
        info = get_asset_info(t)
        st.sidebar.caption(f"**{t}** — {info['name']}  \n{info['category']} · {info['sector']}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Estado del sistema**")
features_ok = os.path.exists('data/normalized_features.csv')
prices_ok   = os.path.exists('data/original_prices.csv')
model_ok    = os.path.exists(model_path)
st.sidebar.markdown(f"{'✅' if features_ok else '❌'} Features CSV")
st.sidebar.markdown(f"{'✅' if prices_ok   else '❌'} Precios CSV")
st.sidebar.markdown(f"{'✅' if model_ok    else '❌'} Modelo {model_algo}")

# ─── Perfil de riesgo del modelo seleccionado ────────────────────────────────
# Metadata viene del propio MODEL_OPTIONS (atada al fichero), no de la BD,
# porque la BD solo guarda el último registro por algoritmo y los backups
# anteriores de PPO/A2C apuntarían a metadata equivocada.
_profile_def = RISK_PROFILES.get(model_profile, RISK_PROFILES['low_turnover'])
model_info = {
    'risk_profile': model_profile,
    'name': _profile_def['name'],
    'phi': _profile_def['phi'],
    'gamma': _profile_def['gamma'],
    'description': _profile_def['description'],
    'reward_type': model_reward,
    'algorithm': model_algo,
}
st.sidebar.markdown("---")
st.sidebar.markdown("**Modelo cargado**")
st.sidebar.markdown(
    f"🎯 **{model_info['algorithm']}** · **{model_info['name']}** (`{model_info['risk_profile']}`)"
)
st.sidebar.caption(
    f"φ = {model_info['phi']} · γ = {model_info['gamma']}  \n"
    f"Recompensa: `{model_info['reward_type']}`"
)


# ─── Carga de datos ───────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """
    Carga los CSVs de features normalizadas y precios originales.

    Retorna una tupla (df_features, df_prices) con ambos DataFrames indexados
    por la primera columna del CSV.
    """
    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv',     index_col=0)
    return df_f, df_p

if not (features_ok and prices_ok):
    st.error("Faltan archivos de datos. Ejecuta primero POST /fase1/preparar-datos.")
    st.stop()

df_f, df_p = load_data()
split_idx  = int(len(df_f) * split_pct)
df_p_test  = df_p.iloc[split_idx:].copy()
tickers_raw = df_p.columns.tolist()                                    # ['IVV_Close', 'BND_Close', ...]
tickers     = [t.replace('_Close', '') for t in tickers_raw]           # ['IVV', 'BND', ...]
ticker_labels = [get_display_name(t) for t in tickers]                 # ['IVV — iShares Core S&P 500 ETF', ...]

st.info(
    f"Período de test: **{df_p_test.index[0][:10]}** → **{df_p_test.index[-1][:10]}**"
    f" | {len(df_p_test)} días de trading | {len(tickers)} activos"
)

# Banner con el perfil de riesgo del modelo cargado
col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 2])
col_a.metric("Algoritmo · Perfil", f"{model_info['algorithm']} · {model_info['name']}")
col_b.metric("φ (drawdown)", model_info['phi'])
col_c.metric("γ (turnover)", model_info['gamma'])
col_d.metric("Recompensa", model_info['reward_type'])
with st.expander("¿Qué significa este perfil?"):
    st.markdown(model_info['description'])


# ─── Constantes visuales ──────────────────────────────────────────────────────
COLORES = {
    'IA_PPO':               '#00d4ff',
    'Especulativo_HMM':     '#ff9f1c',
    'Equal_Weight_Mensual': '#f0a500',
    'Buy_and_Hold':         '#7ed957',
    'Cartera_60_40':        '#ff6b6b',
    'Markowitz_MV':         '#c77dff',
    'Random_Uniform':       '#8d99ae',  # gris azulado (lower bound)
    'Momentum_TopK':        '#ef476f',  # rosa fuerte (factor competitivo)
}

NOMBRES = {
    'IA_PPO':               f'IA {model_algo} (DRL)',  # algo dinámico según selección
    'Especulativo_HMM':     'Especulativo (GMM+KMeans)',
    'Equal_Weight_Mensual': 'Equal Weight',
    'Buy_and_Hold':         'Buy & Hold',
    'Cartera_60_40':        'Cartera 60/40',
    'Markowitz_MV':         'Markowitz MV',
    'Random_Uniform':       'Random Uniform',
    'Momentum_TopK':        'Momentum Top-3 (60d)',
}

DESCRIPCIONES_METRICAS = {
    'Retorno Total (%)': (
        'Cuánto creció (o cayó) el capital desde el inicio hasta el final del período de test. '
        'Un 35% significa que $10.000 se convirtieron en $13.500. '
        'Es la métrica más intuitiva, pero engañosa sola: un 35% con bajadas del 50% por el camino '
        'es muy distinto a un 35% con una curva suave.'
    ),
    'CAGR (%)': (
        'Tasa de Crecimiento Anual Compuesto (Compound Annual Growth Rate). '
        'Responde a: "¿a qué ritmo anual creció esta cartera, como si fuera constante?". '
        'Si el test dura 2 años y el retorno total es 35%, el CAGR es ~16% anual. '
        'Permite comparar estrategias evaluadas en períodos de distinta duración.'
    ),
    'Volatilidad Anualizada (%)': (
        'Cuánto oscila el valor de la cartera de un día para otro, expresado en términos anuales. '
        'Se calcula como la desviación estándar de los retornos diarios multiplicada por √252 '
        '(días hábiles en un año). Una volatilidad del 20% significa que en un año típico '
        'la cartera puede subir o bajar ~20% solo por la variabilidad diaria. '
        'Alta volatilidad no es sinónimo de pérdida, pero sí de mayor incertidumbre.'
    ),
    'Sharpe Ratio': (
        'La métrica más usada en gestión de carteras profesional. '
        'Mide cuánto retorno se obtiene por cada unidad de riesgo asumido: '
        '(retorno_anual − tasa_libre_de_riesgo) / volatilidad_anual. '
        'Un Sharpe de 1.0 significa que por cada punto de volatilidad se obtiene un punto de retorno. '
        'Por debajo de 0.5 es mediocre. Entre 1 y 2 es bueno. Por encima de 2 es excepcional. '
        'Una cartera con menos retorno pero mucho menos riesgo puede tener mejor Sharpe que otra con más retorno.'
    ),
    'Sortino Ratio': (
        'Variante del Sharpe más justa para estrategias asimétricas. '
        'La diferencia clave: el Sharpe penaliza toda la volatilidad (incluyendo los días buenos), '
        'mientras que el Sortino solo penaliza la volatilidad negativa — las caídas. '
        'Una estrategia que tiene muchos días muy positivos y pocos negativos pequeños '
        'tendrá Sharpe modesto pero Sortino alto. '
        'Si el Sortino es mucho mayor que el Sharpe, la estrategia tiene retornos asimétricos hacia arriba.'
    ),
    'Max Drawdown (%)': (
        'La peor caída que habría sufrido un inversor que entrase en el peor momento posible. '
        'Se calcula como la mayor caída desde cualquier máximo histórico hasta el siguiente mínimo. '
        'Un MDD de −20% significa que en algún momento la cartera valía un 20% menos que en su pico previo. '
        'Es la métrica que más duele psicológicamente: un inversor real habría visto esfumarse '
        'ese porcentaje de su dinero antes de recuperarse. '
        'Cuanto más cerca de 0, más estable y fácil de mantener la estrategia.'
    ),
    'Valor Final ($)': (
        'Capital total al cierre del período de test, incluyendo comisiones y reinversión de beneficios. '
        'Depende directamente del capital inicial configurado en el panel lateral.'
    ),
}


# Diccionario de estrategias mostrado en el segundo glosario del dashboard.
# Describe que hace cada baseline y por que esta en la comparativa, para que
# un evaluador del TFM o un usuario no tecnico entienda con que se compara
# al agente PPO.
DESCRIPCIONES_ESTRATEGIAS = {
    'IA PPO (DRL)': (
        'Agente de Deep Reinforcement Learning entrenado con Proximal Policy Optimization (PPO). '
        'Aprende a asignar pesos del 0 al 100 % a cada activo del universo en funcion del estado de mercado, '
        'optimizando una recompensa que combina Sharpe rolling 20d, penalizacion por Maximum Drawdown y por turnover. '
        'Es la propuesta central del TFM.'
    ),
    'Equal Weight': (
        'Cartera equiponderada (1/N) sobre el universo, con rebalanceo mensual. '
        'Es la baseline pasiva clasica: \"reparte el capital igual entre todos los activos y olvidate\". '
        'Sin pretensiones de pronostico, solo diversificacion mecanica.'
    ),
    'Buy & Hold': (
        'Compra inicial 1/N y nunca rebalancea. El activo mas rentable va ganando peso progresivamente. '
        'Baseline de minima friccion: cero costes de transaccion mas alla de la compra inicial. '
        'En mercados alcistas amplios, es muy dificil de batir.'
    ),
    'Cartera 60/40': (
        'Cartera institucional canonica: 60 % renta variable (IVV, S&P 500) + 40 % renta fija (BND, bonos USA). '
        'Es el benchmark de referencia en gestion patrimonial conservadora desde decadas. '
        'Rebalanceo mensual al 60/40 objetivo.'
    ),
    'Markowitz MV': (
        'Optimizacion media-varianza clasica de Markowitz (1952). En cada inicio de mes mira los retornos '
        'y correlaciones de los ultimos 12 meses, calcula la combinacion de pesos que maximiza el ratio '
        'de Sharpe esperado, y rebalancea a esos pesos. Es el baseline teorico financiero.'
    ),
    'Random Uniform': (
        'Cartera con pesos aleatorios uniformes sobre el simplex (Dirichlet con todos los alfas iguales a 1), '
        'rebalanceo mensual con semilla fija para reproducibilidad. '
        'Es el "lower bound de cordura" del estudio: si el agente PPO no supera consistentemente a esta '
        'estrategia, no esta aprendiendo nada que el azar no haga.'
    ),
    'Momentum Top-3 (60d)': (
        'Cada mes selecciona los 3 activos con mejor retorno acumulado en los ultimos 60 dias y les asigna '
        'pesos equiponderados (33 % cada uno), poniendo el resto a cero. Implementa el factor "momentum '
        'cross-sectional" documentado por Jegadeesh & Titman (1993). Es una baseline competitiva del sector '
        'cuantitativo: si PPO no la bate, no aporta valor sobre un factor financiero conocido y trivial.'
    ),
    'Especulativo (GMM+KMeans)': (
        'Agente no supervisado de contraste: detecta el regimen de mercado con un Gaussian Mixture Model '
        '(3 estados) y agrupa los activos en clusters por momentum con K-Means. Una matriz heuristica fija '
        'asigna pesos por cluster segun el regimen detectado. No aprende por reward — es un agente '
        'estructurado fijo, util para responder a la pregunta: "ipara este problema hace falta DRL o basta '
        'con segmentacion clasica?".'
    ),
}

LAYOUT_OSCURO = dict(template='plotly_dark', hovermode='x unified',
                     legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))


# ─── Backtest ────────────────────────────────────────────────────────────────
if st.button("▶  Ejecutar Backtest Completo", type="primary", use_container_width=True):

    if not model_ok:
        st.error(f"Modelo no encontrado en {model_path}. Entrena con POST /fase3/entrenar-academico")
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
        model = ALGO_CLASSES[model_algo].load(model_path)
        obs, _ = env_test.reset()
        done   = False
        ppo_equity      = [initial_bal]
        weight_history  = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env_test.step(action)
            ppo_equity.append(info['value'])
            w = np.clip(action, 0, 1)
            w = w / (w.sum() + 1e-6)
            weight_history.append(w)

        ppo_series   = pd.Series(ppo_equity, name='IA_PPO')

        # ── Baselines 
        baseline_results = ejecutar_baselines(
            df_p_test,
            initial_balance=initial_bal,
            commission=commission,
            ticker_equity='IVV_Close',
            ticker_bond='BND_Close'
        )

        # ── Agente Especulativo (GMM + K-Means) 
        speculative_path = 'models/speculative_gmm.pkl'
        if os.path.exists(speculative_path):
            import pickle
            try:
                with open(speculative_path, 'rb') as f:
                    spec_agent = pickle.load(f)
                df_f_test = df_f.iloc[split_idx:]
                spec_series = spec_agent.backtest(
                    df_f_test, df_p_test,
                    initial_balance=initial_bal, commission=commission
                )
                baseline_results['Especulativo_HMM'] = spec_series
            except (KeyError, ValueError) as e:
                # HMM detector entrenado con universo n=15 (data leakage).
                # No compatible con universo honest n=17. Skip silente para
                # no romper dashboard. Re-entrenar HMM con universo nuevo
                # queda como mantenimiento opcional.
                st.warning(f'Baseline GMM+HMM omitida (detector incompatible con universo actual): {type(e).__name__}')

        all_series  = {'IA_PPO': ppo_series, **baseline_results}
        df_metrics  = tabla_comparativa(all_series)

    # Construir eje de fechas: las series tienen 1 punto extra al inicio (balance inicial)
    # Se añade un día hábil anterior al test como "día 0" para ese punto.
    test_dates = pd.to_datetime(df_p_test.index)
    from pandas.tseries.offsets import BDay
    date_d0 = test_dates[0] - BDay(1)
    dates   = [date_d0] + test_dates.tolist()

    # ═════════════════════════════════
    # SECCIÓN 1: Métricas comparativas
    # ═════════════════════════════════
    st.markdown("---")
    st.markdown("## 1. Métricas de Rendimiento")
    st.markdown(
        "Todas las estrategias se evalúan sobre el mismo período de test — "
        "datos que el agente PPO **nunca vio** durante el entrenamiento. "
        "La comparación es justa: mismo capital inicial, mismas comisiones, mismo período."
    )
    st.markdown(
        "**¿Qué mirar primero?** El **Sharpe Ratio** resume la calidad de la estrategia en un solo número: "
        "cuánto retorno se obtuvo por cada unidad de riesgo asumido. "
        "El **Max Drawdown** dice cuánto habrías perdido en el peor momento. "
        "Una estrategia con Sharpe alto y MDD bajo es la que un inversor real puede mantener sin entrar en pánico."
    )

    # Cards en filas de 4 para que las etiquetas y valores sean legibles
    # incluso con 7-8 estrategias (con 8 columnas en una sola fila el
    # texto se truncaba a "Sharpe..." y no se distinguian las estrategias).
    estrategias = list(df_metrics.index)
    CARDS_POR_FILA = 4
    for i in range(0, len(estrategias), CARDS_POR_FILA):
        fila = estrategias[i:i + CARDS_POR_FILA]
        cols = st.columns(CARDS_POR_FILA)
        for col, name in zip(cols, fila):
            sharpe    = df_metrics.loc[name, 'Sharpe Ratio']
            total_ret = df_metrics.loc[name, 'Retorno Total (%)']
            mdd       = df_metrics.loc[name, 'Max Drawdown (%)']
            col.metric(
                label=NOMBRES.get(name, name),
                value=f"Sharpe {sharpe:.2f}",
                delta=f"Ret {total_ret:.1f}%  |  MDD {mdd:.1f}%",
                delta_color="normal" if total_ret >= 0 else "inverse"
            )

    st.markdown("### Tabla completa")
    with st.expander("📖  Glosario — qué significa cada métrica"):
        for metric_name, desc in DESCRIPCIONES_METRICAS.items():
            st.markdown(f"**{metric_name}**")
            st.markdown(f"> {desc}")
            st.markdown("")

    with st.expander("🧭  Glosario — qué hace cada estrategia"):
        for strat_name, desc in DESCRIPCIONES_ESTRATEGIAS.items():
            st.markdown(f"**{strat_name}**")
            st.markdown(f"> {desc}")
            st.markdown("")

    # Renombrar fila del agente al algoritmo seleccionado para que la tabla
    # refleje qué se está evaluando (no siempre PPO).
    agent_label = f'IA_{model_algo}'
    df_metrics_display = df_metrics.rename(index={'IA_PPO': agent_label})

    def highlight_ppo(row):
        """
        Resalta la fila del agente DRL en la tabla de métricas con un estilo
        de fondo azul oscuro para distinguirla visualmente de los baselines.
        """
        return ['background-color: #1e3a5f; color: white; font-weight: bold'
                if row.name == agent_label else '' for _ in row]

    st.dataframe(
        df_metrics_display.style.apply(highlight_ppo, axis=1).format("{:.2f}"),
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
    for name, series in all_series.items():
        fig_eq.add_trace(go.Scatter(
            x=dates[:len(series)],
            y=series.values,
            name=NOMBRES.get(name, name),
            line=dict(color=COLORES.get(name, '#aaa'), width=3 if name == 'IA_PPO' else 1.5),
            hovertemplate=f"<b>{NOMBRES.get(name, name)}</b><br>%{{x|%d %b %Y}}: $%{{y:,.2f}}<extra></extra>"
        ))
    fig_eq.add_hline(y=initial_bal, line_dash="dash", line_color="white",
                     opacity=0.3, annotation_text="Capital inicial", annotation_font_color="white")
    fig_eq.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha", yaxis_title="Valor ($)", height=450)
    st.plotly_chart(fig_eq, use_container_width=True)

    # ════════════════════
    # SECCIÓN 3: Drawdown
    # ═════════════════════
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
    for name, series in all_series.items():
        rolling_max = series.cummax()
        dd = (series - rolling_max) / (rolling_max + 1e-8) * 100
        fig_dd.add_trace(go.Scatter(
            x=dates[:len(dd)],
            y=dd.values,
            name=NOMBRES.get(name, name),
            line=dict(color=COLORES.get(name, '#aaa'), width=1.5),
            fill='tozeroy',
            hovertemplate=f"<b>{NOMBRES.get(name, name)}</b><br>%{{x|%d %b %Y}}: %{{y:.2f}}%<extra></extra>"
        ))
    fig_dd.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                         yaxis_title="Drawdown (%)", yaxis_ticksuffix="%", height=350)
    st.plotly_chart(fig_dd, use_container_width=True)

    # ═════════════════════════════════
    # SECCIÓN 4: Asset Allocation
    # ═════════════════════════════════
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
        last_w = np.array(weight_history[-1]).flatten()
        if len(last_w) == len(tickers):
            fig_pie = go.Figure(go.Pie(
                labels=ticker_labels, values=last_w, hole=0.35,
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
            "- **Franjas muy cambiantes** = alta rotación, lo que mata el rendimiento con comisiones"
        )
        st.caption("Clic en la leyenda para aislar un activo concreto")
        if weight_history and len(weight_history[0]) == len(tickers):
            df_w = pd.DataFrame(weight_history, columns=tickers,
                                index=pd.to_datetime(df_p_test.index[:len(weight_history)]))
            fig_w = go.Figure()
            for ticker, label in zip(tickers, ticker_labels):
                fig_w.add_trace(go.Scatter(
                    x=df_w.index, y=df_w[ticker].values,
                    name=label, stackgroup='one',
                    hovertemplate=f"<b>{label}</b> %{{x|%d %b %Y}}: %{{y:.1%}}<extra></extra>"
                ))
            fig_w.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                                yaxis_title="Peso", yaxis_tickformat=".0%", height=350)
            st.plotly_chart(fig_w, use_container_width=True)

    # ═══════════════════════════════════════════
    # SECCIÓN 5: Diagnóstico del Entrenamiento
    # ═══════════════════════════════════════════
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
- Ambas curvas suben juntas → el agente generalizó, funcionará en datos nuevos
- Train sube pero Eval no → **sobreajuste**: memorizó el pasado pero no aprendió nada transferible
- El sistema guarda automáticamente el modelo en el momento donde Eval es máximo, antes de que empiece a degradarse

---

### Gráfica 3 — ¿Funciona en distintos períodos de mercado?

Esta gráfica responde a: "¿los buenos resultados del backtest son casualidad o son robustos?"

Se divide toda la historia disponible en varias ventanas temporales. En cada una, el agente
se entrena desde cero con datos anteriores y se evalúa en el período siguiente que nunca vio.
Es el equivalente financiero del **k-fold cross-validation** en machine learning.

**¿Qué buscar?**
- Sharpe positivo y consistente en la mayoría de ventanas -> la estrategia funciona en distintos regímenes (crisis, rally, consolidación)
- Alta varianza entre ventanas -> el rendimiento depende de qué período toque: suerte, no habilidad
- Sharpe decreciente en las ventanas más recientes -> el modelo está sesgado hacia el pasado lejano

---

### Gráfica 4 — Expanding Window: ¿Mejora con más datos?

Similar a la gráfica 3, pero con una diferencia clave: el entrenamiento **empieza siempre
desde el primer día** del dataset y crece progresivamente. Cada ventana usa TODA la
historia disponible hasta ese momento para entrenar, y evalúa en los 3 meses siguientes.

Esto simula lo que se haría en producción real: "uso todo lo que sé hasta hoy para
predecir mañana". A medida que el agente ve más historia, debería mejorar.

**¿Qué buscar?**
- Sharpe que **mejora o se mantiene** conforme avanzan las ventanas -> el modelo aprovecha tener más datos
- Sharpe que empeora con más datos -> los datos antiguos confunden al modelo (el mercado cambió estructuralmente)
- Comparar con la gráfica 3 (rolling): si expanding da mejores resultados medios, usar toda la historia es beneficioso

**La tabla debajo de cada gráfica** detalla los períodos exactos de train y test
de cada ventana, para que puedas cruzarlos con eventos de mercado conocidos
(COVID 2020, bear market 2022, rally 2023, corrección cripto 2025...).
        """)

    # PNG paths algo-aware: cada algoritmo tiene su panel de diagnóstico
    # nativo con métricas específicas. PPO conserva los nombres clásicos
    # sin sufijo por compatibilidad; A2C/SAC usan sufijo _{algo}.
    if model_algo == 'PPO':
        diag_path = 'src/reports/training_diagnostics.png'
        overfit_path = 'src/reports/overfitting_analysis.png'
    else:
        diag_path = f'src/reports/training_diagnostics_{model_algo.lower()}.png'
        overfit_path = f'src/reports/overfitting_analysis_{model_algo.lower()}.png'

    col_r1, col_r2 = st.columns(2)
    diag_desc_map = {
        'PPO': "Entropía, value loss, explained variance, KL y clip fraction. "
               "Confirman que el algoritmo PPO convergió de forma estable.",
        'A2C': "Entropía, policy loss, value loss, explained variance y learning rate. "
               "Métricas nativas de A2C (sin clipping, sin KL contra política antigua).",
        'SAC': "Actor loss, critic loss (Q-MSE), coeficiente de entropía adaptativo "
               "y learning rate. Métricas nativas off-policy de SAC.",
    }
    diag_desc = diag_desc_map.get(model_algo, "Diagnóstico interno del algoritmo.")
    for col, file_path, title, desc in [
        (col_r1, diag_path,
         "1. Salud del entrenamiento",
         diag_desc),
        (col_r2, overfit_path,
         "2. ¿Memorizó o aprendió?",
         "Reward en datos de train vs datos de evaluación nunca vistos. "
         "Si las dos curvas van juntas, el agente generalizó correctamente."),
    ]:
        col.markdown(f"**{title}**")
        col.caption(desc)
        if os.path.exists(file_path):
            col.image(file_path, use_container_width=True)
        else:
            col.info(f"Pendiente de generar.\n\n`{file_path}`")

    # Walk-Forward y Expanding Window apilados a ancho completo. Antes
    # estaban en col_wf1/col_wf2 (mitad pantalla cada uno) y la tabla de
    # detalle quedaba ilegible. Apilados ocupan todo el ancho disponible
    # y la tabla embebida en el PNG es legible.

    st.markdown("**3. Walk-Forward (Rolling Window)**")
    st.caption(
        "Ventana de entrenamiento fija que se desliza en el tiempo. "
        "Cada ventana entrena con los últimos N días y evalúa en los siguientes. "
        "Muestra si la estrategia funciona en distintos regímenes de mercado."
    )
    wf_png = 'src/reports/walk_forward_analysis.png'
    if os.path.exists(wf_png):
        st.image(wf_png, use_container_width=True)
    else:
        st.info("Pendiente. Ejecuta POST /admin/fase3/walk-forward")

    st.markdown("")  # separador visual

    st.markdown("**4. Expanding Window**")
    st.caption(
        "El entrenamiento empieza siempre desde el primer día y crece progresivamente. "
        "Cada ventana usa TODA la historia disponible hasta ese momento y evalúa "
        "en los siguientes 3 meses. Simula lo que harías en producción: "
        "\"uso todo lo que sé hasta hoy para predecir mañana\"."
    )
    ew_png = 'src/reports/expanding_window_analysis.png'
    if os.path.exists(ew_png):
        st.image(ew_png, use_container_width=True)
    else:
        st.info("Pendiente. Ejecuta POST /admin/fase3/expanding-window")

    # ════════════════════════════════════
    # SECCIÓN 6: Retornos Diarios del PPO
    # ═══════════════════════════════════
    st.markdown("---")
    st.markdown("## 6. Retornos Diarios del Agente PPO")
    st.markdown(
        "Cada barra muestra el retorno porcentual de la cartera PPO en un día concreto. "
        "**Verde** = día positivo, **rojo** = día negativo. "
        "Una distribución con muchas barras verdes pequeñas y pocas rojas grandes indica "
        "una estrategia con **asimetría positiva** (Sortino > Sharpe). "
        "Si predominan las rojas, el agente está asumiendo demasiado riesgo."
    )

    ppo_series = all_series.get('IA_PPO')
    if ppo_series is not None and len(ppo_series) > 2:
        rets = ppo_series.pct_change().dropna()
        colors = ['#00e676' if r >= 0 else '#ff5252' for r in rets]
        # Generar fechas para el eje X
        ret_dates = dates[1:len(rets)+1] if len(dates) > len(rets) else list(range(len(rets)))

        fig_ret = go.Figure()
        fig_ret.add_trace(go.Bar(
            x=ret_dates, y=rets.values * 100,
            marker_color=colors,
            hovertemplate='%{x|%d %b %Y}<br>Retorno: %{y:.2f}%<extra></extra>',
        ))
        fig_ret.add_hline(y=0, line_color="white", opacity=0.3)
        fig_ret.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                              yaxis_title="Retorno diario (%)", yaxis_ticksuffix="%",
                              height=350, bargap=0.1)
        st.plotly_chart(fig_ret, use_container_width=True)

    # ═══════════════════════════════════════════
    # SECCIÓN 7: Volatilidad Rolling Comparativa
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 7. Volatilidad Rolling (20 días, anualizada)")
    st.markdown(
        "Cuánto oscila el valor de cada estrategia en una ventana de 20 días, "
        "expresado en términos anuales. Picos de volatilidad coinciden con eventos de mercado "
        "(correcciones, crisis). El PPO debería mostrar **menor volatilidad que Buy & Hold** "
        "si aprendió a gestionar el riesgo. "
        "**Haz clic en la leyenda** para aislar estrategias."
    )

    fig_vol = go.Figure()
    for name, series in all_series.items():
        if series is None or len(series) < 25:
            continue
        rets_s = series.pct_change().dropna()
        rolling_vol = rets_s.rolling(20).std() * (252 ** 0.5) * 100
        vol_dates = dates[1:len(rolling_vol)+1] if len(dates) > len(rolling_vol) else list(range(len(rolling_vol)))
        fig_vol.add_trace(go.Scatter(
            x=vol_dates[:len(rolling_vol)],
            y=rolling_vol.values,
            name=NOMBRES.get(name, name),
            line=dict(color=COLORES.get(name, '#aaa'), width=2 if name == 'IA_PPO' else 1.2),
            hovertemplate=f"<b>{NOMBRES.get(name, name)}</b><br>%{{x|%d %b %Y}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_vol.update_layout(**LAYOUT_OSCURO, xaxis_title="Fecha",
                          yaxis_title="Volatilidad (%)", yaxis_ticksuffix="%",
                          height=400)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ════════════════════════════════════════════════
    # SECCIÓN 8: Análisis de Regímenes de Volatilidad
    # ════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 8. Análisis de Regímenes de Volatilidad")
    st.markdown(
        "Clasifica cada día del período de test en un régimen de volatilidad: "
        "**calma** (verde), **transición** (naranja) o **crisis** (rojo). "
        "La clasificación se basa en la volatilidad rolling de 20 días del activo de referencia (IVV) "
        "comparada con percentiles del período de entrenamiento. "
        "Permite evaluar si el agente PPO se comporta de forma diferente en cada régimen."
    )

    regimes_png = 'src/reports/regime_analysis.png'
    if os.path.exists(regimes_png):
        st.image(regimes_png, use_container_width=True)
    else:
        # Intentar generar el análisis de regímenes al vuelo
        try:
            from src.training_drl.regime_analysis import analyze_regimes
            if model_ok:
                with st.spinner("Generando análisis de regímenes..."):
                    analyze_regimes(
                        features_path='data/normalized_features.csv',
                        prices_path='data/original_prices.csv',
                        model_path=model_path,
                    )
                if os.path.exists(regimes_png):
                    st.image(regimes_png, use_container_width=True)
                else:
                    st.info("No se pudo generar el análisis de regímenes.")
            else:
                st.info("Se necesita un modelo entrenado para el análisis de regímenes.")
        except Exception as e:
            st.error(f"Error al generar análisis de regímenes: {e}")
            import traceback
            st.code(traceback.format_exc())
