"""
Servicio de simulación personalizada para el usuario inversor.

Permite a un usuario seleccionar activos, capital y comisión, y ejecutar
un backtest con el modelo PPO entrenado + baselines comparativas.

Reutiliza los mismos componentes que el dashboard Streamlit:
  - PortfolioEnv para el agente PPO
  - run_baselines para las estrategias de referencia
  - compute_metrics para las métricas financieras

La diferencia con el Streamlit es que aquí el usuario elige sus propios
activos y capital, mientras que Streamlit siempre usa los datos del pipeline.

Requisitos:
  - Modelo PPO entrenado (models/best_model_academic/best_model.zip)
  - Datos preparados (data/normalized_features.csv, data/original_prices.csv)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional
from stable_baselines3 import PPO

from src.training_drl.environment_trading import PortfolioEnv
from src.benchmarking.baselines import run_baselines, compute_metrics
from src.auth.models import SessionLocal
from src.auth import universe_repository as universe_repo


def get_available_strategies() -> list[dict]:
    """
    Retorna las estrategias de inversión disponibles para el simulador.

    Cada estrategia tiene un identificador, nombre legible y descripción.
    Solo PPO y Especulativo requieren modelo entrenado.
    """
    model_exists = os.path.exists('models/best_model_academic/best_model.zip')
    spec_exists  = os.path.exists('models/speculative_gmm.pkl')

    return [
        {
            'id': 'ppo',
            'name': 'IA PPO (Deep Reinforcement Learning)',
            'available': model_exists,
            'description': 'Agente entrenado con PPO que optimiza Sharpe rolling con penalización de drawdown y turnover.',
        },
        {
            'id': 'speculative',
            'name': 'Especulativo (GMM + K-Means)',
            'available': spec_exists,
            'description': 'Agente no supervisado que detecta regímenes de mercado y asigna pesos por cluster.',
        },
        {
            'id': 'equal_weight',
            'name': 'Equal Weight Mensual',
            'available': True,
            'description': 'Reparto equitativo entre todos los activos con rebalanceo mensual.',
        },
        {
            'id': 'buy_and_hold',
            'name': 'Buy & Hold',
            'available': True,
            'description': 'Compra inicial equitativa sin rebalanceo posterior.',
        },
        {
            'id': 'portfolio_60_40',
            'name': 'Cartera 60/40',
            'available': True,
            'description': '60% renta variable (primer activo) / 40% renta fija (segundo activo).',
        },
    ]


def run_simulation(capital: float = 10000,
                   commission: float = 0.001,
                   split_pct: float = 0.8) -> dict:
    """
    Ejecuta la simulación completa con el modelo PPO y las baselines.

    Usa los datos ya preparados en data/ y el modelo entrenado en models/.
    No permite elegir activos individuales — usa todo el universo del pipeline.
    Esta restricción existe porque el modelo PPO fue entrenado con un observation
    space específico que depende del número y orden de activos del dataset.

    Para simulaciones con activos personalizados, habría que reentrenar el modelo
    (Fase futura: meta-agente con screener dinámico).

    Parameters
    ----------
    capital    : capital inicial de la simulación
    commission : comisión por operación (0.001 = 0.1%)
    split_pct  : fracción train/test (0.8 = 80% train, 20% test)

    Returns
    -------
    dict con:
      - metrics: dict {estrategia: {sharpe, retorno, mdd, ...}}
      - equity_curves: dict {estrategia: [valores...]}
      - test_period: {start, end, days}
      - weights_ppo: [[w1, w2, ...], ...] pesos del PPO por día
      - tickers: lista de activos
    """
    features_path = 'data/normalized_features.csv'
    prices_path   = 'data/original_prices.csv'
    model_path    = 'models/best_model_academic/best_model.zip'

    # Validar que existen los ficheros necesarios
    if not os.path.exists(features_path):
        return {"error": "Datos no preparados. El administrador debe ejecutar /admin/fase1/preparar-datos"}
    if not os.path.exists(model_path):
        return {"error": "Modelo no entrenado. El administrador debe ejecutar /admin/fase3/entrenar-academico"}

    # Validar que el modelo PPO fue entrenado con el mismo universo que los datos actuales
    db = SessionLocal()
    try:
        validation = universe_repo.validate_model_compatibility(db, model_type="ppo")
        if not validation["compatible"]:
            return {"error": validation["error"]}
    finally:
        db.close()

    # Cargar datos
    df_f = pd.read_csv(features_path, index_col=0)
    df_p = pd.read_csv(prices_path, index_col=0)

    split_idx   = int(len(df_f) * split_pct)
    df_p_test   = df_p.iloc[split_idx:].copy()
    tickers_raw = df_p.columns.tolist()
    tickers     = [t.replace('_Close', '') for t in tickers_raw]

    # ── PPO ──────────────────────────────────────────────────────────────────
    env_test = PortfolioEnv(
        features_path, prices_path,
        start_idx=split_idx,
        commission=commission,
        initial_balance=capital,
    )
    model   = PPO.load(model_path)
    obs, _  = env_test.reset()
    done    = False
    ppo_equity = [capital]
    ppo_weights = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env_test.step(action)
        ppo_equity.append(float(info['value']))
        w = np.clip(action, 0, 1)
        w = w / (w.sum() + 1e-6)
        ppo_weights.append(w.tolist())

    ppo_series = pd.Series(ppo_equity, name='IA_PPO')

    # ── Baselines ────────────────────────────────────────────────────────────
    baseline_results = run_baselines(
        df_p_test,
        initial_balance=capital,
        commission=commission,
        ticker_equity=tickers_raw[0] if tickers_raw else 'IVV_Close',
        ticker_bond=tickers_raw[1] if len(tickers_raw) > 1 else 'BND_Close',
    )

    # ── Especulativo (si existe y es compatible con los datos actuales) ──────
    spec_path = 'models/speculative_gmm.pkl'
    if os.path.exists(spec_path):
        try:
            import pickle
            with open(spec_path, 'rb') as f:
                spec_agent = pickle.load(f)
            df_f_test = df_f.iloc[split_idx:]
            spec_series = spec_agent.backtest(
                df_f_test, df_p_test,
                initial_balance=capital, commission=commission,
            )
            baseline_results['Especulativo_HMM'] = spec_series
        except (KeyError, Exception) as e:
            # El modelo especulativo fue entrenado con un universo de activos distinto
            # al actual (ej. screener generó tickers diferentes). Se omite sin crashear.
            print(f"  [AVISO] Modelo especulativo incompatible con datos actuales: {e}")

    # ── Unir y calcular métricas ─────────────────────────────────────────────
    all_series = {'IA_PPO': ppo_series, **baseline_results}

    metrics = {}
    equity_curves = {}
    for name, series in all_series.items():
        if series is not None and len(series) > 1:
            metrics[name] = compute_metrics(series)
            # Convertir a float nativo Python para evitar errores de serialización JSON
            # (numpy float32/int64 no son JSON serializable por defecto en FastAPI)
            equity_curves[name] = [float(v) for v in series.values]

    # Período de test
    test_start = df_p_test.index[0] if len(df_p_test) > 0 else "N/A"
    test_end   = df_p_test.index[-1] if len(df_p_test) > 0 else "N/A"

    # Fechas del período de test como strings para el eje X de las gráficas.
    # Incluye un día extra al inicio (día anterior al test) para el punto de capital inicial.
    test_dates = [str(d)[:10] for d in df_p_test.index]
    # PPO tiene un punto extra al inicio (capital inicial antes del primer paso)
    if test_dates:
        from pandas.tseries.offsets import BDay
        day_before = str(pd.Timestamp(test_dates[0]) - BDay(1))[:10]
        ppo_dates = [day_before] + test_dates
    else:
        ppo_dates = test_dates

    return {
        "metrics": metrics,
        "equity_curves": equity_curves,
        "dates": test_dates,
        "ppo_dates": ppo_dates,  # 1 punto extra para el capital inicial
        "test_period": {
            "start": str(test_start)[:10],
            "end": str(test_end)[:10],
            "days": len(df_p_test),
        },
        "weights_ppo": ppo_weights,
        "tickers": tickers,
        "capital": capital,
        "commission": commission,
    }
