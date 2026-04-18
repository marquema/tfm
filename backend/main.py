"""
API principal del TFM — AI-Driven Portfolio Management.

Arquitectura de routers:
  - /auth/*      → login, registro, gestión de usuarios (público + admin)
  - /admin/*     → screener, datos, entrenamiento, modelos (solo admin)
  - /investor/*  → simulación personalizada (solo inversor/admin)
  - /universo    → diccionario de activos (público)
  - /estado      → estado del sistema (público)
"""

from fastapi import FastAPI, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
import numpy as np

from src.pipeline_getdata.data_downloader import descargar_dividendos, generar_dataset
from src.training_drl.environment_trading import PortfolioEnv
from src.training_drl.training_analysis import (
    entrenar_academico, walk_forward_validation, expanding_window_validation
)
from src.training_drl.sensitivity_analysis import run_sensitivity_analysis
from src.unsupervised.speculative_agent import SpeculativeAgent
from src.pipeline_getdata.asset_registry import get_universe
from src.pipeline_getdata.market_screener import MarketScreener
from src.pipeline_getdata.universe_config import save_config  # legacy JSON (fallback)

from sqlalchemy.orm import Session
from src.auth.models import init_db, get_db, User, SessionLocal
from src.auth.auth_router import router as auth_router
from src.auth.auth_service import require_admin, get_current_user
from src.auth import universe_repository as universe_repo
from src.investor.investor_router import router as investor_router


# ─── App y middleware ─────────────────────────────────────────────────────────

app = FastAPI(title="TFM Trading AI API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:4201"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Inicialización ──────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    """
    Al arrancar la API:
      1. Crea las tablas de la base de datos si no existen
      2. Crea un usuario admin por defecto si no hay ninguno
    """
    init_db()

    # Crear admin por defecto si la tabla está vacía
    from src.auth.models import SessionLocal, User
    from src.auth.auth_service import hash_password
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                email="admin@tfm.com",
                hashed_pwd=hash_password("admin123"),
                full_name="Administrador TFM",
                role="admin",
            )
            db.add(admin)
            db.commit()
            print("  Usuario admin creado: admin@tfm.com / admin123")
    finally:
        db.close()


# ─── Montar routers ──────────────────────────────────────────────────────────

app.include_router(auth_router)
app.include_router(investor_router)


# ─── Modelos de datos ────────────────────────────────────────────────────────

class DownloadConfig(BaseModel):
    tickers: Optional[List[str]] = None  # None = usa último screener o CORE_UNIVERSE
    start: str = "2014-01-01"
    end: str = "2026-03-01"


# ─── Helpers para background tasks ────────────────────────────────────────────

def _create_lock(path: str):
    """Crea un fichero de lock para señalizar que un background task está corriendo."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('running')

def _remove_lock(path: str):
    """Elimina el fichero de lock cuando el background task termina."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# ─── ENDPOINTS PÚBLICOS ──────────────────────────────────────────────────────

@app.get("/universo", tags=["Público"])
async def get_universe_endpoint(level: str = 'core'):
    """Diccionario de activos con metadatos completos."""
    df = get_universe(level)
    return df.reset_index().to_dict(orient='records')


@app.get("/screener/last", tags=["Público"])
async def get_last_screener(db: Session = Depends(get_db)):
    """
    Retorna el último screener activo con sus métricas por activo.

    A diferencia de /universo (que devuelve el CORE_UNIVERSE hardcodeado),
    este endpoint refleja los candidatos reales seleccionados por el
    screener — incluyendo sector, Sharpe rolling, volumen y volatilidad
    calculados sobre los datos de mercado en el periodo usado.
    """
    screener = universe_repo.get_active_screener(db)
    if screener is None:
        return {"available": False, "candidates": [], "details": []}

    return {
        "available": True,
        "candidates": screener.candidates or [],
        "n_candidates": screener.n_candidates,
        "start_date": screener.start_date,
        "end_date": screener.end_date,
        "filters": screener.filters_used,
        "details": screener.details or [],
        "created_at": str(screener.created_at)[:19] if screener.created_at else None,
        "created_by": screener.created_by,
    }


@app.get("/walk-forward/results", tags=["Público"])
async def get_walk_forward_results():
    """
    Retorna los resultados del último walk-forward ejecutado.

    Incluye métricas por ventana con fechas de train y test para
    que el frontend pueda mostrar gráficas con períodos reales.
    """
    csv_path = 'src/reports/walk_forward_results.csv'
    if not os.path.exists(csv_path):
        return {"available": False, "error": "Walk-forward no ejecutado aún."}

    df = pd.read_csv(csv_path, index_col=0)
    windows = df.to_dict(orient='records')
    return {
        "available": True,
        "n_windows": len(windows),
        "windows": windows,
        "summary": {
            "sharpe_mean": round(df['Sharpe Ratio'].mean(), 3),
            "sharpe_std": round(df['Sharpe Ratio'].std(), 3),
            "retorno_mean": round(df['Retorno Total (%)'].mean(), 1),
            "mdd_mean": round(df['Max Drawdown (%)'].mean(), 1),
            "windows_positive_sharpe": int((df['Sharpe Ratio'] > 0).sum()),
        }
    }


@app.get("/resultados/tabla-final", tags=["Público"])
async def get_final_table(db: Session = Depends(get_db)):
    """
    Tabla conclusiva del TFM: PPO vs todas las estrategias.

    Devuelve toda la información necesaria para la memoria:
      - Contexto: fechas, activos, perfil de riesgo del modelo
      - Métricas: Sharpe, Retorno, MDD, CAGR, Volatilidad, Sortino, Valor Final
      - Comparativa: PPO vs cada baseline (gana/pierde por métrica)
      - Resumen: conclusión en una frase
    """
    from src.investor.simulation_service import run_simulation
    from src.training_drl.risk_profiles import RISK_PROFILES

    result = run_simulation(capital=10000, commission=0.001)

    if "error" in result:
        return {"available": False, "error": result["error"]}

    metrics = result.get("metrics", {})
    if not metrics:
        return {"available": False, "error": "No se pudieron calcular métricas."}

    # ─── Contexto del entrenamiento ───────────────────────────────────────────
    # Leer universo activo y modelo de la BD para saber con qué se entrenó
    universe = universe_repo.get_active_universe(db)
    model_record = universe_repo.get_latest_model(db, model_type="ppo")

    # Perfil de riesgo usado en el entrenamiento
    risk_profile_id = None
    risk_profile_info = None
    if model_record and model_record.train_metrics:
        risk_profile_id = model_record.train_metrics.get('risk_profile', 'balanced')
        risk_profile_info = RISK_PROFILES.get(risk_profile_id)

    training_context = {
        "universe": {
            "tickers": universe.tickers if universe else [],
            "n_assets": universe.n_assets if universe else 0,
            "data_start": universe.start_date if universe else None,
            "data_end": universe.end_date if universe else None,
            "n_days_total": universe.n_days if universe else 0,
            "n_features": universe.n_features if universe else 0,
        },
        "model": {
            "risk_profile": risk_profile_id or 'balanced',
            "risk_profile_name": risk_profile_info['name'] if risk_profile_info else 'Equilibrado',
            "phi": risk_profile_info['phi'] if risk_profile_info else 0.02,
            "gamma": risk_profile_info['gamma'] if risk_profile_info else 0.01,
            "steps": model_record.steps if model_record else None,
            "trained_at": str(model_record.created_at)[:19] if model_record else None,
        },
        "test_period": result.get("test_period"),
        "commission": "0.1%",
        "initial_capital": "$10,000",
        "split": "80% train / 20% test",
    }

    # ─── Mejor estrategia por métrica ─────────────────────────────────────────
    best_sharpe = max(metrics.items(), key=lambda x: x[1].get('Sharpe Ratio', -999))
    best_retorno = max(metrics.items(), key=lambda x: x[1].get('Retorno Total (%)', -999))
    best_mdd = max(metrics.items(), key=lambda x: x[1].get('Max Drawdown (%)', -999))
    best_sortino = max(metrics.items(), key=lambda x: x[1].get('Sortino Ratio', -999))

    # ─── PPO vs cada baseline ─────────────────────────────────────────────────
    ppo = metrics.get('IA_PPO', {})
    comparisons = {}
    for name, m in metrics.items():
        if name == 'IA_PPO':
            continue
        comparisons[name] = {
            'ppo_wins_sharpe': bool(ppo.get('Sharpe Ratio', 0) > m.get('Sharpe Ratio', 0)),
            'ppo_wins_retorno': bool(ppo.get('Retorno Total (%)', 0) > m.get('Retorno Total (%)', 0)),
            'ppo_wins_mdd': bool(ppo.get('Max Drawdown (%)', 0) > m.get('Max Drawdown (%)', 0)),
            'ppo_wins_sortino': bool(ppo.get('Sortino Ratio', 0) > m.get('Sortino Ratio', 0)),
            'sharpe_diff': float(round(ppo.get('Sharpe Ratio', 0) - m.get('Sharpe Ratio', 0), 3)),
            'retorno_diff': float(round(ppo.get('Retorno Total (%)', 0) - m.get('Retorno Total (%)', 0), 2)),
        }

    n_baselines = len(comparisons)
    n_wins_sharpe = sum(1 for c in comparisons.values() if c['ppo_wins_sharpe'])
    n_wins_retorno = sum(1 for c in comparisons.values() if c['ppo_wins_retorno'])
    n_wins_sortino = sum(1 for c in comparisons.values() if c['ppo_wins_sortino'])

    # ─── Resumen textual ──────────────────────────────────────────────────────
    test_info = result.get("test_period", {})
    summary_lines = [
        f"Período de test: {test_info.get('start', '?')} a {test_info.get('end', '?')} ({test_info.get('days', '?')} días).",
        f"Universo: {len(universe.tickers) if universe else '?'} activos.",
        f"Perfil de riesgo: {risk_profile_info['name'] if risk_profile_info else 'Equilibrado'}.",
        f"PPO supera en Sharpe a {n_wins_sharpe}/{n_baselines} estrategias.",
        f"PPO supera en Retorno a {n_wins_retorno}/{n_baselines} estrategias.",
        f"PPO supera en Sortino a {n_wins_sortino}/{n_baselines} estrategias.",
        f"Mejor Sharpe global: {best_sharpe[0]} ({best_sharpe[1].get('Sharpe Ratio'):.3f}).",
    ]

    return {
        "available": True,
        "training_context": training_context,
        "metrics": metrics,
        "best_by_metric": {
            "sharpe": {"strategy": best_sharpe[0], "value": float(best_sharpe[1].get('Sharpe Ratio', 0))},
            "retorno": {"strategy": best_retorno[0], "value": float(best_retorno[1].get('Retorno Total (%)', 0))},
            "mdd": {"strategy": best_mdd[0], "value": float(best_mdd[1].get('Max Drawdown (%)', 0))},
            "sortino": {"strategy": best_sortino[0], "value": float(best_sortino[1].get('Sortino Ratio', 0))},
        },
        "ppo_vs_baselines": comparisons,
        "summary": summary_lines,
    }


@app.get("/expanding-window/results", tags=["Público"])
async def get_expanding_window_results():
    """
    Retorna los resultados del último expanding window ejecutado.

    Mismo formato que /walk-forward/results pero con train desde el día 0
    (train crece en cada ventana, test de 3 meses).
    """
    csv_path = 'src/reports/expanding_window_results.csv'
    if not os.path.exists(csv_path):
        return {"available": False, "error": "Expanding window no ejecutado aún."}

    df = pd.read_csv(csv_path, index_col=0)
    windows = df.to_dict(orient='records')
    return {
        "available": True,
        "n_windows": len(windows),
        "windows": windows,
        "summary": {
            "sharpe_mean": round(df['Sharpe Ratio'].mean(), 3),
            "sharpe_std": round(df['Sharpe Ratio'].std(), 3),
            "retorno_mean": round(df['Retorno Total (%)'].mean(), 1),
            "mdd_mean": round(df['Max Drawdown (%)'].mean(), 1),
            "windows_positive_sharpe": int((df['Sharpe Ratio'] > 0).sum()),
        }
    }


@app.get("/sensitivity/results", tags=["Público"])
async def get_sensitivity_results():
    """
    Retorna la tabla de resultados del análisis de sensibilidad.

    Lee src/reports/sensitivity_analysis.csv generado por run_sensitivity_analysis
    y lo devuelve como JSON listo para el frontend.
    """
    csv_path = 'src/reports/sensitivity_analysis.csv'
    if not os.path.exists(csv_path):
        return {"available": False, "error": "Análisis de sensibilidad no ejecutado aún."}

    df = pd.read_csv(csv_path, index_col=0)
    configs = df.reset_index().to_dict(orient='records')

    best_sharpe_row = df['Sharpe Ratio'].idxmax()
    best_return_row = df['Retorno Total (%)'].idxmax()
    best_mdd_row = df['Max Drawdown (%)'].idxmax()  # menos negativo = mejor

    return {
        "available": True,
        "n_configs": len(configs),
        "configs": configs,
        "best_by_metric": {
            "sharpe": {"config": best_sharpe_row, "value": float(df.loc[best_sharpe_row, 'Sharpe Ratio'])},
            "retorno": {"config": best_return_row, "value": float(df.loc[best_return_row, 'Retorno Total (%)'])},
            "mdd": {"config": best_mdd_row, "value": float(df.loc[best_mdd_row, 'Max Drawdown (%)'])},
        },
    }


@app.get("/risk-profiles", tags=["Público"])
async def get_risk_profiles():
    """
    Lista los perfiles de riesgo disponibles para el entrenamiento PPO.

    Cada perfil define phi (penalización drawdown) y gamma (penalización turnover)
    de la función de recompensa del agent.
    """
    from src.training_drl.risk_profiles import list_profiles
    return list_profiles()


@app.get("/estado", tags=["Público"])
async def get_system_status():
    """Estado del sistema: qué fases están completadas."""
    # Un fichero .lock indica que hay un background task corriendo.
    # Se crea al lanzar y se borra al terminar.
    training_running = os.path.exists('models/.training.lock')
    wf_running       = os.path.exists('models/.walkforward.lock')
    ew_running       = os.path.exists('models/.expanding.lock')
    sa_running       = os.path.exists('models/.sensitivity.lock')

    return {
        "fase1_datos":          os.path.exists('data/normalized_features.csv'),
        "fase3_modelo_acad":    os.path.exists('models/best_model_academic/best_model.zip'),
        "fase3_training_done":  os.path.exists('models/best_model_academic/best_model.zip') and not training_running,
        "fase3_wf_done":        os.path.exists('src/reports/walk_forward_results.csv') and not wf_running,
        "fase3_ew_done":        os.path.exists('src/reports/expanding_window_results.csv') and not ew_running,
        "fase3_sa_done":        os.path.exists('src/reports/sensitivity_analysis.csv') and not sa_running,
        "fase4_especulativo":   os.path.exists('models/speculative_gmm.pkl'),
        "background_running":   training_running or wf_running or ew_running or sa_running,
    }


# ─── ENDPOINTS DE ADMINISTRADOR ──────────────────────────────────────────────
# Todos requieren JWT con role='admin'

@app.post("/admin/fase1/screener", tags=["Admin"])
async def run_screener(
    start_date: str = "2020-01-01",
    end_date: str = "2026-04-01",
    top_n: int = 15,
    max_per_sector: int = 3,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Screener: filtra S&P 500 a los mejores candidatos (solo admin).

    Los candidatos seleccionados se guardan en BD y se usan automáticamente
    como tickers por defecto en /admin/fase1/preparar-datos.
    No es necesario copiar tickers manualmente entre endpoints.
    """
    screener = MarketScreener(max_per_sector=max_per_sector)
    result = screener.run(
        start_date=start_date, end_date=end_date,
        top_n=top_n,
        # IVV: benchmark de mercado (proxy S&P 500)
        # BND: renta fija (cobertura en crisis)
        # IBIT: Bitcoin ETF (reserva de valor digital)
        # ETHA: Ethereum ETF (plataforma DeFi/smart contracts)
        # IBIT y ETHA no están en el S&P 500 pero son requisito del TFM
        # ("Integrando Criptoactivos en la Inversión Tradicional").
        force_include=['IVV', 'BND', 'IBIT', 'ETHA']
    )

    details_records = (
        result['details'].to_dict(orient='records')
        if not result['details'].empty else []
    )

    # Guardar resultado en BD para que preparar-datos lo use como default
    universe_repo.save_screener_result(
        db,
        candidates=result['candidates'],
        start_date=start_date,
        end_date=end_date,
        filters={"top_n": top_n, "max_per_sector": max_per_sector},
        details=details_records,
        created_by=admin.email,
    )

    return {
        "candidates": result['candidates'],
        "n_candidates": len(result['candidates']),
        "details": details_records,
        "filtered_out": result['filtered_out'],
        "info": "Estos tickers se usarán por defecto en /admin/fase1/preparar-datos",
    }


@app.post("/admin/fase1/preparar-datos", tags=["Admin"])
async def prepare_data(config: DownloadConfig = None,
                          admin: User = Depends(require_admin),
                          db: Session = Depends(get_db)):
    """
    Descarga precios, genera features y guarda CSVs (solo admin).

    Si no se pasan tickers explícitamente, usa los del último screener.
    Si no hay screener previo, usa el CORE_UNIVERSE (8 activos por defecto).
    """
    # Resolver tickers: explícitos > último screener > core universe
    if config is None:
        config = DownloadConfig()

    # Detectar y descartar tickers inválidos:
    # Pydantic/Angular pueden enviar ["string"], None, o lista vacía
    BLACKLIST = {"string", "null", "undefined", "none", ""}
    if config.tickers is not None:
        config.tickers = [
            t.upper().strip() for t in config.tickers
            if t and t.lower().strip() not in BLACKLIST and len(t) <= 10
        ]

    # Si no quedan tickers válidos, usar los de BD
    if not config.tickers:
        config.tickers = universe_repo.get_default_tickers(db)

    # Garantizar que IBIT y ETHA siempre están presentes (requisito del TFM:
    # "Integrando Criptoactivos en la Inversión Tradicional")
    for crypto in ['IBIT', 'ETHA']:
        if crypto not in config.tickers:
            config.tickers.append(crypto)

    print(f"  Tickers resueltos: {config.tickers}")

    descargar_dividendos(config.tickers, config.start, config.end)
    df_features, _ = generar_dataset(config.tickers, config.start, config.end)

    # Registrar universo en BD (desactiva los anteriores)
    universe = universe_repo.create_universe(
        db,
        tickers=config.tickers,
        start_date=config.start,
        end_date=config.end,
        n_features=df_features.shape[1],
        n_days=len(df_features),
        created_by=admin.email,
    )

    # Legacy: también guardar JSON para retrocompatibilidad con Streamlit
    save_config(config.tickers, config.start, config.end,
                df_features.shape[1], len(df_features))

    return {
        "status": "Datos preparados",
        "universe_id": universe.id,
        "tickers": universe.tickers,
        "n_features": universe.n_features,
        "n_days": universe.n_days,
    }


@app.get("/admin/fase2/validar-datos", tags=["Admin"],
         dependencies=[Depends(require_admin)])
async def validate_data():
    """Valida integridad de los CSVs generados (solo admin)."""
    if not os.path.exists('data/normalized_features.csv'):
        return {"ok": False, "error": "Ejecuta primero /admin/fase1/preparar-datos"}

    try:
        df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
        df_p = pd.read_csv('data/original_prices.csv', index_col=0)

        n_nan = int(df_f.isnull().values.sum())
        n_inf = int(np.isinf(df_f.values).sum())

        env = PortfolioEnv("data/normalized_features.csv", "data/original_prices.csv")
        env.reset()

        return {
            "ok": n_nan == 0 and n_inf == 0,
            "features": {"filas": len(df_f), "columnas": df_f.shape[1],
                         "nan": n_nan, "inf": n_inf},
            "precios": {"filas": len(df_p), "activos": df_p.shape[1]},
            "entorno": {"n_assets": env.n_assets, "obs_shape": env.observation_space.shape[0]},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/admin/fase3/entrenar-academico", tags=["Admin"])
async def start_training(background_tasks: BackgroundTasks,
                                 steps: int = 500000, patience: int = 8,
                                 risk_profile: str = 'balanced',
                                 admin: User = Depends(require_admin),
                                 db: Session = Depends(get_db)):
    """
    Entrena PPO con validación académica completa (solo admin).

    risk_profile: perfil de riesgo que determina phi y gamma del reward.
      - balanced:     phi=0.02, gamma=0.01 (equilibrio)
      - conservative: phi=0.05, gamma=0.01 (preservar capital)
      - low_turnover: phi=0.02, gamma=0.02 (mínima rotación, mejor Sharpe)
      - aggressive:   phi=0.01, gamma=0.005 (máxima libertad)
    """
    from src.training_drl.risk_profiles import get_profile

    # Validar perfil
    try:
        profile = get_profile(risk_profile)
    except ValueError as e:
        return {"error": str(e)}

    universe = universe_repo.get_active_universe(db)
    if universe is None:
        return {"error": "Ejecuta primero /admin/fase1/preparar-datos"}

    # Registrar modelo en BD con el perfil usado
    model_record = universe_repo.register_model(
        db, universe_id=universe.id, model_type="ppo",
        model_path="models/best_model_academic/best_model.zip", steps=steps,
    )

    _create_lock('models/.training.lock')

    def train_and_update_status(model_id: int, **kwargs):
        try:
            entrenar_academico(**kwargs)
            db_session = SessionLocal()
            try:
                universe_repo.update_model_status(db_session, model_id, "ready",
                                                  metrics={"risk_profile": risk_profile})
                print(f"  [BD] Modelo PPO id={model_id} marcado como 'ready' "
                      f"(perfil: {risk_profile}).")
            finally:
                db_session.close()
        except Exception as e:
            db_session = SessionLocal()
            try:
                universe_repo.update_model_status(db_session, model_id, "failed")
                print(f"  [BD] Modelo PPO id={model_id} marcado como 'failed': {e}")
            finally:
                db_session.close()
        finally:
            _remove_lock('models/.training.lock')

    background_tasks.add_task(
        train_and_update_status,
        model_id=model_record.id,
        total_timesteps=steps,
        patience=patience,
        risk_profile=risk_profile,
    )
    return {
        "message": f"Entrenamiento iniciado (máx. {steps:,} pasos, patience={patience}, "
                   f"perfil: {profile['name']}).",
        "risk_profile": risk_profile,
        "phi": profile['phi'],
        "gamma": profile['gamma'],
        "universe_id": universe.id,
        "model_id": model_record.id,
    }


@app.post("/admin/fase3/walk-forward", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def start_walk_forward(background_tasks: BackgroundTasks,
                                steps_por_ventana: int = 100000):
    """Walk-forward validation temporal (solo admin)."""
    _create_lock('models/.walkforward.lock')

    def wf_with_lock(**kwargs):
        try:
            walk_forward_validation(**kwargs)
        finally:
            _remove_lock('models/.walkforward.lock')

    background_tasks.add_task(
        wf_with_lock,
        features_path='data/normalized_features.csv',
        prices_path='data/original_prices.csv',
        total_timesteps=steps_por_ventana,
    )
    return {"message": f"Walk-forward (rolling) iniciado ({steps_por_ventana:,} pasos/ventana)."}


@app.post("/admin/fase3/expanding-window", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def start_expanding_window(background_tasks: BackgroundTasks,
                                    steps_por_ventana: int = 100000,
                                    min_train_days: int = 504,
                                    test_days: int = 63):
    """
    Expanding Window Validation (solo admin).

    A diferencia del walk-forward rolling (ventana fija), aquí el train empieza
    siempre desde el día 0 y crece en cada ventana. Cada ventana entrena con
    TODA la historia disponible hasta ese momento y evalúa en los siguientes
    test_days días (63 = 3 meses por defecto, según indicación del tutor).

    Genera más puntos de evaluación que el rolling: ~12 ventanas con 5 años
    de datos vs ~3-4 del rolling.
    """
    _create_lock('models/.expanding.lock')

    def ew_with_lock(**kwargs):
        try:
            expanding_window_validation(**kwargs)
        finally:
            _remove_lock('models/.expanding.lock')

    background_tasks.add_task(
        ew_with_lock,
        features_path='data/normalized_features.csv',
        prices_path='data/original_prices.csv',
        min_train_days=min_train_days,
        test_days=test_days,
        total_timesteps=steps_por_ventana,
    )
    return {
        "message": f"Expanding window iniciado ({steps_por_ventana:,} pasos/ventana, "
                   f"train mínimo={min_train_days}d, test={test_days}d).",
    }


@app.post("/admin/fase3/sensitivity-analysis", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def start_sensitivity_analysis(background_tasks: BackgroundTasks,
                                        steps_por_config: int = 200000):
    """
    Análisis de sensibilidad: entrena 4 configuraciones del reward PPO
    y genera tabla comparativa (solo admin).

    Configuraciones: A (actual), B (más MDD), C (más turnover), D (agresivo).
    Genera: src/reports/sensitivity_analysis.csv y .png

    Tarda ~4x el tiempo de un entrenamiento normal.
    """
    _create_lock('models/.sensitivity.lock')

    def sa_with_lock(**kwargs):
        try:
            run_sensitivity_analysis(**kwargs)
        finally:
            _remove_lock('models/.sensitivity.lock')

    background_tasks.add_task(
        sa_with_lock,
        total_timesteps=steps_por_config,
    )
    return {
        "message": f"Análisis de sensibilidad iniciado ({steps_por_config:,} pasos × 4 configs).",
    }


@app.post("/admin/fase4/ajustar-especulativo", tags=["Admin"])
async def fit_speculative_agent(split_pct: float = 0.8,
                                admin: User = Depends(require_admin),
                                db: Session = Depends(get_db)):
    """Ajusta agente especulativo GMM + K-Means (solo admin)."""
    universe = universe_repo.get_active_universe(db)
    if universe is None:
        return {"error": "Ejecuta primero /admin/fase1/preparar-datos"}

    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv', index_col=0)

    split_idx = int(len(df_f) * split_pct)
    agent = SpeculativeAgent(n_regimes=3, n_clusters=3, cluster_window=60)
    agent.fit(df_f.iloc[:split_idx], df_p.iloc[:split_idx])

    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/speculative_gmm.pkl', 'wb') as f:
        pickle.dump(agent, f)

    # Registrar modelo como ready (es síncrono — ya terminó)
    model_record = universe_repo.register_model(
        db, universe_id=universe.id, model_type="speculative",
        model_path="models/speculative_gmm.pkl",
    )
    universe_repo.update_model_status(db, model_record.id, "ready")

    equity = agent.backtest(df_f.iloc[split_idx:], df_p.iloc[split_idx:])
    ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    return {
        "status": "Agente especulativo ajustado",
        "universe_id": universe.id,
        "tickers": universe.tickers,
        "retorno_test": f"{ret:.1f}%",
        "valor_final": f"${equity.iloc[-1]:,.2f}",
    }
