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
from typing import List
import os
import pandas as pd
import numpy as np

from src.pipeline_getdata.data_downloader import descargar_dividendos, generar_dataset
from src.training_drl.environment_trading import PortfolioEnv
from src.training_drl.training_analysis import entrenar_academico, walk_forward_validation
from src.unsupervised.speculative_agent import SpeculativeAgent
from src.pipeline_getdata.asset_registry import get_tickers, get_universe
from src.pipeline_getdata.market_screener import MarketScreener

from src.auth.models import init_db
from src.auth.auth_router import router as auth_router
from src.auth.auth_service import require_admin
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
    tickers: List[str] = get_tickers('core')
    start: str = "2014-01-01"
    end: str = "2026-03-01"


# ─── ENDPOINTS PÚBLICOS ──────────────────────────────────────────────────────

@app.get("/universo", tags=["Público"])
async def ver_universo(level: str = 'core'):
    """Diccionario de activos con metadatos completos."""
    df = get_universe(level)
    return df.reset_index().to_dict(orient='records')


@app.get("/estado", tags=["Público"])
async def ver_estado():
    """Estado del sistema: qué fases están completadas."""
    return {
        "fase1_datos":        os.path.exists('data/normalized_features.csv'),
        "fase3_modelo_std":   os.path.exists('models/best_model/best_model.zip'),
        "fase3_modelo_acad":  os.path.exists('models/best_model_academic/best_model.zip'),
        "fase4_especulativo": os.path.exists('models/speculative_gmm.pkl'),
    }


# ─── ENDPOINTS DE ADMINISTRADOR ──────────────────────────────────────────────
# Todos requieren JWT con role='admin'

@app.post("/admin/fase1/screener", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def ejecutar_screener(
    start_date: str = "2020-01-01",
    end_date: str = "2026-04-01",
    top_n: int = 15,
    max_per_sector: int = 3
):
    """Screener: filtra S&P 500 a los mejores candidatos (solo admin)."""
    screener = MarketScreener(max_per_sector=max_per_sector)
    result = screener.run(
        start_date=start_date, end_date=end_date,
        top_n=top_n, force_include=['IVV', 'BND']
    )
    return {
        "candidates": result['candidates'],
        "n_candidates": len(result['candidates']),
        "details": result['details'].to_dict(orient='records') if not result['details'].empty else [],
        "filtered_out": result['filtered_out'],
    }


@app.post("/admin/fase1/preparar-datos", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def preparar_datos(config: DownloadConfig):
    """Descarga precios, genera features y guarda CSVs (solo admin)."""
    descargar_dividendos(config.tickers, config.start, config.end)
    generar_dataset(config.tickers, config.start, config.end)
    return {"status": "Datos preparados"}


@app.get("/admin/fase2/validar-datos", tags=["Admin"],
         dependencies=[Depends(require_admin)])
async def validar_datos():
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


@app.post("/admin/fase3/entrenar-academico", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def iniciar_entrenamiento(background_tasks: BackgroundTasks,
                                 steps: int = 500000, patience: int = 8):
    """Entrena PPO con validación académica completa (solo admin)."""
    background_tasks.add_task(entrenar_academico, total_timesteps=steps, patience=patience)
    return {"message": f"Entrenamiento iniciado (máx. {steps:,} pasos, patience={patience})."}


@app.post("/admin/fase3/walk-forward", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def iniciar_walk_forward(background_tasks: BackgroundTasks,
                                steps_por_ventana: int = 100000):
    """Walk-forward validation temporal (solo admin)."""
    background_tasks.add_task(
        walk_forward_validation,
        features_path='data/normalized_features.csv',
        prices_path='data/original_prices.csv',
        total_timesteps=steps_por_ventana
    )
    return {"message": f"Walk-forward iniciado ({steps_por_ventana:,} pasos/ventana)."}


@app.post("/admin/fase4/ajustar-especulativo", tags=["Admin"],
          dependencies=[Depends(require_admin)])
async def ajustar_especulativo(split_pct: float = 0.8):
    """Ajusta agente especulativo GMM + K-Means (solo admin)."""
    if not os.path.exists('data/normalized_features.csv'):
        return {"error": "Ejecuta primero /admin/fase1/preparar-datos"}

    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv', index_col=0)

    split_idx = int(len(df_f) * split_pct)
    agente = SpeculativeAgent(n_regimes=3, n_clusters=3, cluster_window=60)
    agente.fit(df_f.iloc[:split_idx], df_p.iloc[:split_idx])

    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/speculative_gmm.pkl', 'wb') as f:
        pickle.dump(agente, f)

    equity = agente.backtest(df_f.iloc[split_idx:], df_p.iloc[split_idx:])
    ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    return {
        "status": "Agente especulativo ajustado",
        "retorno_test": f"{ret:.1f}%",
        "valor_final": f"${equity.iloc[-1]:,.2f}",
    }
