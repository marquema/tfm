from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from src.pipeline_getdata.data_downloader import descargar_dividendos, generar_dataset
from src.environment_trading import PortfolioEnv
from src.training_analysis import entrenar_academico, walk_forward_validation

app = FastAPI(title="TFM Trading AI API")

# --- Modelos de Datos ---
class DownloadConfig(BaseModel):
    tickers: List[str] = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
    start: str = "2014-01-01"
    end: str = "2026-03-01"

# --- Variables Globales ---
env = None


# ─── FASE 1: Datos ───────────────────────────────────────────────────────────

@app.post("/fase1/preparar-datos")
async def preparar_datos(config: DownloadConfig):
    """Descarga precios y dividendos, genera features y guarda los CSVs en data/."""
    descargar_dividendos(config.tickers, config.start, config.end)
    generar_dataset(config.tickers, config.start, config.end)
    return {"status": "Datos preparados",
            "features": "data/normalized_features.csv",
            "precios":  "data/original_prices.csv"}


# ─── FASE 2: Entorno ─────────────────────────────────────────────────────────

@app.post("/fase2/inicializar-entorno")
async def init_env():
    """Carga los CSVs y prepara el entorno Gymnasium completo."""
    global env
    if not os.path.exists('data/normalized_features.csv'):
        return {"error": "Primero ejecuta /fase1/preparar-datos"}

    env = PortfolioEnv("data/normalized_features.csv", "data/original_prices.csv")
    env.reset()
    return {"status": "Entorno listo", "n_assets": env.n_assets,
            "n_pasos": len(env.df_features), "n_features": env.df_features.shape[1]}


# ─── FASE 3: Entrenamiento ───────────────────────────────────────────────────

@app.post("/fase3/entrenar-academico")
async def iniciar_entrenamiento_academico(
    background_tasks: BackgroundTasks,
    steps: int = 500000,
    patience: int = 8
):
    """
    Entrenamiento PPO con validación académica completa:
      - Early stopping por reward out-of-sample (patience=N evaluaciones sin mejora)
      - Detección de sobreajuste (gap train/eval)
      - Diagnóstico de entropía, value loss y explained variance

    Reportes generados:
      - reports/training_diagnostics.png
      - reports/overfitting_analysis.png
      - models/best_model_academic/best_model.zip
    """
    background_tasks.add_task(
        entrenar_academico,
        total_timesteps=steps,
        patience=patience
    )
    return {
        "message": f"Entrenamiento académico iniciado (máx. {steps:,} pasos, patience={patience}).",
        "reportes": [
            "reports/training_diagnostics.png",
            "reports/overfitting_analysis.png",
            "models/best_model_academic/best_model.zip"
        ]
    }


@app.post("/fase3/walk-forward")
async def iniciar_walk_forward(
    background_tasks: BackgroundTasks,
    steps_por_ventana: int = 100000
):
    """
    Validación Walk-Forward con ventanas de tamaño fijo (2 años train + 1 año test).

    El número de ventanas y su tamaño se calculan internamente a partir del dataset
    disponible en data/normalized_features.csv. Con datos desde 2018 produce 6 ventanas;
    si se amplía el rango histórico, las ventanas se añaden automáticamente.

    steps_por_ventana: pasos de entrenamiento PPO por cada ventana. Ajustar según
    recursos disponibles (100k = rápido/orientativo, 300k = más preciso).

    Reportes generados:
      - reports/walk_forward_results.csv
      - reports/walk_forward_analysis.png
    """
    background_tasks.add_task(
        walk_forward_validation,
        features_path='data/normalized_features.csv',
        prices_path='data/original_prices.csv',
        total_timesteps=steps_por_ventana
    )
    return {
        "message": f"Walk-forward iniciado ({steps_por_ventana:,} pasos/ventana). "
                   "El numero de ventanas se calcula del dataset.",
        "reportes": [
            "reports/walk_forward_results.csv",
            "reports/walk_forward_analysis.png"
        ]
    }


# ─── INFERENCIA ──────────────────────────────────────────────────────────────

@app.get("/inferencia/pesos-actuales")
async def predecir_pesos():
    """
    Carga el mejor modelo académico y devuelve los pesos recomendados
    para el último estado disponible en los datos de test (20% final).
    """
    modelo_path = "models/best_model_academic/best_model.zip"
    if not os.path.exists(modelo_path):
        return {"error": f"No hay modelo entrenado en {modelo_path}. Ejecuta /fase3/entrenar-academico"}

    df_f      = pd.read_csv('data/normalized_features.csv', index_col=0)
    split_idx = int(len(df_f) * 0.8)

    # Usar el entorno de test para que el agente opere en datos no vistos
    env_test = PortfolioEnv(
        'data/normalized_features.csv',
        'data/original_prices.csv',
        start_idx=split_idx
    )
    modelo = PPO.load(modelo_path)
    obs, _ = env_test.reset()

    # Avanzar hasta el último paso disponible
    done = False
    while not done:
        action, _ = modelo.predict(obs, deterministic=True)
        obs, _, done, _, info = env_test.step(action)

    # Normalizar la última acción
    weights = np.clip(action, 0, 1)
    weights = weights / (weights.sum() + 1e-6)

    tickers = pd.read_csv('data/original_prices.csv', index_col=0).columns.tolist()
    return {
        "pesos_recomendados": {t: f"{float(w)*100:.2f}%" for t, w in zip(tickers, weights)},
        "valor_final_test":   f"${info['value']:,.2f}"
    }


# ─── ESTADO ──────────────────────────────────────────────────────────────────

@app.get("/estado")
async def ver_estado():
    """Comprueba qué fases están completadas."""
    return {
        "fase1_datos":        os.path.exists('data/normalized_features.csv'),
        "fase3_modelo_std":   os.path.exists('models/best_model/best_model.zip'),
        "fase3_modelo_acad":  os.path.exists('models/best_model_academic/best_model.zip'),
        "reportes": {
            "backtest":      os.path.exists('reports/backtest_principal.png'),
            "diagnostico":   os.path.exists('reports/training_diagnostics.png'),
            "overfitting":   os.path.exists('reports/overfitting_analysis.png'),
            "walk_forward":  os.path.exists('reports/walk_forward_analysis.png'),
            "regimenes":     os.path.exists('reports/regime_analysis.png'),
        }
    }
