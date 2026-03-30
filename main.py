from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data_pipeline import generar_dataset_completo
from src.trading_env import PortfolioEnv
from src.train import (
    entrenar_con_validacion,
    entrenar_ablacion,
    entrenar_multisemilla,
    walk_forward_validation,
)

app = FastAPI(title="TFM Trading AI API")

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class DownloadConfig(BaseModel):
    tickers: List[str] = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'AWK', 'CB']
    start: str = "2000-01-01"
    end: str = "2026-03-23"


# ---------------------------------------------------------------------------
# Phase 1 - Data
# ---------------------------------------------------------------------------

@app.post("/phase1/prepare-data")
async def prepare_data(config: DownloadConfig):
    """Download Yahoo Finance data and generate features_normalizadas.csv + precios_originales.csv."""
    dataset_norm, _ = generar_dataset_completo(config.tickers, config.start, config.end)
    return {"features": dataset_norm.shape[1], "days": len(dataset_norm)}


# ---------------------------------------------------------------------------
# Phase 2 - Status
# ---------------------------------------------------------------------------

@app.get("/status")
async def get_status():
    """Check which pipeline stages have been completed."""
    return {
        "phase1_done":    os.path.exists('data/features_normalizadas.csv'),
        "model_trained":  os.path.exists('models/best_model/best_model.zip'),
        "ablation_done":  os.path.exists('models/ablation_log_return/best_model.zip'),
        "multiseed_done": os.path.exists('models/multisemilla.csv'),
        "wfv_done":       os.path.exists('models/wfv_resultados.csv'),
    }


# ---------------------------------------------------------------------------
# Phase 3 - Training
# ---------------------------------------------------------------------------

@app.post("/phase3/train")
async def start_training(background_tasks: BackgroundTasks, steps: int = 300000):
    """Main PPO training (reward_mode=sharpe_drawdown). Saves best model to models/best_model/."""
    background_tasks.add_task(entrenar_con_validacion, total_timesteps=steps)
    return {"message": f"Main training started ({steps} steps)."}


@app.post("/phase3/ablation")
async def start_ablation(background_tasks: BackgroundTasks, steps: int = 300000):
    """Ablation baseline (reward_mode=log_return, no drawdown penalty)."""
    background_tasks.add_task(entrenar_ablacion, total_timesteps=steps)
    return {"message": f"Ablation study started ({steps} steps)."}


@app.post("/phase3/multiseed")
async def start_multiseed(background_tasks: BackgroundTasks, steps: int = 300000):
    """Train with 3 different seeds. Saves Sharpe mean +/- std to models/multisemilla.csv."""
    background_tasks.add_task(entrenar_multisemilla, total_timesteps=steps)
    return {"message": f"Multi-seed training started ({steps} steps x 3 seeds)."}


@app.post("/phase3/walk-forward")
async def start_walk_forward(background_tasks: BackgroundTasks, steps: int = 100000):
    """Walk-forward validation with 3 chronological windows. Saves to models/wfv_resultados.csv."""
    background_tasks.add_task(walk_forward_validation, total_timesteps=steps)
    return {"message": f"Walk-forward validation started ({steps} steps x 3 windows)."}


# ---------------------------------------------------------------------------
# Phase 4 - Inference
# ---------------------------------------------------------------------------

@app.get("/predict")
async def predict_action():
    """Load best model and return recommended portfolio weights for today."""
    model_path = "models/best_model/best_model.zip"
    if not os.path.exists(model_path):
        return {"error": "No trained model found. Run /phase3/train first."}
    if not os.path.exists('data/features_normalizadas.csv'):
        return {"error": "No data found. Run /phase1/prepare-data first."}

    model = PPO.load(model_path)
    env   = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv')
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    # Normalise to portfolio weights (same logic as trading_env)
    action_sum = float(np.sum(action))
    if action_sum > 1e-6:
        risky_fraction = min(action_sum, 1.0)
        weights = (action / action_sum) * risky_fraction
    else:
        weights = np.zeros(len(action))
    cash_pct = max(1.0 - float(np.sum(weights)), 0.0)

    tickers = pd.read_csv('data/precios_originales.csv', index_col=0).columns.tolist()
    tickers = [t.replace("_Close", "") for t in tickers]

    result = {tickers[i]: f"{float(weights[i])*100:.2f}%" for i in range(len(tickers))}
    result["CASH"] = f"{cash_pct*100:.2f}%"

    return {"recommended_weights": result}
