from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from src.downloader_dataset import generar_dataset_completo
from src.environment_trading import PortfolioEnv
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from src.train import entrenar_modelo, entrenar_con_validacion
import numpy as np
# Ejemplo rápido con FastAPI
from fastapi import FastAPI
from stable_baselines3 import PPO

app = FastAPI(title="TFM Trading AI API")
model = PPO.load("models/best_model/best_model.zip")

# --- Modelos de Datos ---
class DownloadConfig(BaseModel):
    tickers: List[str] = 'IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB'
    start: str = "2014-01-01"
    end: str = "2026-03-23"

# --- Variables Globales ---
env = None

@app.post("/fase1/preparar-datos")
async def preparar_datos(config: DownloadConfig):
    """Ejecuta la descarga y creación de features completas (técnicas + dividend dynamics)."""
    dataset_norm, _ = generar_dataset_completo(config.tickers, config.start, config.end)
    return {"features": dataset_norm.shape[1], "dias": len(dataset_norm)}

@app.post("/fase2/inicializar-entorno")
async def init_env():
    """Carga los CSVs y prepara el entorno de Gymnasium."""
    global env
    if not os.path.exists('data/features_normalizadas.csv'):
        return {"error": "Primero debes ejecutar la Fase 1"}
    
    df_f = pd.read_csv("data/features_normalizadas.csv", index_col=0)
    df_p = pd.read_csv("data/precios_originales.csv", index_col=0)
    
    #env = PortfolioEnv(df_f, df_p)
    env = PortfolioEnv("data/features_normalizadas.csv", "data/precios_originales.csv")
    obs, _ = env.reset()

    for _ in range(5):
        accion_aleatoria = env.action_space.sample() # La IA "tonta"
        obs, reward, done, _, info = env.step(accion_aleatoria)
        print(f"Valor cartera: {info['value']:.2f}$ | Reward: {reward:.4f}")


    return {"status": "Entorno listo", "assets": env.n_assets, "initial_obs": obs.tolist()}


@app.post("/fase2/ejecutar-paso")
async def ejecutar_paso(pesos: List[float]):
    """
    Recibe una lista de pesos (ej: [0.5, 0.5, 0, 0...]) 
    y devuelve el nuevo estado del mercado.
    """
    global env
    if env is None:
        return {"error": "El entorno no está inicializado"}
    
    action = np.array(pesos, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    return {
        "valor_cartera": info["value"],
        "recompensa": reward,
        "finalizado": done,
        "proxima_observacion_resumen": obs[:5].tolist() # Solo enviamos 5 para no saturar
    }    
    

    

@app.get("/estado")
async def ver_estado():
    return {"fase1_completada": os.path.exists('data/features_normalizadas.csv'),
            "entorno_activo": env is not None}


@app.post("/fase3/entrenar")
async def iniciar_entrenamiento(background_tasks: BackgroundTasks, steps: int = 100000):
    """Lanza el entrenamiento en segundo plano."""
    #background_tasks.add_task(entrenar_modelo, total_timesteps=steps)
    background_tasks.add_task(entrenar_con_validacion, total_timesteps=steps)
    
    return {"message": f"Entrenamiento iniciado para {steps} pasos. Revisa la consola para el progreso."}

@app.get("/fase3/predecir-accion")
async def predecir_accion():
    """Carga el modelo guardado y sugiere los pesos para el día actual."""
    if not os.path.exists("models/ppo_portfolio_manager.zip"):
        return {"error": "No hay ningún modelo entrenado todavía."}
    
    # 1. Cargar el modelo y el entorno
    model = PPO.load("models/ppo_portfolio_manager")
    global env
    if env is None:
        env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv')
    
    # 2. Obtener la observación actual (el "hoy" del mercado)
    obs, _ = env.reset() # En un caso real, aquí usaríamos los datos más recientes
    
    # 3. Pedirle a la IA su decisión
    action, _states = model.predict(obs, deterministic=True)
    
    # 4. Normalizar para que sumen 1 (Portfolio Weights)
    weights = action / (np.sum(action) + 1e-6)
    
    # Asociar pesos con nombres de activos
    tickers = pd.read_csv('data/precios_originales.csv', index_col=0).columns.tolist()
    resultado = {tickers[i]: f"{float(weights[i])*100:.2f}%" for i in range(len(tickers))}
    
    return {"recomendacion_pesos": resultado}





@app.post("/predict")
def predict(data: dict):
    # Lógica para convertir JSON a array de numpy
    obs = np.array(data['features'])
    action, _ = model.predict(obs, deterministic=True)
    return {"weights": action.tolist()}