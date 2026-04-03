from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from src.pipeline_getdata.data_downloader import descargar_dividendos, generar_dataset
from src.training_drl.environment_trading import PortfolioEnv
from src.training_drl.training_analysis import entrenar_academico, walk_forward_validation
from src.unsupervised.speculative_agent import SpeculativeAgent

app = FastAPI(title="TFM Trading AI API")

# --- Modelos de Datos ---
class DownloadConfig(BaseModel):
    tickers: List[str] = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
    start: str = "2014-01-01"
    end: str = "2026-03-01"


# ─── FASE 1: Datos ───────────────────────────────────────────────────────────

@app.post("/fase1/preparar-datos")
async def preparar_datos(config: DownloadConfig):
    """Descarga precios y dividendos, genera features y guarda los CSVs en data/."""
    descargar_dividendos(config.tickers, config.start, config.end)
    generar_dataset(config.tickers, config.start, config.end)
    return {"status": "Datos preparados"}


# ─── FASE 2: Validación de datos ─────────────────────────────────────────────

@app.get("/fase2/validar-datos")
async def validar_datos():
    """
    Valida que los CSVs generados en Fase 1 son compatibles con el entorno
    de entrenamiento. Útil para ejecutar antes de un entrenamiento largo.

    Comprueba: dimensiones, ausencia de NaN/Inf, coherencia de índices
    y que el entorno Gymnasium se instancia correctamente.
    """
    if not os.path.exists('data/normalized_features.csv'):
        return {"ok": False, "error": "Ejecuta primero /fase1/preparar-datos"}

    try:
        import pandas as pd, numpy as np
        df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
        df_p = pd.read_csv('data/original_prices.csv',     index_col=0)

        n_nan  = int(df_f.isnull().values.sum())
        n_inf  = int(np.isinf(df_f.values).sum())
        n_nanp = int(df_p.isnull().values.sum())

        env = PortfolioEnv("data/normalized_features.csv", "data/original_prices.csv")
        env.reset()

        return {
            "ok":        n_nan == 0 and n_inf == 0,
            "features":  {"filas": len(df_f), "columnas": df_f.shape[1],
                          "nan": n_nan, "inf": n_inf,
                          "fecha_inicio": df_f.index[0], "fecha_fin": df_f.index[-1]},
            "precios":   {"filas": len(df_p), "activos": df_p.shape[1], "nan": n_nanp},
            "entorno":   {"n_assets": env.n_assets, "obs_shape": env.observation_space.shape[0]},
            "aviso":     "Datos limpios, listo para entrenar." if n_nan == 0 and n_inf == 0
                         else f"ATENCION: {n_nan} NaN y {n_inf} Inf en features."
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
      - src/reports/walk_forward_results.csv
      - src/reports/walk_forward_analysis.png
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
            "src/reports/walk_forward_results.csv",
            "src/reports/walk_forward_analysis.png"
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
        "fase4_especulativo": os.path.exists('models/speculative_gmm.pkl'),
        "reportes": {
            "backtest":      os.path.exists('reports/backtest_principal.png'),
            "diagnostico":   os.path.exists('reports/training_diagnostics.png'),
            "overfitting":   os.path.exists('reports/overfitting_analysis.png'),
            "walk_forward":  os.path.exists('reports/walk_forward_analysis.png'),
            "regimenes":     os.path.exists('reports/regime_analysis.png'),
        }
    }


# ─── FASE 4: Agente Especulativo No Supervisado ─────────────────────────────

@app.post("/fase4/ajustar-especulativo")
async def ajustar_especulativo(split_pct: float = 0.8):
    """
    Ajusta el agente especulativo basado en GMM + K-Means (aprendizaje no supervisado).

    A diferencia del PPO, este modelo se ajusta en segundos — no necesita BackgroundTasks.
    Detecta regímenes de mercado y agrupa activos por comportamiento dinámico.

    El resultado se guarda como pickle para usarlo en el backtest del dashboard.
    Se entrena solo sobre el split de train para evitar lookahead bias.
    """
    if not os.path.exists('data/normalized_features.csv'):
        return {"error": "Ejecuta primero /fase1/preparar-datos"}

    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv', index_col=0)

    split_idx = int(len(df_f) * split_pct)
    features_train = df_f.iloc[:split_idx]
    precios_train  = df_p.iloc[:split_idx]

    agente = SpeculativeAgent(n_regimes=3, n_clusters=3, cluster_window=60)
    agente.fit(features_train, precios_train)

    # Guardar modelo ajustado
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/speculative_gmm.pkl', 'wb') as f:
        pickle.dump(agente, f)

    # Backtest rápido sobre test para dar feedback inmediato
    features_test = df_f.iloc[split_idx:]
    precios_test  = df_p.iloc[split_idx:]
    equity = agente.backtest(features_test, precios_test)

    ret_total = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    return {
        "status": "Agente especulativo ajustado y guardado",
        "modelo": "models/speculative_gmm.pkl",
        "regimenes_detectados": agente.detector.descripcion_regimenes(features_train).to_dict(orient='records'),
        "backtest_test": {
            "retorno_total": f"{ret_total:.1f}%",
            "valor_final": f"${equity.iloc[-1]:,.2f}"
        }
    }
