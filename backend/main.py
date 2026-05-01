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
            #todo: quitar la pass del print
            print("  Usuario admin creado: admin@tfm.com / admin123")
    finally:
        db.close()


# ─── routers 
app.include_router(auth_router)
app.include_router(investor_router)

# todo: quitar modelos pydantic de aqui
# todo: las fechas deberían ir avanzando, la de end, mes a mes
# asegurar que el screener usado es el último. CORE_UNIVERSE, solo para la primera vez

# ─── Modelos de datos 

class DownloadConfig(BaseModel):
    tickers: Optional[List[str]] = None  # None = usa último screener o CORE_UNIVERSE
    # start = 2017-11-09 (no 2014-01-01) porque ETH-USD —proxy de ETHA—
    # solo existe en Yahoo Finance desde esa fecha. Arrancar antes obligaria
    # a rellenar ETHA con bfill durante ~3.8 anos, practica desaconsejada
    # por el tutor en su revision academica.
    start: str = "2017-11-09"
    end: str = "2026-04-16"


# ─── Helpers para background tasks 

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

#todo: adaptar universo, para usar el último screener ejecutado, y sino, el core
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
    Tabla de extracción de conclusiones del TFM: PPO vs todas las estrategias.

    Devuelve toda la información necesaria para la memoria:
      - Contexto del entrenamiento: fechas, activos, perfil de riesgo del modelo.
            Esto demuestra trazabilidad: cualquier resultado se puede vincular exactamente al model
            y datos que lo produjeron. Tribunal pregunta "¿exactamente con qué entrenaste?".
      - Métricas por estrategia: Sharpe, Retorno, MDD, CAGR, Volatilidad, Sortino, Valor Final
            Mejor estrategia por métrica: "¿Quién gana en cada categoría?"  Si PPO aparece en varias categorías como ganador, 
            es el titular del TFM.
      - Comparativa: PPO vs cada baseline (gana/pierde por métrica) "¿Bate PPO a cada estrategia clásica?"
            Para cada baseline: bools ppo_wins_sharpe, ppo_wins_retorno, ppo_wins_mdd, ppo_wins_sortino + diferencias numéricas.
            Comparación principal que valida la hipótesis del TFM: que un 
            agente DRL puede competir con (o batir) las técnicas tradicionales de gestión de cartera.            
      - Resumen: "En una frase, ¿qué hay que recordar?"Líneas como 
            "PPO supera en Sharpe a X de 5 estrategias", "Periodo de test: 2024-09-15 a 2026-04-26 
            (425 días)", "Mejor Sharpe global: IA_PPO (1.780)". Material directo para 
            meter en la memoria del TFM y la presentación.
            
    Dónde se consume este endpoint:
        Frontend Angular (final-table.component.ts): pinta una página completa con contexto, tabla comparativa, 
        ganadores resaltados y comparaciones uno a uno. Es la que verá el tribunal en la demo.
        Memoria del TFM: cualquier captura de pantalla o tabla de la sección "resultados" sale de aquí.

    """
    from src.investor.simulation_service import run_simulation
    from src.training_drl.risk_profiles import RISK_PROFILES

    result = run_simulation(capital=10000, commission=0.001)

    if "error" in result:
        return {"available": False, "error": result["error"]}

    metrics = result.get("metrics", {})
    if not metrics:
        return {"available": False, "error": "No se pudieron calcular métricas."}

    # ─── Contexto del entrenamiento 
    # Leer universo activo y modelo de la BD para saber con qué se entrenó
    universe = universe_repo.get_active_universe(db)
    model_record = universe_repo.get_latest_model(db, model_type="ppo")

    # Perfil de riesgo usado en el entrenamiento
    risk_profile_id = None
    risk_profile_info = None
    if model_record and model_record.train_metrics:
        risk_profile_id = model_record.train_metrics.get('risk_profile', 'low_turnover')
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
            "risk_profile": risk_profile_id or 'low_turnover',
            "risk_profile_name": risk_profile_info['name'] if risk_profile_info else 'Bajo Turnover',
            "phi": risk_profile_info['phi'] if risk_profile_info else 0.02,
            "gamma": risk_profile_info['gamma'] if risk_profile_info else 0.02,
            "steps": model_record.steps if model_record else None,
            "trained_at": str(model_record.created_at)[:19] if model_record else None,
        },
        "test_period": result.get("test_period"),
        "commission": "0.1%",
        "initial_capital": "$10,000",
        "split": "80% train / 20% test",
    }

    # ─── Mejor estrategia por métrica 
    best_sharpe = max(metrics.items(), key=lambda x: x[1].get('Sharpe Ratio', -999))
    best_retorno = max(metrics.items(), key=lambda x: x[1].get('Retorno Total (%)', -999))
    best_mdd = max(metrics.items(), key=lambda x: x[1].get('Max Drawdown (%)', -999))
    best_sortino = max(metrics.items(), key=lambda x: x[1].get('Sortino Ratio', -999))

    # ─── PPO vs cada baseline 
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
        f"Perfil de riesgo: {risk_profile_info['name'] if risk_profile_info else 'Bajo Turnover'}.",
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
    
    Qué información devuelve
        Lee el CSV src/reports/expanding_window_results.csv (generado por expanding_window_validation() 
        en training_analysis.py) y lo servimos en JSON con tres bloques:

        Detalle por ventana (windows)
            Cada ventana del expanding window con:
            Fechas de train (inicio y fin) y de test (inicio y fin).
            Número de días de cada periodo.
            Métricas evaluadas en el periodo de test: Sharpe Ratio, Retorno Total (%), MDD, 
            Volatilidad anualizada.
            Ejemplo (con min_train=504, test_days=63, sobre 5 años):
        
        Resumen agregado (summary)
            Sharpe medio entre todas las ventanas + desviación estándar.
            Retorno medio.
            MDD medio.
            Cuántas ventanas tienen Sharpe > 0 (cuántas fueron rentables).
        
        Para qué sirve esto en el TFM
            Responde a la pregunta del tribunal: "¿el agente PPO funciona bien sólo en un 
            periodo concreto, o de forma robusta a lo largo del tiempo?"
            Recordatorio rápido del expanding window (vs walk-forward rolling):
                En rolling window la ventana de train tiene tamaño fijo y se desliza.
                En expanding window el train empieza siempre en el día 0 y crece. Cada 
                nueva ventana entrena con TODA la historia conocida hasta ese momento y 
                prueba en los siguientes meses. Simula mejor el caso real ("uso todo lo que 
                sé hasta hoy para predecir mañana") y genera más ventanas (~12 vs ~3-4 con 
                rolling), lo que da más robustez estadística.

        Cómo se plasma en el TFM
            Tres lecturas posibles del resumen agregado:
                Sharpe medio > 0 con desviación baja: concluimos que  el agente generaliza, funciona en 
                distintos regímenes de mercado. Es lo que defiendo.
                Sharpe medio > 0 con desviación alta: concluimos el agente funciona "de media" pero 
                hay ventanas malas. Introduzco matiz: "la política es rentable pero sensible 
                al régimen de mercado".
                Degradación monotónica (sistemática) en ventanas recientes (Sharpe baja sistemáticamente 
                conforme las ventanas avanzan en el tiempo): Es mala señal: el agente aprendió
                patrones del pasado lejano que ya no se dan. Sería un punto débil que el 
                tribunal te marcaría.

        Dónde se consume en el frontend
            validation.component.ts usa este endpoint junto con el de walk-forward para pintar la página /validacion:
            Tabs Walk-Forward / Expanding Window.
            4 graficas de resumen (Sharpe medio, retorno medio, MDD medio, % ventanas positivas).
            Gráficas Plotly: Sharpe por ventana y Retorno por ventana.
            Tabla detallada con todas las ventanas.

        Resumen de una frase
            Es la ventana de lectura para que la página /validacion muestre la robustez temporal 
            del agente PPO, sin tener que reejecutar el análisis (que tarda horas) cada vez 
            que entras al frontend.    
    """
    #todo: esta ruta en duro tenemos que revisarla
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
    #todo: ruta hardcodeada
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
    wf_running = os.path.exists('models/.walkforward.lock')
    ew_running  = os.path.exists('models/.expanding.lock')
    sa_running  = os.path.exists('models/.sensitivity.lock')

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


# ─── ENDPOINTS DE ADMINISTRADOR
# Todos requieren JWT con role='admin'
# todo: el end_date, debería ir aumentando al siguiente mes según pasen las fechas.
#podemos pensar en una clase pydantic que tenga un depends, en lugar de parametros disgregados.
@app.post("/admin/fase1/screener", tags=["Admin"])
async def run_screener(
    start_date: str = "2014-01-01",
    end_date: str = "2026-04-01",
    top_n: int = 15,
    max_per_sector: int = 3,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Screener: filtra S&P 500 a los mejores candidatos (solo admin).

    Los candidatos seleccionados se guardan en BD y se usan automáticamente como tickers por defecto en /admin/fase1/preparar-datos.
    No es necesario copiar tickers manualmente entre endpoints.

    El periodo por defecto comienza en 2014 para abarcar multiples regimenes
    de mercado (alcistas, COVID 2020, subida de tipos 2022, recuperacion 2024).
    Los criptoactivos IBIT y ETHA solo cotizan desde 2024; el data_downloader
    aplica una sustitucion por sus subyacentes (BTC-USD y ETH-USD) en el
    periodo previo a su lanzamiento, asumida como proxy academico defendible.
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
    Si no hay screener previo, usa el CORE_UNIVERSE (9 activos de respaldo).
    """
    # Resolver tickers: explícitos -> último screener -> core universe
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

    # Garantizar que IBIT y ETHA siempre están presentes - requisito del TFM:
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
        #todo: rutas en duro.
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
                                 risk_profile: str = 'low_turnover',
                                 admin: User = Depends(require_admin),
                                 db: Session = Depends(get_db)):
    """
    Lanza el entrenamiento académico del agente PPO en segundo plano.

    Rol en el TFM:
        Es el "botón rojo" de la fase 3 del pipeline. Una vez el admin ha
        ejecutado el screener (fase 1) y preparado los datos (fase 1bis),
        este endpoint dispara la fase de aprendizaje propiamente dicha:
        crea el agente PPO, lo entrena durante decenas de miles de
        episodios sobre los datos disponibles y lo deja listo para que
        los usuarios investor puedan simularlo en la fase 4.

    Por qué BackgroundTasks (asíncrono):
        Solución: la tarea se delega a `BackgroundTasks` (mecanismo nativo
        de FastAPI) y el endpoint responde inmediatamente con el
        identificador del modelo registrado. El admin puede cerrar el
        navegador y consultar `/estado` periódicamente para ver si la
        ejecución terminó (el frontend Angular hace polling automático).

    Flujo interno (5 pasos):
        1. Validación del perfil de riesgo solicitado contra el catálogo
           definido en risk_profiles.py. Si el perfil no existe, devuelve
           error 400 sin tocar nada más.
        2. Verificación de la precondición de datos: comprueba que existe
           un Universe activo en BD (lo crea `prepare_data` en fase 1).
           Si no, devuelve mensaje guiando al siguiente paso correcto.
        3. Registro del modelo en BD con estado inicial 'training' y
           vinculación al Universe activo. Esto deja traza inmediata para
           que el dashboard, el endpoint /estado y la tabla final puedan
           mostrar "hay un entrenamiento en curso" sin esperar al resultado.
        4. Creación del fichero de lock (models/.training.lock). Es la
           señal compartida entre procesos: cualquier consulta a /estado
           lo detecta y reporta `background_running=True`. El frontend lo
           usa para deshabilitar botones y mostrar banner de operación
           activa.
        5. Encolado de la tarea de entrenamiento real (`entrenar_academico`)
           en background. La closure `train_and_update_status` envuelve la
           ejecución para que, sea cual sea el resultado:
             - éxito: marca modelo como 'ready' con el perfil en train_metrics (clave para que la tabla final pueda
                        mostrar "modelo entrenado con perfil X").
             - fallo: marca modelo como 'failed' (visible en logs y BD).
             - siempre: libera el lock para desbloquear la UI.

    Sobre el parámetro `admin: User = Depends(require_admin)`:
        FastAPI ejecuta `require_admin` ANTES del cuerpo de la función.
        Esa dependencia valida el JWT del header Authorization, comprueba
        rol == 'admin' y, si todo es correcto, inyecta el User aquí. Si
        falla cualquier paso (token caducado, rol incorrecto, usuario
        deshabilitado), FastAPI corta la ejecución con HTTPException 401/403
        sin que esta función llegue a ejecutarse.

    Hiperparámetros expuestos:
        - steps (int, default=500 000):
              Pasos máximos de PPO. Es un máximo: el early stopping
              automático puede detener el entrenamiento antes si el modelo
              deja de mejorar en out-of-sample(test) (ver OverfitDetectorCallback
              en training_analysis.py). 500k es el valor calibrado para
              converger sobre datasets de 5-7 años de histórico diario.
        - patience (int, default=8):
              Evaluaciones consecutivas sin mejora en eval antes de aplicar
              early stopping. La función real ajusta esta paciencia de
              forma adaptativa (entre 5 y 15) según el ratio total_steps/
              eval_freq, garantizando que paciencia no sea ni demasiado
              estricta para datasets cortos ni demasiado laxa para largos.
        - risk_profile (str, default='low_turnover'):
              Selecciona la combinación (phi, gamma) que pondera las
              penalizaciones del reward compuesto del entorno
              (PortfolioEnv: R = Sharpe_rolling − phi·MDD − gamma·Turnover).
              Catálogo definido en risk_profiles.py:
                - low_turnover : phi=0.02, gamma=0.02 — PERFIL PRINCIPAL DEL TFM.
                                 Mejor Sharpe en el análisis de sensibilidad → es
                                 el default y el modelo final se entrena con éste.
                - balanced     : phi=0.02, gamma=0.01 — alternativa equilibrada,
                                 baseline conservador del catálogo.
                - conservative : phi=0.05, gamma=0.01 — prioriza preservar capital,
                                 penaliza drawdowns 2.5× más.
                - aggressive   : phi=0.01, gamma=0.005 — mínimas penalizaciones,
                                 máxima libertad.

    Returns
    -------
    dict
        Confirmación inmediata (NO espera a que el entrenamiento termine):
          - message: descripción humana del entrenamiento lanzado.
          - risk_profile : id del perfil usado (para trazabilidad).
          - phi, gamma : valores numéricos efectivos del perfil.
          - universe_id: id del Universe sobre el que se entrena.
          - model_id: id del TrainedModel registrado en BD (status
                           inicial 'training', se actualizará al terminar).

    Raises
    ------
    HTTPException 401/403
        Inyectado por require_admin si el JWT es inválido o el rol no es
        'admin'. Lanzado antes de ejecutar este código.

    Notas para la memoria del TFM:
        - El endpoint es la materialización práctica del enfoque "MLOps aplicado a DRL financiero": separación clara entre disparo,
          registro de metadata, ejecución larga en background y consulta posterior de estado, todo trazable en BD.
        - La elección de `risk_profile` queda persistida en TrainedModel.train_metrics, lo que permite reproducibilidad y
          comparativas posteriores (sensitivity analysis cruza con esta información para validar la robustez de la calibración).
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
    """
    Lanza la validación Walk-Forward (rolling) en segundo plano.

    Rol en el TFM:
        Es el "botón rojo"/clave de la validación temporal académica. Mientras
        que /entrenar-academico produce UN modelo final, este endpoint
        me ayuda a responder a una pregunta de tribunal: "¿el agente PPO funciona en cualquier época de mercado o solo casualmente acertó en el split
        80/20 del entrenamiento principal?".

        El walk-forward divide el histórico en ventanas deslizantes consecutivas, entrena un PPO independiente en cada una y evalúa
        en el periodo posterior. Si el Sharpe medio es positivo en TODAS las ventanas con varianza moderada, la política generaliza. Si
        oscila mucho o se degrada en ventanas recientes, no.

        Documentación: Es la metodología estándar en finanzas cuantitativas (López de Prado, 2018, "Advances in Financial Machine Learning", cap. 7).

    Por qué BackgroundTasks (asíncrono):
        El walk-forward sobre 5-7 años de datos genera 4-6 ventanas. Cada
        ventana entrena un PPO DESDE CERO — no se reutiliza el modelo
        principal — para garantizar que cada evaluación es genuinamente
        out-of-sample respecto a su propio train. Total estimado:
            n_ventanas x steps_por_ventana x tiempo_por_step ≈ 2-4 horas.
        Imposible mantener la conexión HTTP abierta tantas horas → toda la lógica se delega a BackgroundTasks.

    Flujo interno (3 pasos):
        1. Crear el fichero de lock `models/.walkforward.lock`. Es la
           señal compartida entre procesos: cualquier consulta posterior a /estado lo detecta y reporta `background_running=True`. El
           frontend Angular lo lee para deshabilitar botones y mostrar el banner "validación en curso".
        2. Envolver la función pesada `walk_forward_validation()` en una
           closure con `try/finally`. Garantía: ocurra lo que ocurra
           (éxito o excepción), el lock se libera. Sin este envoltorio,
           un fallo de la validación dejaría la UI bloqueada eternamente
           ("lock zombi").
        3. Encolar la tarea en `BackgroundTasks` con las rutas a los CSVs
           del pipeline y el número de pasos por ventana. La tarea corre
           DESPUÉS de devolver la respuesta HTTP, en el mismo proceso
           uvicorn (no es un worker externo).

    Sobre `dependencies=[Depends(require_admin)]`:
        Variante de FastAPI cuando solo importa la protección,
        no el objeto User. La dependencia se ejecuta antes del cuerpo:
        valida JWT, comprueba rol == 'admin' y, si falla, corta con
        HTTPException 401/403 sin que el código de aquí se ejecute.        

    Lo que pasa "por debajo" (dentro de walk_forward_validation):
        1. Divide el histórico en ventanas deslizantes de tamaño fijo
           (train_days=504 ≈ 2 años, test_days=252 ≈ 1 año, parámetros
           internos de la función).
        2. Para cada ventana: entrena un PPO desde cero con
           `steps_por_ventana` pasos, evalúa en el test correspondiente y
           guarda métricas (Sharpe, Retorno, MDD, fechas de train/test).
        3. Desliza la ventana 1 periodo de test (1 año) y repite hasta
           agotar el dataset.
        4. Genera dos artefactos persistentes:
             - src/reports/walk_forward_results.csv — métricas por ventana.
             - src/reports/walk_forward_analysis.png — gráfica + tabla de fechas.
        Nota técnica sobre data leakage: las features llegan ya
        normalizadas con z-score global (training_analysis.py contiene un
        TODO para recalcular stats por ventana — efecto pequeño pero
        académicamente conviene mencionarlo en la memoria).

    Cómo se consultan los resultados:
        Este endpoint NO devuelve métricas. Solo lanza el proceso. Cuando
        termine, los resultados se obtienen vía:
            GET /walk-forward/results
        que lee el CSV generado y lo sirve como JSON al frontend
        (validation.component.ts pinta el panel /validacion con tablas
        y gráficas Plotly por ventana).

    Hiperparámetros expuestos:
        - steps_por_ventana (int, default=100 000):
              Pasos de PPO POR CADA ventana, NO totales. Si el dataset
              genera 6 ventanas, son 6 entrenamientos x 100 k pasos =
              600 k pasos en total. Subir este valor mejora cada modelo
              individual a costa de tiempo; bajar acelera la validación
              pero arriesga modelos infraentrenados que producirían
              métricas pesimistas no representativas de la política real.

    Returns
    -------
    dict
        Confirmación inmediata (no espera al resultado real):
          - message : descripción humana de lo que se acaba de lanzar.

    Notas para la memoria del TFM:
        - Diferencia esencial con expanding_window: aquí la ventana de
          train tiene TAMAÑO FIJO y se desliza; en expanding_window el
          train empieza siempre en el día 0 y CRECE. Walk-forward responde a preguntas de tribunal como
          "¿funciono igual en distintas épocas?"; expanding_window
          responde "¿funciono mejor cuanta más historia tengo?". Lanzar
          ambos da una doble validación que el tribunal valora.
        - Lectura defendible del resumen agregado (devuelto por
          /walk-forward/results): Sharpe medio > 0 con desviación baja en
          todas las ventanas → la política generaliza. Sharpe medio > 0
          con desviación alta → es rentable de media pero sensible al
          régimen. Degradación monotónica en ventanas recientes → mala
          señal: el modelo aprendió patrones que ya no se dan.
    """
    # 1) Marcamos "ocupado" antes de lanzar el background — el frontend
    #    detecta este lock vía /estado y bloquea la UI.
    _create_lock('models/.walkforward.lock')

    # 2) Closure que envuelve la validación: garantiza liberación del lock
    #    SIEMPRE (try/finally), incluso si walk_forward_validation lanza
    #    excepción. Sin esto un fallo dejaría la UI bloqueada para siempre.
    def wf_with_lock(**kwargs):
        try:
            walk_forward_validation(**kwargs)
        finally:
            _remove_lock('models/.walkforward.lock')

    # 3) Encolamos la tarea pesada en background. FastAPI la ejecuta
    #    DESPUÉS de devolver la respuesta HTTP, así el cliente no espera
    #    las 2-4 horas que dura la validación completa.
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
    Lanza la validación Expanding Window en segundo plano.

    Rol en el TFM:
        Es la validación temporal "más cercana a producción real". Mientras
        el walk-forward rolling desliza una ventana de tamaño fijo (olvida
        lo viejo), el expanding window deja CRECER el train desde el día 0
        en cada iteración: "acumulo todo lo que sé y reentreno con todo".
        Eso simula exactamente lo que debo hacer en producción — reentrenar
        cada noche(o cuando corresponda) con el histórico actualizado hasta hoy.

    Pregunta del TFM que responde:
        "¿El agente PPO mejora cuanto más historia tiene, o llega un punto
         donde más datos no ayudan (e incluso confunden por cambio de
         régimen)?"

        Tres lecturas posibles del CSV resultante:
          - Sharpe creciente por ventana → más datos = mejor; conviene
            reentrenar siempre con todo.
          - Sharpe estancado → saturación; train muy largo  no aporta y solo encarece el cómputo.
          - Sharpe decreciente en ventanas recientes → los datos antiguos
            están metiendo ruido (cambio de régimen); rolling sería mejor.

    Diferencia visual frente al walk-forward rolling:

        ROLLING (walk-forward)               EXPANDING (este endpoint)

        V1: [TRAIN 2019-21]TEST              V1: [TRAIN 2019-21]TEST
        V2:    [TRAIN 2020-22]TEST           V2: [TRAIN 2019-22]TEST
        V3:       [TRAIN 2021-23]TEST        V3: [TRAIN 2019-23]TEST
        V4:          [TRAIN 2022-24]TEST     V4: [TRAIN 2019-24]TEST
                                             ...

    Por qué BackgroundTasks (asíncrono):
        Las últimas ventanas tardan más que las primeras porque entrenan
        con más datos: es decir, la duración total escala (no es lineal) con el
        número de ventanas. Estimación realista:
            n_ventanas × steps × tiempo_por_step
            ~ 12       × 100k  × ~30 ms          ≈ 6 horas
        Imposible mantener la conexión HTTP abierta tantas horas:
        la lógica se delega a BackgroundTasks

    Flujo interno (3 pasos, idéntico al walk-forward salvo nombres):
        1. Crear el fichero de lock `models/.expanding.lock`. Lock
           SEPARADO del walk-forward para que /estado pueda distinguir
           qué validación está corriendo (no interfieren entre sí si
           lanzo ambas seguidas).
        2. Envolver la función pesada `expanding_window_validation()` en
           una closure con `try/finally`. Garantía de liberación del lock
           ocurra lo que ocurra (éxito o excepción): evita "lock zombi"
           que dejaría la UI bloqueada eternamente.
        3. Encolar la tarea en `BackgroundTasks` con los CSVs del
           pipeline y los tres hiperparámetros expuestos.

    Sobre `dependencies=[Depends(require_admin)]`:
        Variante  de FastAPI cuando solo importa la protección,
        no el objeto User. La dependencia se ejecuta antes del cuerpo:
        valida JWT, comprueba rol == 'admin' y, si falla, corta con
        HTTPException 401/403 sin que el código de aquí se ejecute.

    Lo que pasa "por debajo" (dentro de expanding_window_validation):
        1. Calcula los índices [0, split, end] de cada ventana, donde el
           start ES SIEMPRE 0 (a diferencia del rolling) y `split` avanza
           `test_days` cada iteración hasta agotar el dataset.
        2. Para cada ventana: entrena un PPO desde cero con
           `steps_por_ventana` pasos sobre [0..split], evalúa en
           [split..split+test_days], y guarda métricas con sus fechas.
        3. Genera dos artefactos persistentes:
             - src/reports/expanding_window_results.csv — métricas por ventana.
             - src/reports/expanding_window_analysis.png — gráfica + tabla.

    Cómo se consultan los resultados:
        Este endpoint NO devuelve métricas. Solo lanza el proceso. Cuando
        termine, los resultados se obtienen vía:
            GET /expanding-window/results
        que lee el CSV generado y lo sirve como JSON al panel
        /validacion del frontend (validation.component.ts pinta tabla,
        gráficas Plotly por ventana y resumen agregado).

    Hiperparámetros expuestos:
        - steps_por_ventana (int, default=100 000):
              Pasos de PPO POR CADA ventana, NO totales. Si se generan 12
              ventanas, son 12 entrenamientos × 100 k pasos = 1.2 M pasos
              en total. Ojo: las últimas ventanas tienen más datos por
              ventana, así que cada step computa un poco más lento.
        - min_train_days (int, default=504 ≈ 2 años):
              Tamaño MÍNIMO del train para la primera ventana. A partir
              de ahí solo crece. 2 años es el mínimo razonable para que
              PPO vea distintos regímenes (alcista/bajista) en el train
              inicial.
        - test_days (int, default=63 ≈ 3 meses):
              Tamaño del test de cada ventana. Recomendación del tutor:
              trimestres en lugar del año del rolling, para tener más
              ventanas (~12 vs ~4) y más resolución temporal. Permite
              detectar degradación más rápido.

    Returns
    -------
    dict
        Confirmación inmediata (no espera al resultado real):
          - message : descripción humana de lo lanzado, con los tres
                      hiperparámetros usados (útil para verificar que
                      la petición llegó tal y como se quería).

    Notas para la memoria del TFM:
        - Combinar rolling + expanding es una decisión académica deliberada:
          ningún tribunal se conformaría con una sola validación
          temporal. La afirmación defendible es: "validamos con dos
          metodologías independientes; walk-forward muestra robustez
          frente a cambios de régimen, expanding window confirma que el
          modelo escala bien al acumular historia. En las N+M ventanas
          evaluadas, el Sharpe medio es positivo con desviación X. La
          política generaliza." — mucho más fuerte que cualquiera de las
          dos por separado.
        - El expanding produce ~3× más ventanas que el rolling para el
          mismo dataset → más robustez estadística.
        - test_days=63 (trimestral) es el sweet spot: lo bastante corto
          para tener muchas ventanas, lo bastante largo para que el
          Sharpe trimestral sea estadísticamente significativo (con
          test_days=21 mensual el ruido domina).
    """
    # 1) Marcamos "ocupado" antes de lanzar — el frontend detecta este
    #    lock vía /estado (campo `fase3_ew_done`) y bloquea la UI. Lock
    #    SEPARADO del walk-forward para no interferir entre ambas validaciones.
    _create_lock('models/.expanding.lock')

    # 2) Closure que envuelve la validación: try/finally garantiza que el
    #    lock se libera SIEMPRE, incluso si expanding_window_validation
    #    lanza excepción (ej. dataset corrupto, OOM, error en PPO). Sin
    #    esto un fallo dejaría la UI bloqueada permanentemente.
    def ew_with_lock(**kwargs):
        try:
            expanding_window_validation(**kwargs)
        finally:
            _remove_lock('models/.expanding.lock')

    # 3) Encolamos la tarea pesada en background. FastAPI la ejecuta
    #    DESPUÉS de devolver la respuesta HTTP, así el cliente no espera
    #    las ~6 horas que dura la validación completa.
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
    Lanza el análisis de sensibilidad de la función de recompensa en background.

    Rol en el TFM:
        Es la "prueba de robustez de la calibración". Mientras
        /entrenar-academico produce UN modelo con UN perfil de riesgo, y
        las validaciones temporales (walk-forward, expanding) prueban la
        ROBUSTEZ TEMPORAL, este endpoint prueba la ROBUSTEZ FRENTE A LOS
        HIPERPARÁMETROS DE LA RECOMPENSA. Entrena cuatro PPO independientes
        — uno por cada combinación (phi, gamma) — sobre los mismos datos
        y los compara en el mismo periodo de test.

    Pregunta del tribunal que responde:
        "¿Por qué phi=0.02 y gamma=0.01? ¿Habéis probado otros valores?
         ¿Cómo de sensible es vuestra política a esa elección?"

        Sin sensitivity, la única respuesta defendible es "los valores se
        eligieron por intuición". Con sensitivity, la respuesta es: "se
        compararon cuatro configuraciones que cubren el rango razonable
        de penalizaciones; todas obtienen Sharpe > 2.2 en out-of-sample,
        lo que evidencia que la política es robusta frente a variaciones
        de phi/gamma dentro de ese rango".

    Configuraciones evaluadas (definidas en sensitivity_analysis.CONFIGS):
        - A (actual) : phi=0.02, gamma=0.01  — baseline equilibrada.
        - B (más MDD) : phi=0.05, gamma=0.01  — preserva capital.
        - C (más turnover)  : phi=0.02, gamma=0.02  — fuerza menos rotación.
        - D (agresivo)  : phi=0.01, gamma=0.005 — máxima libertad.

        Coinciden con los cuatro perfiles de risk_profiles.py (balanced /
        conservative / low_turnover / aggressive) — el sensitivity es la
        evidencia empírica que justifica la existencia del catálogo.

    Por qué BackgroundTasks (asíncrono):
        Tarda 4× lo que un entrenamiento normal porque entrena 4 modelos
        secuencialmente desde cero. Estimación con steps_por_config=200 k
        y dataset estándar:
            4 configs × 200 000 pasos × ~30 ms ≈ 6-8 horas.
        Imposible mantener la conexión HTTP abierta tanto tiempo

    Flujo interno (3 pasos, idéntico patrón que walk-forward/expanding):
        1. Crear el fichero de lock `models/.sensitivity.lock`. Lock
           SEPARADO de los otros (training, walk-forward, expanding) para
           que /estado distinga qué proceso está corriendo y el frontend
           no confunda señales.
        2. Envolver `run_sensitivity_analysis()` en una closure con
           try/finally para garantizar liberación del lock incluso si
           uno de los cuatro entrenamientos lanza excepción.
        3. Encolar la tarea en BackgroundTasks. Solo se expone
           `total_timesteps` — los otros parámetros (rutas a CSVs,
           split_pct=0.8) usan los defaults del módulo, alineados con el
           pipeline estándar.

    Sobre `dependencies=[Depends(require_admin)]`:
        Variante de FastAPI: la dependencia se ejecuta antes
        del cuerpo, valida JWT y rol == 'admin', y corta con HTTPException
        401/403 si falla. Mantiene la firma limpia (sin parámetro `admin`
        sin usar) por coherencia con los endpoints de validación temporal
        (walk-forward, expanding).

    Lo que pasa "por debajo" (dentro de run_sensitivity_analysis):
        Para cada configuración del CONFIGS:
          1. Crear PortfolioEnv de train con el (phi, gamma) específico.
          2. Crear PortfolioEnv de eval con el mismo (phi, gamma).
          3. Entrenar PPO desde cero con `total_timesteps` pasos +
             EvalCallback (guarda el mejor modelo durante el train).
          4. Cargar el mejor modelo y backtestearlo en test.
          5. Calcular métricas (Sharpe, Retorno, MDD, Volatilidad…) y
             agregarlas al DataFrame de resultados.
        Al final genera dos artefactos:
          - src/reports/sensitivity_analysis.csv — tabla comparativa.
          - src/reports/sensitivity_analysis.png — gráfica de barras 2×2.
        Ver detalle en sensitivity_analysis.py.

    Cómo se consultan los resultados:
        Este endpoint NO devuelve métricas; solo lanza el proceso. Cuando
        termine, los resultados se obtienen vía:
            GET /sensitivity/results
        que lee el CSV generado y lo sirve como JSON al panel de admin
        del frontend (admin.component.ts pinta una tabla con la mejor
        configuración por métrica resaltada).

    Hiperparámetros expuestos:
        - steps_por_config (int, default=200 000):
              Pasos de PPO POR CADA configuración. Es la mitad de los
              500 k del entrenamiento principal: para una comparación
              relativa entre configuraciones, 200 k son suficientes
              (no necesitamos converger al óptimo absoluto, sino ver
              qué configuración funciona mejor con el mismo presupuesto
              computacional). Subir este valor mejora la calidad
              individual de cada modelo a costa de tiempo total.

    Returns
    -------
    dict
        Confirmación inmediata (no espera al resultado real):
          - message : descripción humana del análisis lanzado, con el
                      total estimado (steps × 4 configs).

    Notas para la memoria del TFM:
        - El sensitivity es el complemento académico ideal de las
          validaciones temporales. La afirmación defendible final es:
          "el modelo es robusto temporalmente (walk-forward + expanding)
          Y robusto frente a la calibración de hiperparámetros del reward
          (sensitivity); no es un afortunado punto óptimo, sino una zona
          estable de configuraciones que producen carteras competitivas".
        - El resultado clave para citar es: "todas las configuraciones
          obtienen Sharpe > 2.2 en out-of-sample". Mostrar la tabla con
          la mejor config por métrica resaltada en celdas verdes (lo que
          hace la página /admin del frontend) deja la conclusión visual.
        - Las 4 configuraciones se definen en CONFIGS (módulo
          sensitivity_analysis.py) y NO se exponen como parámetros del
          endpoint a propósito: cambiar el conjunto evaluado requiere
          decisión académica, no se debe parametrizar a la ligera. 
    """
    # 1) Marcamos "ocupado" antes de lanzar — el frontend detecta este
    #    lock vía /estado y bloquea la UI. Lock SEPARADO del resto de
    #    procesos para que /estado pueda distinguir qué tarea está corriendo.
    _create_lock('models/.sensitivity.lock')

    # 2) Closure que envuelve el análisis: try/finally garantiza que el
    #    lock se libera SIEMPRE, incluso si uno de los 4 entrenamientos
    #    lanza excepción (ej. OOM en el último, dataset corrupto, error
    #    en el callback). Sin esto un fallo dejaría la UI bloqueada.
    def sa_with_lock(**kwargs):
        try:
            run_sensitivity_analysis(**kwargs)
        finally:
            _remove_lock('models/.sensitivity.lock')

    # 3) Encolamos la tarea pesada en background. FastAPI la ejecuta
    #    DESPUÉS de devolver la respuesta HTTP, así el cliente no espera
    #    las ~6-8 horas que dura el análisis completo.
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
    """
    Ajusta el agente especulativo (GMM + K-Means) sobre el universo activo.

    Rol en el TFM:
        Crea el "agente contraste" del PPO. No aprende por reward — solo
        detecta regímenes de mercado y reglas heurísticas (ver
        speculative_agent.py). Se compara con el PPO en la tabla final
        para responder al tribunal: "¿de verdad hace falta DRL o bastaba
        con detectar patrones?".

    Por qué SÍNCRONO (sin BackgroundTasks):
        Ajustar GMM + KMeans sobre un universo de 9-20 activos toma
        segundos, no horas. No hay timeout HTTP que temer ni necesidad
        de polling desde el frontend. Devuelve directamente el resultado
        del backtest out-of-sample.

    Flujo (4 pasos):
      1. Validar precondición — debe existir un Universe activo en BD
         (lo crea /admin/fase1/preparar-datos).
      2. Cargar features y precios; aplicar split train/test (default 80/20).
      3. Crear el agente con la configuración estándar (3 regímenes,
         3 clusters, ventana rodante de 60 días) y entrenarlo SOLO con
         train para evitar lookahead bias.
      4. Persistir el modelo como pickle, registrarlo en BD como 'ready'
         vinculado al Universe activo, y devolver el resultado del
         backtest out-of-sample.

    Returns
    -------
    dict
        Confirmación con: tickers usados, universe_id, retorno % en test
        y valor final de la cartera. El frontend de admin lo muestra
        como confirmación visual al admin.
    """
    # 1) Precondición: hace falta un universo activo (lo crea fase 1).
    universe = universe_repo.get_active_universe(db)
    if universe is None:
        return {"error": "Ejecuta primero /admin/fase1/preparar-datos"}

    # 2) Cargar datasets y aplicar split temporal. Importante: el agente
    # solo verá el primer 80 % durante fit; el 20 % restante queda como
    # out-of-sample para evaluar honestamente.
    df_f = pd.read_csv('data/normalized_features.csv', index_col=0)
    df_p = pd.read_csv('data/original_prices.csv', index_col=0)
    split_idx = int(len(df_f) * split_pct)

    # 3) Crear agente con la configuración estándar del TFM.
    # n_regimes=3 (calma/transición/crisis), n_clusters=3 (defensivo/
    # neutro/agresivo), cluster_window=60 días (~3 meses). Estos valores
    # son coherentes con DEFAULT_ALLOCATION y se justifican en
    # speculative_agent.py.
    agent = SpeculativeAgent(n_regimes=3, n_clusters=3, cluster_window=60)
    agent.fit(df_f.iloc[:split_idx], df_p.iloc[:split_idx])

    # Persistencia con pickle: el agente lleva dentro un GMM (sklearn) y
    # un K-Means con scaler entrenado, todo serializable a un único .pkl.
    # Después /investor/simulate lo carga para hacer backtest sin reentrenar.
    import pickle
    os.makedirs('models', exist_ok=True)
    with open('models/speculative_gmm.pkl', 'wb') as f:
        pickle.dump(agent, f)

    # 4) Registrar modelo en BD vinculado al universo. Estado directamente
    # 'ready' porque el ajuste fue síncrono — ya terminó cuando llegamos aquí.
    # Esto da trazabilidad: la tabla trained_models guarda qué especulativo
    # corresponde a qué Universe, evitando comparaciones cruzadas absurdas.
    model_record = universe_repo.register_model(
        db, universe_id=universe.id, model_type="speculative",
        model_path="models/speculative_gmm.pkl",
    )
    universe_repo.update_model_status(db, model_record.id, "ready")

    # Backtest inmediato sobre el split de test para devolver un resultado
    # tangible al admin. No es la métrica final del TFM (esa la calcula
    # /resultados/tabla-final usando compute_metrics), pero da feedback
    # rápido del orden de magnitud del retorno conseguido.
    equity = agent.backtest(df_f.iloc[split_idx:], df_p.iloc[split_idx:])
    ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    return {
        "status": "Agente especulativo ajustado",
        "universe_id": universe.id,
        "tickers": universe.tickers,
        "retorno_test": f"{ret:.1f}%",
        "valor_final": f"${equity.iloc[-1]:,.2f}",
    }