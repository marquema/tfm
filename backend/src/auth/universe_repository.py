"""
Repositorio de acceso a datos para universos, modelos y simulaciones.

Capa de abstracción entre la lógica de negocio y la base de datos.
Todas las operaciones de CRUD pasan por aquí — ningún endpoint ni servicio
hace queries directas a SQLAlchemy. Esto permite:

  1. Cambiar el motor de BD (SQLite → PostgreSQL) sin tocar lógica de negocio
  2. Testear con mocks sin necesitar una BD real
  3. Centralizar validaciones de integridad referencial

Fallback: si la BD está vacía pero existe un universe_config.json legacy,
lo importa automáticamente para mantener retrocompatibilidad.
"""

import os
import json
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from src.auth.models import Universe, TrainedModel, Simulation, User, ScreenerResult


_LEGACY_CONFIG_PATH = os.path.join('data', 'universe_config.json')


# ─── Universos ────────────────────────────────────────────────────────────────

def create_universe(db: Session, tickers: list, start_date: str, end_date: str,
                    n_features: int, n_days: int, created_by: str = None) -> Universe:
    """
    Crea un nuevo universo de activos y lo marca como activo.

    Desactiva todos los universos anteriores — solo puede haber uno activo.
    Los modelos entrenados para universos anteriores siguen existiendo en BD
    pero no se usarán en nuevas simulaciones.

    Parameters
    ----------
    db          : sesión de SQLAlchemy
    tickers     : lista de tickers ['IVV', 'BND', ...]
    start_date  : fecha inicio del dataset 'YYYY-MM-DD'
    end_date    : fecha fin del dataset
    n_features  : columnas en normalized_features.csv
    n_days      : filas (días de trading)
    created_by  : email del admin que ejecutó la preparación de datos

    Returns
    -------
    Universe creado
    """
    # Desactivar universos anteriores
    db.query(Universe).filter(Universe.is_active == True).update({"is_active": False})

    universe = Universe(
        tickers=sorted(tickers),
        n_assets=len(tickers),
        start_date=start_date,
        end_date=end_date,
        n_features=n_features,
        n_days=n_days,
        is_active=True,
        created_by=created_by,
    )
    db.add(universe)
    db.commit()
    db.refresh(universe)
    return universe


def get_active_universe(db: Session) -> Optional[Universe]:
    """
    Retorna el universo activo actual.

    Si no hay ninguno en BD pero existe universe_config.json legacy,
    lo importa automáticamente (migración transparente).

    Returns
    -------
    Universe o None si no hay datos preparados.
    """
    universe = db.query(Universe).filter(Universe.is_active == True).first()

    # Fallback: migrar desde JSON legacy si la BD está vacía
    if universe is None and os.path.exists(_LEGACY_CONFIG_PATH):
        universe = _migrate_from_json(db)

    return universe


def get_active_tickers(db: Session) -> Optional[list]:
    """Retorna los tickers del universo activo, o None si no hay datos."""
    universe = get_active_universe(db)
    return universe.tickers if universe else None


def get_universe_history(db: Session) -> list[Universe]:
    """Retorna todos los universos (activos e inactivos), ordenados por fecha."""
    return db.query(Universe).order_by(Universe.created_at.desc()).all()


# ─── Modelos entrenados ───────────────────────────────────────────────────────

def register_model(db: Session, universe_id: int, model_type: str,
                   model_path: str, steps: int = None) -> TrainedModel:
    """
    Registra un nuevo modelo entrenado vinculado a un universo.

    Parameters
    ----------
    universe_id : ID del universo con el que se entrenó
    model_type  : 'ppo' o 'speculative'
    model_path  : ruta al fichero del modelo
    steps       : pasos de entrenamiento (solo para PPO)

    Returns
    -------
    TrainedModel creado con status='training'
    """
    model = TrainedModel(
        universe_id=universe_id,
        model_type=model_type,
        model_path=model_path,
        status="training",
        steps=steps,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def update_model_status(db: Session, model_id: int, status: str,
                        best_eval: float = None, metrics: dict = None):
    """Actualiza el estado de un modelo tras el entrenamiento."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if model:
        model.status = status
        if best_eval is not None:
            model.best_eval = best_eval
        if metrics is not None:
            model.train_metrics = metrics
        db.commit()


def get_latest_model(db: Session, model_type: str,
                     universe_id: int = None) -> Optional[TrainedModel]:
    """
    Retorna el último modelo entrenado de un tipo, opcionalmente filtrado
    por universo.

    Parameters
    ----------
    model_type  : 'ppo' o 'speculative'
    universe_id : si se especifica, solo busca modelos de ese universo.
                  Si None, busca en el universo activo.
    """
    query = db.query(TrainedModel).filter(
        TrainedModel.model_type == model_type,
        TrainedModel.status == "ready",
    )

    if universe_id:
        query = query.filter(TrainedModel.universe_id == universe_id)
    else:
        active = get_active_universe(db)
        if active:
            query = query.filter(TrainedModel.universe_id == active.id)

    return query.order_by(TrainedModel.created_at.desc()).first()


def validate_model_compatibility(db: Session, model_type: str) -> dict:
    """
    Valida que existe un modelo compatible con el universo activo.

    Returns
    -------
    dict con:
      - 'compatible': True/False
      - 'error': mensaje de error si no compatible
      - 'model': TrainedModel si compatible
      - 'universe': Universe activo
    """
    universe = get_active_universe(db)
    if universe is None:
        return {
            "compatible": False,
            "error": "No hay datos preparados. Ejecuta /admin/fase1/preparar-datos",
        }

    model = get_latest_model(db, model_type, universe_id=universe.id)
    if model is None:
        # Buscar si hay modelo de otro universo para dar mensaje más claro
        any_model = get_latest_model(db, model_type)
        if any_model:
            old_universe = db.query(Universe).filter(Universe.id == any_model.universe_id).first()
            return {
                "compatible": False,
                "error": (
                    f"El modelo {model_type} fue entrenado con un universo diferente.\n"
                    f"  Universo del modelo: {old_universe.tickers if old_universe else '?'}\n"
                    f"  Universo actual:     {universe.tickers}\n"
                    f"  Solución: re-entrena el modelo con /admin/fase3/entrenar-academico"
                ),
            }
        return {
            "compatible": False,
            "error": f"No hay modelo {model_type} entrenado. Ejecuta /admin/fase3/entrenar-academico",
        }

    return {"compatible": True, "model": model, "universe": universe}


# ─── Simulaciones ─────────────────────────────────────────────────────────────

def save_simulation(db: Session, user_id: int, universe_id: int,
                    capital: float, commission: float,
                    results: dict) -> Simulation:
    """Guarda el resultado de una simulación en el historial del inversor."""
    # Simplificar los resultados para no guardar arrays enormes en la BD
    summary = {}
    if "metrics" in results:
        summary["metrics"] = results["metrics"]
    if "test_period" in results:
        summary["test_period"] = results["test_period"]

    sim = Simulation(
        user_id=user_id,
        universe_id=universe_id,
        capital=capital,
        commission=commission,
        results_json=summary,
    )
    db.add(sim)
    db.commit()
    db.refresh(sim)
    return sim


def get_user_simulations(db: Session, user_id: int) -> list[Simulation]:
    """Retorna el historial de simulaciones de un inversor."""
    return (db.query(Simulation)
              .filter(Simulation.user_id == user_id)
              .order_by(Simulation.created_at.desc())
              .all())


# ─── Screener ─────────────────────────────────────────────────────────────────

def save_screener_result(db: Session, candidates: list, start_date: str,
                         end_date: str, filters: dict = None,
                         created_by: str = None) -> ScreenerResult:
    """
    Guarda el resultado de un screener y lo marca como activo.

    Desactiva resultados anteriores — solo el último screener se usa como
    default para preparar-datos.

    Parameters
    ----------
    db          : sesión de SQLAlchemy
    candidates  : lista de tickers seleccionados por el screener
    start_date  : fecha inicio usada en el screener
    end_date    : fecha fin usada
    filters     : parámetros del screener (top_n, max_per_sector, etc.)
    created_by  : email del admin

    Returns
    -------
    ScreenerResult creado
    """
    db.query(ScreenerResult).filter(ScreenerResult.is_active == True).update({"is_active": False})

    result = ScreenerResult(
        candidates=sorted(candidates),
        n_candidates=len(candidates),
        start_date=start_date,
        end_date=end_date,
        filters_used=filters,
        is_active=True,
        created_by=created_by,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def get_active_screener(db: Session) -> Optional[ScreenerResult]:
    """Retorna el último resultado de screener activo, o None si no hay."""
    return db.query(ScreenerResult).filter(ScreenerResult.is_active == True).first()


def get_default_tickers(db: Session) -> list:
    """
    Retorna los tickers por defecto para preparar-datos.

    Prioridad:
      1. Último screener activo (si existe)
      2. CORE_UNIVERSE del asset_registry (fallback estático)

    Esta función resuelve el problema de tener que copiar tickers manualmente:
    si el admin ejecutó el screener, preparar-datos los recoge automáticamente.
    """
    screener = get_active_screener(db)
    if screener and screener.candidates:
        return screener.candidates

    # Fallback: universo core estático
    from src.pipeline_getdata.asset_registry import get_tickers
    return get_tickers('core')


# ─── Migración legacy ─────────────────────────────────────────────────────────

def _migrate_from_json(db: Session) -> Optional[Universe]:
    """
    Importa universe_config.json legacy a la BD si existe.

    Se ejecuta una sola vez — tras importar, los siguientes accesos
    ya usan la BD directamente. Esto garantiza retrocompatibilidad
    con datos preparados antes de implementar la BD.
    """
    try:
        with open(_LEGACY_CONFIG_PATH, 'r') as f:
            config = json.load(f)

        universe = Universe(
            tickers=config.get("tickers", []),
            n_assets=config.get("n_assets", len(config.get("tickers", []))),
            start_date=config.get("start_date", ""),
            end_date=config.get("end_date", ""),
            n_features=config.get("n_features"),
            n_days=config.get("n_days"),
            is_active=True,
            created_by="migrado_desde_json",
        )
        db.add(universe)
        db.commit()
        db.refresh(universe)
        print(f"  [MIGRACIÓN] universe_config.json importado a BD (id={universe.id})")
        return universe
    except Exception as e:
        print(f"  [AVISO] No se pudo migrar universe_config.json: {e}")
        return None
