"""
Router del inversor: simulación personalizada de carteras.

Endpoints protegidos con JWT (requieren rol 'investor' o 'admin'):
  - GET  /investor/strategies  → estrategias disponibles
  - POST /investor/simulate    → ejecuta backtest personalizado
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.auth.models import User
from src.auth.auth_service import require_investor
from src.investor.simulation_service import get_available_strategies, run_simulation

router = APIRouter(prefix="/investor", tags=["Inversor"])


class SimulationRequest(BaseModel):
    capital: float = 10000
    commission: float = 0.001
    split_pct: float = 0.8


@router.get("/strategies")
def list_strategies(user: User = Depends(require_investor)):
    """
    Lista las estrategias de inversión disponibles.

    Cada estrategia indica si está disponible (modelo entrenado) o no.
    El inversor puede usar esta información para decidir qué simular.
    """
    return {
        "strategies": get_available_strategies(),
        "user": user.email,
    }


@router.post("/simulate")
def simulate(req: SimulationRequest,
             user: User = Depends(require_investor)):
    """
    Ejecuta una simulación de backtest con el modelo PPO y baselines.

    El inversor elige el capital inicial y la comisión. Los activos son
    los del universo configurado por el administrador (los que se usaron
    para entrenar el modelo PPO).

    Retorna: métricas comparativas, equity curves, pesos del PPO por día,
    y período de test. El frontend Angular usa estos datos para pintar
    las gráficas interactivas.
    """
    result = run_simulation(
        capital=req.capital,
        commission=req.commission,
        split_pct=req.split_pct,
    )
    result["user"] = user.email
    return result
