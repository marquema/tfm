"""
Modelos de base de datos del TFM.

Arquitectura desacoplada del motor de BD:
  - Los modelos usan SQLAlchemy ORM puro (sin SQL directo ni funciones específicas
    de SQLite). Esto permite migrar a PostgreSQL, MySQL o cualquier motor soportado
    por SQLAlchemy cambiando únicamente la variable DATABASE_URL.
  - La conexión se configura en un solo punto (_DB_URL). En producción se leería
    de una variable de entorno: os.environ.get('DATABASE_URL', 'sqlite:///...')

Tablas:
  - users:          credenciales y roles (admin / investor)
  - universes:      configuraciones de universos de activos (qué tickers, fechas, etc.)
  - trained_models: modelos entrenados vinculados a un universo concreto
  - simulations:    historial de simulaciones ejecutadas por los inversores

Relaciones:
  Universe 1──N TrainedModel   (un universo puede tener varios modelos entrenados)
  Universe 1──N Simulation     (las simulaciones usan un universo concreto)
  User     1──N Simulation     (cada inversor tiene su historial de simulaciones)
"""

import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, Boolean, Text, ForeignKey, JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# ─── Configuración de conexión ────────────────────────────────────────────────
# Cambiar a PostgreSQL en producción:
#   DATABASE_URL = "postgresql://user:pass@host:5432/tfm_db"
# El resto del código no necesita modificarse.

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tfm.db')
_DB_URL  = os.environ.get('DATABASE_URL', f"sqlite:///{os.path.abspath(_DB_PATH)}")

engine = create_engine(
    _DB_URL, echo=False,
    # check_same_thread solo aplica a SQLite; PostgreSQL lo ignora
    connect_args={"check_same_thread": False} if 'sqlite' in _DB_URL else {},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


# ─── Modelos ──────────────────────────────────────────────────────────────────

class User(Base):
    """
    Usuario de la plataforma.

    Roles:
      - 'admin':    acceso completo (screener, entrenamiento, gestión de usuarios)
      - 'investor': simulación personalizada y visualización de resultados
    """
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    email      = Column(String(255), unique=True, index=True, nullable=False)
    hashed_pwd = Column(String(255), nullable=False)
    full_name  = Column(String(255), default="")
    role       = Column(String(50), default="investor")
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relación: un usuario tiene muchas simulaciones
    simulations = relationship("Simulation", back_populates="user")


class ScreenerResult(Base):
    """
    Resultado de una ejecución del screener de mercado.

    Almacena los candidatos seleccionados para que el siguiente paso
    (preparar-datos) los use como default sin que el admin tenga que
    copiar tickers manualmente.

    Solo se mantiene un resultado activo (el último). Los anteriores
    quedan como historial pero no se usan como default.
    """
    __tablename__ = "screener_results"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    candidates   = Column(JSON, nullable=False)      # ['AAPL', 'MSFT', ...]
    n_candidates = Column(Integer, nullable=False)
    start_date   = Column(String(10), nullable=False)
    end_date     = Column(String(10), nullable=False)
    filters_used = Column(JSON, nullable=True)       # {top_n, max_per_sector, ...}
    is_active    = Column(Boolean, default=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    created_by   = Column(String(255), nullable=True)


class Universe(Base):
    """
    Configuración de un universo de activos.

    Cada vez que el admin ejecuta /admin/fase1/preparar-datos con un conjunto
    de tickers, se crea un registro de Universe. Los modelos entrenados y las
    simulaciones quedan vinculados a este registro, garantizando coherencia.

    Si se cambian los tickers, se crea un nuevo Universe — los modelos del
    anterior quedan marcados como incompatibles automáticamente.
    """
    __tablename__ = "universes"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    tickers    = Column(JSON, nullable=False)       # ['IVV', 'BND', ...]
    n_assets   = Column(Integer, nullable=False)
    start_date = Column(String(10), nullable=False)  # 'YYYY-MM-DD'
    end_date   = Column(String(10), nullable=False)
    n_features = Column(Integer, nullable=True)
    n_days     = Column(Integer, nullable=True)
    is_active  = Column(Boolean, default=True)       # solo uno activo a la vez
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)  # email del admin que lo creó

    # Relaciones
    trained_models = relationship("TrainedModel", back_populates="universe")
    simulations    = relationship("Simulation", back_populates="universe")


class TrainedModel(Base):
    """
    Registro de un modelo entrenado vinculado a un universo.

    Almacena metadatos del entrenamiento (tipo de modelo, pasos, métricas)
    y la ruta al fichero del modelo (.zip para PPO, .pkl para especulativo).
    """
    __tablename__ = "trained_models"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    universe_id   = Column(Integer, ForeignKey("universes.id"), nullable=False)
    model_type    = Column(String(50), nullable=False)  # 'ppo', 'speculative'
    model_path    = Column(String(500), nullable=False)  # ruta al fichero
    status        = Column(String(50), default="training")  # training, ready, failed
    steps         = Column(Integer, nullable=True)
    best_eval     = Column(Float, nullable=True)
    train_metrics = Column(JSON, nullable=True)  # {sharpe, retorno, ...}
    created_at    = Column(DateTime, default=datetime.utcnow)

    # Relación: pertenece a un universo
    universe = relationship("Universe", back_populates="trained_models")


class Simulation(Base):
    """
    Registro de una simulación ejecutada por un inversor.

    Guarda los parámetros de entrada (capital, comisión) y un resumen de los
    resultados para que el inversor pueda consultar su historial sin tener
    que reejecutar el backtest.
    """
    __tablename__ = "simulations"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    universe_id  = Column(Integer, ForeignKey("universes.id"), nullable=False)
    capital      = Column(Float, default=10000)
    commission   = Column(Float, default=0.001)
    results_json = Column(JSON, nullable=True)  # resumen de métricas
    created_at   = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    user     = relationship("User", back_populates="simulations")
    universe = relationship("Universe", back_populates="simulations")


# ─── Inicialización ──────────────────────────────────────────────────────────

def init_db():
    """
    Crea todas las tablas en la base de datos si no existen.

    Seguro de ejecutar múltiples veces — SQLAlchemy solo crea las tablas
    que faltan, no modifica las existentes. En producción se usaría Alembic
    para migraciones de esquema.
    """
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Generador de sesiones de BD para FastAPI Depends().

    Garantiza que la sesión se cierra correctamente tras cada petición.
    Compatible con cualquier motor SQLAlchemy (SQLite, PostgreSQL, etc.).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
