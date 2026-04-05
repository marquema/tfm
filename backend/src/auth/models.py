"""
Modelos de base de datos para autenticación y gestión de usuarios.

Usa SQLAlchemy con SQLite como almacenamiento. SQLite es suficiente para el TFM
y se puede migrar a PostgreSQL en producción sin cambiar el código de los modelos.

La tabla 'users' almacena:
  - Credenciales (email + password hasheado con bcrypt)
  - Rol (admin / investor) para control de acceso por endpoint
  - Timestamps de creación y último login
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# Base de datos SQLite en el directorio data/ del backend
_DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'users.db')
_DB_URL = f"sqlite:///{os.path.abspath(_DB_PATH)}"

engine = create_engine(_DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    """
    Modelo de usuario de la plataforma.

    Roles:
      - 'admin':    acceso completo (screener, entrenamiento, gestión de usuarios)
      - 'investor': acceso a simulación personalizada y visualización de resultados
    """
    __tablename__ = "users"

    email      = Column(String, primary_key=True, index=True)
    hashed_pwd = Column(String, nullable=False)
    full_name  = Column(String, default="")
    role       = Column(String, default="investor")  # 'admin' o 'investor'
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


def init_db():
    """
    Crea las tablas en la base de datos si no existen.

    Se llama al arrancar la API. Si la tabla 'users' ya existe, no hace nada.
    Si no existe ningún usuario admin, crea uno por defecto.
    """
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    Generador de sesiones de base de datos para usar con Depends() de FastAPI.

    Garantiza que la sesión se cierra correctamente tras cada petición,
    incluso si hay errores.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
