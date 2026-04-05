"""
Servicio de autenticación: registro, login y validación de tokens JWT.

Seguridad implementada:
  - Passwords: hash con bcrypt (nunca se almacenan en texto plano).
    bcrypt incluye salt automático, lo que protege contra rainbow tables.
  - Tokens: JWT firmado con HS256 (clave secreta + expiración configurable).
    El token contiene el email y el rol del usuario, lo que permite al backend
    verificar permisos sin consultar la base de datos en cada petición.
  - Expiración: 24 horas por defecto. En producción se reduciría a 1-2 horas
    con refresh tokens.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.auth.models import User, get_db

# ─── Configuración de seguridad ──────────────────────────────────────────────
# En producción, SECRET_KEY vendría de una variable de entorno (nunca hardcodeada).
# Para el TFM es aceptable tenerla aquí.
SECRET_KEY = "tfm-drl-portfolio-2026-secret-key-change-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

# bcrypt para hash de passwords
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Esquema OAuth2 para extraer el token del header Authorization: Bearer <token>
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ─── Funciones de password ───────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Genera el hash bcrypt de un password en texto plano."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Verifica si un password en texto plano coincide con su hash bcrypt."""
    return _pwd_context.verify(plain, hashed)


# ─── Funciones de JWT ────────────────────────────────────────────────────────

def create_token(email: str, role: str) -> str:
    """
    Genera un token JWT firmado con el email y rol del usuario.

    El token expira tras TOKEN_EXPIRE_HOURS horas. El frontend debe
    enviarlo en cada petición como: Authorization: Bearer <token>

    Parameters
    ----------
    email : email del usuario (se usa como 'sub' del JWT)
    role  : 'admin' o 'investor'

    Returns
    -------
    str : token JWT firmado
    """
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {
        "sub": email,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decodifica y valida un token JWT.

    Raises
    ------
    HTTPException 401 si el token es inválido, expirado o no contiene los campos requeridos.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        role  = payload.get("role")
        if email is None or role is None:
            raise HTTPException(status_code=401, detail="Token inválido: faltan campos")
        return {"email": email, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido o expirado")


# ─── Dependencias FastAPI ────────────────────────────────────────────────────

def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)) -> User:
    """
    Dependencia de FastAPI que extrae el usuario actual del token JWT.

    Uso en endpoints:
        @app.get("/ruta")
        def mi_endpoint(user: User = Depends(get_current_user)):
            ...

    Raises
    ------
    HTTPException 401 si el token es inválido o el usuario no existe.
    """
    data = decode_token(token)
    user = db.query(User).filter(User.email == data["email"]).first()
    if user is None:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Usuario desactivado")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """
    Dependencia que exige rol de administrador.

    Uso: @app.get("/admin/ruta", dependencies=[Depends(require_admin)])

    Raises
    ------
    HTTPException 403 si el usuario no es admin.
    """
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Acceso restringido a administradores")
    return user


def require_investor(user: User = Depends(get_current_user)) -> User:
    """
    Dependencia que exige rol de inversor (o admin, que también puede simular).

    Raises
    ------
    HTTPException 403 si el usuario no tiene rol válido.
    """
    if user.role not in ("investor", "admin"):
        raise HTTPException(status_code=403, detail="Acceso restringido a inversores")
    return user


# ─── Operaciones de usuario ──────────────────────────────────────────────────

def register_user(db: Session, email: str, password: str,
                  full_name: str = "", role: str = "investor") -> User:
    """
    Registra un nuevo usuario en la base de datos.

    Parameters
    ----------
    db       : sesión de SQLAlchemy
    email    : email del usuario (será su identificador único)
    password : password en texto plano (se hashea antes de guardar)
    full_name: nombre completo (opcional)
    role     : 'admin' o 'investor' (por defecto 'investor')

    Returns
    -------
    User : el usuario creado

    Raises
    ------
    HTTPException 400 si el email ya está registrado.
    """
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email ya registrado")

    user = User(
        email=email,
        hashed_pwd=hash_password(password),
        full_name=full_name,
        role=role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """
    Verifica las credenciales de un usuario.

    Returns
    -------
    User si las credenciales son correctas, None si no lo son.
    """
    user = db.query(User).filter(User.email == email).first()
    if user is None or not verify_password(password, user.hashed_pwd):
        return None
    # Actualizar último login
    user.last_login = datetime.utcnow()
    db.commit()
    return user
