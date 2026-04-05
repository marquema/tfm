"""
Router de autenticación: login, registro y gestión de usuarios.

Endpoints públicos (sin token):
  - POST /auth/login     → devuelve JWT token
  - POST /auth/register  → crea usuario inversor

Endpoints protegidos (requieren admin):
  - GET  /auth/users     → lista todos los usuarios
  - DELETE /auth/users/{email} → elimina un usuario
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.auth.models import User, get_db
from src.auth.auth_service import (
    authenticate_user, register_user, create_token,
    get_current_user, require_admin,
)

router = APIRouter(prefix="/auth", tags=["Autenticación"])


# ─── Modelos de request/response ─────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str = ""

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str
    full_name: str

class UserResponse(BaseModel):
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str
    last_login: str | None


# ─── Endpoints públicos ──────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):
    """
    Inicia sesión y devuelve un token JWT.

    El token debe enviarse en las peticiones posteriores como:
        Authorization: Bearer <token>

    Usa el formato estándar OAuth2PasswordRequestForm (username + password),
    donde 'username' es el email del usuario.
    """
    user = authenticate_user(db, form.username, form.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")

    token = create_token(user.email, user.role)
    return TokenResponse(
        access_token=token,
        role=user.role,
        email=user.email,
        full_name=user.full_name,
    )


@router.post("/register", response_model=UserResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Registra un nuevo usuario con rol 'investor'.

    Para crear administradores, usar el endpoint /auth/users/promote (admin only).
    """
    user = register_user(db, req.email, req.password, req.full_name, role="investor")
    return UserResponse(
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=str(user.created_at),
        last_login=str(user.last_login) if user.last_login else None,
    )


@router.get("/me", response_model=UserResponse)
def get_me(user: User = Depends(get_current_user)):
    """Retorna los datos del usuario autenticado (a partir del token JWT)."""
    return UserResponse(
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=str(user.created_at),
        last_login=str(user.last_login) if user.last_login else None,
    )


# ─── Endpoints de admin ──────────────────────────────────────────────────────

@router.get("/users", response_model=list[UserResponse])
def list_users(admin: User = Depends(require_admin),
               db: Session = Depends(get_db)):
    """Lista todos los usuarios registrados (solo admin)."""
    users = db.query(User).all()
    return [
        UserResponse(
            email=u.email, full_name=u.full_name, role=u.role,
            is_active=u.is_active, created_at=str(u.created_at),
            last_login=str(u.last_login) if u.last_login else None,
        )
        for u in users
    ]


@router.delete("/users/{email}")
def delete_user(email: str,
                admin: User = Depends(require_admin),
                db: Session = Depends(get_db)):
    """Elimina un usuario por email (solo admin)."""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if user.email == admin.email:
        raise HTTPException(status_code=400, detail="No puedes eliminarte a ti mismo")
    db.delete(user)
    db.commit()
    return {"detail": f"Usuario {email} eliminado"}
