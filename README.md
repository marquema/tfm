# TFM — Optimización Dinámica de Carteras Híbridas de ETFs mediante Deep Reinforcement Learning

Monorepo del Trabajo Final de Máster. Contiene el backend de IA (Python/FastAPI),
el frontend web (Angular) y placeholders para evolución futura (Vue, app móvil).

---

## Estructura del monorepo

```
Impl_tfm/
├── backend/                         # Python — IA, API, pipeline de datos
│   ├── .venv/                       #   entorno virtual Python (no se sube a git)
│   ├── main.py                      #   API FastAPI (entry point)
│   ├── requirements.txt
│   ├── src/
│   │   ├── training_drl/            #   PPO: entorno, entrenamiento, validación
│   │   ├── unsupervised/            #   GMM + KMeans: agente especulativo
│   │   ├── pipeline_getdata/        #   descarga, features, screener, registro
│   │   ├── feature_engineering/     #   indicadores técnicos y estadísticos
│   │   ├── benchmarking/            #   baselines (Equal Weight, 60/40, Markowitz)
│   │   └── reports/                 #   dashboard Streamlit, results viewer
│   ├── data/                        #   CSVs generados (no se sube a git)
│   └── models/                      #   modelos entrenados (no se sube a git)
│
├── frontend-angular/                # Angular — dashboard web en tiempo real
│   ├── src/
│   ├── package.json
│   └── angular.json
│
├── frontend-vue/                    # Vue 3 — evolución futura
│   └── README.md
│
├── mobile/                          # App móvil — evolución futura
│   └── README.md
│
├── .gitignore
└── README.md                        # este fichero
```

---

## Backend (Python)

### Instalación

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

### Arrancar la API

```bash
cd backend
uvicorn main:app --reload
```

API en `http://localhost:8000`. Documentación automática en `http://localhost:8000/docs`.

### Credenciales por defecto

Al arrancar por primera vez, se crea automáticamente un usuario administrador:

| Campo | Valor |
|-------|-------|
| Email | `admin@tfm.com` |
| Password | `admin123` |
| Rol | `admin` |

> Cambiar estas credenciales en producción. Se definen en `main.py` → `startup()`.

### Endpoints de la API

**Públicos (sin autenticación):**
```
POST /auth/login                        ← devuelve JWT token
POST /auth/register                     ← registrar nuevo inversor
GET  /universo                          ← diccionario de activos
GET  /estado                            ← estado del sistema
```

**Administrador (requieren JWT con role=admin):**
```
POST /admin/fase1/screener              ← filtra S&P 500 a 10-15 candidatos
POST /admin/fase1/preparar-datos        ← descarga + features + normalización
GET  /admin/fase2/validar-datos         ← verificar integridad
POST /admin/fase3/entrenar-academico    ← entrenar PPO (1M pasos)
POST /admin/fase3/walk-forward          ← validación temporal
POST /admin/fase4/ajustar-especulativo  ← GMM + KMeans (segundos)
GET  /auth/users                        ← listar usuarios
DELETE /auth/users/{email}              ← eliminar usuario
```

**Inversor (requieren JWT con role=investor o admin):**
```
GET  /investor/strategies               ← estrategias disponibles
POST /investor/simulate                 ← backtest personalizado (capital, comisión)
GET  /auth/me                           ← datos del usuario autenticado
```

### Dashboard Streamlit

```bash
cd backend
.venv\Scripts\activate
streamlit run src/reports/app_dashboard.py
```

### Backtest por consola

```bash
cd backend
.venv\Scripts\activate
python src/reports/results_viewer.py
```

---

## Frontend Angular

### Instalación

```bash
cd frontend-angular
npm install
```

### Desarrollo

```bash
cd frontend-angular
ng serve
```

Se abre en `http://localhost:4200`. Se conecta al backend en `http://localhost:8000`.

---

## Estrategias comparadas

| Estrategia | Tipo | Descripción |
|---|---|---|
| IA PPO (DRL) | Deep Reinforcement Learning | Agente que aprende por experiencia |
| Especulativo (GMM+KMeans) | No supervisado | Regímenes + clustering dinámico |
| Equal Weight | Baseline | Mismo peso a todos, rebalanceo mensual |
| Buy & Hold | Baseline | Comprar y no tocar |
| Cartera 60/40 | Baseline | Clásica renta variable / fija |
| Markowitz MV | Baseline | Media-varianza optimizada |

---

## Función de recompensa del agente PPO

$$R_t = \text{Sharpe}_{20d}(t) - \phi \cdot \text{MDD}(t) - \gamma \cdot \text{Turnover}(t)$$

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| φ (phi) | 0.02 | Penalización por drawdown |
| γ (gamma) | 0.01 | Penalización por rotación de cartera |
| Ventana Sharpe | 20 días | Horizonte para Sharpe rolling |
