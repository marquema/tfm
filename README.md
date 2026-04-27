# TFM — Optimización Dinámica de Carteras Híbridas de ETFs mediante Deep Reinforcement Learning

> Trabajo Final de Máster que evalúa si un agente DRL (PPO) puede competir
> con las estrategias clásicas de gestión de cartera al incorporar
> criptoactivos al universo tradicional de inversión.

---

## Hipótesis y objetivo

Las técnicas clásicas de gestión de cartera (Markowitz, 60/40, Buy & Hold,
Equal Weight) dejan de capturar la dinámica de mercados modernos cuando se
incorporan activos no tradicionales como Bitcoin (IBIT) o Ethereum (ETHA).
El TFM **demuestra empíricamente** si un agente Proximal Policy Optimization
(PPO), entrenado con un reward compuesto (Sharpe rolling penalizado por
drawdown y turnover), supera de forma robusta a esas baselines en
out-of-sample(test), en distintos regímenes de mercado y bajo distintas
calibraciones del reward.

La validación se sostiene sobre tres ejes:

- **Robustez temporal** — walk-forward y expanding window.
- **Robustez frente a la calibración** — análisis de sensibilidad sobre
  4 perfiles de riesgo.
- **Robustez por régimen** — segmentación de métricas por calma /
  transición / crisis.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (UI capa final)                    │
│  ┌──────────────────────┐         ┌──────────────────────────────┐  │
│  │  Angular (producto)  │         │  Streamlit (exploración)     │  │
│  │  /admin /investor    │         │  Dashboard interactivo       │  │
│  │  /resultados ...     │         │  para defensa y debug        │  │
│  └──────────┬───────────┘         └──────────┬───────────────────┘  │
└─────────────┼────────────────────────────────┼─────────────────────-┘
              │     REST + JWT                 │ lectura directa
┌─────────────▼────────────────────────────────▼─────────────────────┐
│                       BACKEND  FastAPI + uvicorn                   │
│                                                                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐     │
│  │ training_drl │  │  unsupervised   │  │ pipeline_getdata   │     │
│  │  (PPO + val) │  │  (GMM + KMeans) │  │  (screener,        │     │
│  │              │  │                 │  │   features,        │     │
│  │              │  │                 │  │   data sources)    │     │
│  └──────────────┘  └─────────────────┘  └────────────────────┘     │
│                                                                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐     │
│  │ benchmarking │  │     reports     │  │       auth         │     │
│  │  (baselines) │  │   (Streamlit,   │  │   (JWT, BD,        │     │
│  │              │  │   results_view) │  │   roles)           │     │
│  └──────────────┘  └─────────────────┘  └────────────────────┘     │
│                                                                    │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ persistencia
              ┌───────────────▼─────────────────┐
              │      SQLite (SQLAlchemy ORM)    │
              │   users, universes,             │
              │   trained_models, simulations,  │
              │   screener_results              │
              └─────────────────────────────────┘
              ┌─────────────────────────────────┐
              │   Filesystem (artefactos)       │
              │   data/*.csv, models/*.zip,     │
              │   src/reports/*.png, *.csv      │
              └─────────────────────────────────┘
```

Patrón clave: **separación de responsabilidades**. Cada módulo se puede
sustituir sin afectar al resto (ej. `data_source.py` permite cambiar Yahoo
Finance por una API real-time sin tocar el entorno PPO).

---

## Estructura del monorepo

```
Impl_tfm/
├── backend/                              # Python — IA, API, pipeline
│   ├── main.py                           #   API FastAPI (entry point)
│   ├── requirements.txt
│   ├── src/
│   │   ├── auth/                         #   JWT, roles, modelos BD
│   │   ├── pipeline_getdata/             #   screener, descarga, registro
│   │   │   ├── data_downloader.py        #     orquesta fase 1 (datos)
│   │   │   ├── data_source.py            #     abstracción Strategy
│   │   │   ├── market_screener.py        #     filtro S&P 500 + cripto
│   │   │   ├── asset_registry.py         #     metadatos de activos
│   │   │   └── universe_config.py        #     legacy JSON (deprecated)
│   │   ├── feature_engineering/
│   │   │   └── data_features.py          #   RSI, MACD, momentum, beta…
│   │   ├── training_drl/
│   │   │   ├── environment_trading.py    #   Gymnasium env (PPO)
│   │   │   ├── training_analysis.py      #   train + walk-forward + expanding
│   │   │   ├── sensitivity_analysis.py   #   sensibilidad de phi/gamma
│   │   │   ├── risk_profiles.py          #   4 perfiles de reward
│   │   │   └── regime_analysis.py        #   evaluación por régimen
│   │   ├── unsupervised/
│   │   │   ├── speculative_agent.py      #   coordinador GMM + K-Means
│   │   │   ├── regime_hmm.py             #   detector de régimen (GMM)
│   │   │   └── asset_clustering.py       #   K-Means rolling de activos
│   │   ├── benchmarking/
│   │   │   └── baselines.py              #   Equal Weight, 60/40, Markowitz…
│   │   ├── investor/
│   │   │   ├── simulation_service.py     #   backtest del inversor
│   │   │   └── investor_router.py        #   endpoints /investor/*
│   │   └── reports/
│   │       ├── app_dashboard.py          #   Streamlit (defensa)
│   │       └── results_viewer.py         #   batch CLI (memoria)
│   ├── data/                             #   CSVs (no se sube a git)
│   └── models/                           #   modelos entrenados (no se sube)
│
├── frontend-angular/                     # Angular — UI producto
│   └── src/app/pages/
│       ├── login/        admin/          dashboard/
│       ├── universe/     status/         final-table/
│       ├── validation/   investor/simulator/
│       └── investor/results/
│
├── frontend-vue/                         # Vue 3 — placeholder evolución futura
├── mobile/                               # App móvil — placeholder evolución futura
└── README.md                             # este fichero
```

---

## Quick start (5 minutos)

```bash
# 1. Backend
cd backend
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate               # Linux/macOS
pip install -r requirements.txt
uvicorn main:app --reload                 # http://localhost:8000

# 2. Frontend Angular (en otra terminal)
cd frontend-angular
npm install
ng serve                                  # http://localhost:4200

# 3. Login con admin@tfm.com / admin123 → /admin
```

---

## Pipeline de ejecución (runbook)

El TFM se ejecuta en **fases secuenciales**. El admin debe completarlas en orden:

| Fase | Acción | Endpoint | Tiempo |
|---|---|---|---|
| **1a** | Filtrar universo S&P 500 + cripto | `POST /admin/fase1/screener` | ~3 min |
| **1b** | Descargar precios + features | `POST /admin/fase1/preparar-datos` | ~2 min |
| **2** | Validar integridad de datos | `GET  /admin/fase2/validar-datos` | <1 s |
| **3a** | Entrenar PPO (modelo principal) | `POST /admin/fase3/entrenar-academico?risk_profile=balanced` | 30 min – 3 h |
| **3b** | Walk-forward validation | `POST /admin/fase3/walk-forward` | 2-4 h |
| **3c** | Expanding window validation | `POST /admin/fase3/expanding-window` | 4-6 h |
| **3d** | Sensitivity analysis (4 configs) | `POST /admin/fase3/sensitivity-analysis` | 6-8 h |
| **4** | Ajustar agente especulativo | `POST /admin/fase4/ajustar-especulativo` | <30 s |
| **5** | Consultar resultados conclusivos | `GET  /resultados/tabla-final` | <5 s |

Las fases 3a-3d pueden ejecutarse en paralelo (cada una es un background task con su propio lock).

---

## Backend (FastAPI)

### Endpoints públicos (sin autenticación)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/auth/login` | Devuelve JWT (form data) |
| `POST` | `/auth/register` | Registrar inversor |
| `GET`  | `/estado` | Estado del sistema (qué fases están listas) |
| `GET`  | `/universo` | Catálogo de activos con metadatos |
| `GET`  | `/screener/last` | Último screener con métricas por activo |
| `GET`  | `/risk-profiles` | Catálogo de perfiles (balanced/conservative/…) |
| `GET`  | `/walk-forward/results` | Métricas por ventana del último walk-forward |
| `GET`  | `/expanding-window/results` | Métricas por ventana del último expanding |
| `GET`  | `/sensitivity/results` | Tabla comparativa de las 4 configs evaluadas |
| `GET`  | `/resultados/tabla-final` | **Informe ejecutivo del TFM** (PPO vs todas) |

### Endpoints de administrador (JWT con `role=admin`)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/admin/fase1/screener` | Filtrar S&P 500 → top N candidatos |
| `POST` | `/admin/fase1/preparar-datos` | Descarga + features + normalización Z-score |
| `GET`  | `/admin/fase2/validar-datos` | Verificar integridad de CSVs |
| `POST` | `/admin/fase3/entrenar-academico` | Entrenar PPO con perfil seleccionado |
| `POST` | `/admin/fase3/walk-forward` | Walk-forward validation (rolling) |
| `POST` | `/admin/fase3/expanding-window` | Expanding window validation |
| `POST` | `/admin/fase3/sensitivity-analysis` | Sensibilidad sobre phi/gamma (4 configs) |
| `POST` | `/admin/fase4/ajustar-especulativo` | Ajustar agente GMM + K-Means |
| `GET`  | `/auth/users` | Listar usuarios |
| `DELETE` | `/auth/users/{email}` | Eliminar usuario |

### Endpoints de inversor (JWT con `role=investor` o `admin`)

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/auth/me` | Datos del usuario autenticado |
| `GET`  | `/investor/strategies` | Estrategias disponibles |
| `POST` | `/investor/simulate` | Backtest personalizado (capital, comisión) |

> Documentación interactiva (Swagger UI): `http://localhost:8000/docs`

### Base de datos

SQLite + SQLAlchemy ORM (migrable a PostgreSQL cambiando `DATABASE_URL`).
Cinco tablas:

- **users** — credenciales bcrypt + rol (admin/investor).
- **screener_results** — candidatos del screener con métricas (Sharpe, sector, volumen).
- **universes** — snapshot del universo activo (tickers, fechas, n_features).
- **trained_models** — modelos entrenados con `train_metrics` (incluye perfil de riesgo).
- **simulations** — historial de backtests por inversor.

Migración ligera vía `ALTER TABLE` en `init_db()` (sin Alembic).

### Credenciales por defecto

Al arrancar por primera vez se crea automáticamente:

| Campo | Valor |
|---|---|
| Email | `admin@tfm.com` |
| Password | `admin123` |
| Rol | `admin` |

> **Cambiar en producción** — definidas en `main.py → startup()`.

---

## Frontend Angular

UI orientada a usuario final con autenticación JWT, tema dark, gráficas Plotly.

| Ruta | Quién | Función |
|---|---|---|
| `/login` | público | Autenticación |
| `/dashboard` | autenticado | Resumen ejecutivo + accesos directos |
| `/admin` | admin | Pipeline completo: screener, datos, entrenar, walk-forward, expanding, sensitivity, especulativo |
| `/universo` | autenticado | Catálogo de activos del universo |
| `/estado` | público | Estado del sistema (fases completadas) |
| `/investor/simulator` | autenticado | Simulación personalizada (capital, comisión) |
| `/investor/results` | autenticado | 6 gráficas Plotly del backtest |
| `/resultados/tabla-final` | autenticado | **Informe ejecutivo TFM** (contexto + métricas + PPO vs baselines) |
| `/validacion` | autenticado | Walk-forward + expanding window (tabs, gráficas, tabla) |

Páginas con polling automático (cada 15 s) en operaciones largas para detectar fin sin recarga manual.

---

## Dashboard Streamlit

Modo de uso académico/exploración (no requiere frontend Angular):

```bash
cd backend
.venv\Scripts\activate
streamlit run src/reports/app_dashboard.py
```

Ocho secciones interactivas:

1. **Métricas comparativas** — tabla Sharpe / Sortino / MDD / CAGR / Vol / Retorno.
2. **Equity curves** — evolución de cartera por estrategia.
3. **Drawdown over time** — caídas desde máximos.
4. **Asset allocation** — pie chart de pesos finales del PPO.
5. **Diagnóstico de entrenamiento** — KL, clip fraction, value loss, explained variance.
6. **Distribución de retornos diarios**.
7. **Volatilidad rolling**.
8. **Análisis por régimen** (delegado a `regime_analysis.py`).

---

## Estrategias comparadas

| Estrategia | Tipo | Descripción | Aporte académico |
|---|---|---|---|
| **IA PPO (DRL)** | Deep RL | Política aprendida por reward composite | Propuesta principal del TFM |
| **Especulativo (GMM+K-Means)** | No supervisado | Detector de régimen + clustering rolling | Agente contraste sin reward |
| **Equal Weight Mensual** | Baseline | 1/N en cada activo, rebalanceo mensual | Difícil de superar (DeMiguel et al., 2009) |
| **Buy & Hold** | Baseline | Comprar y no rebalancear | Pasividad pura — comparativa de costes |
| **Cartera 60/40** | Baseline | 60% IVV + 40% BND | Benchmark institucional clásico |
| **Markowitz MV** | Baseline | Optimización media-varianza (ventana 12m) | Estado del arte académico (Markowitz, 1952) |

---

## Función de recompensa y perfiles de riesgo

El reward compuesto del PPO en cada step:

```
R_t = Sharpe_rolling_20d(t) − φ · MDD(t) − γ · Turnover(t)
```

- `Sharpe_rolling_20d`: rentabilidad ajustada por riesgo de los últimos 20 días.
- `MDD(t)`: caída actual desde el máximo histórico (penaliza riesgo de cola).
- `Turnover(t)`: rotación de cartera (desincentiva operativa excesiva).

Cuatro perfiles configurables (`src/training_drl/risk_profiles.py`):

| Perfil | φ | γ | Filosofía |
|---|---|---|---|
| `balanced` | 0.02 | 0.01 | **Por defecto en el TFM**. Equilibrio retorno/riesgo |
| `conservative` | 0.05 | 0.01 | Penaliza drawdowns 2.5× más → preserva capital |
| `low_turnover` | 0.02 | 0.02 | Penaliza turnover → mejor Sharpe en sensitivity |
| `aggressive` | 0.01 | 0.005 | Mínimas penalizaciones → máxima libertad |

El perfil usado se persiste en BD (`TrainedModel.train_metrics.risk_profile`) y se muestra en la tabla final y el dashboard.

---

## Validación temporal

Tres metodologías independientes y complementarias:

### Walk-forward (rolling)

Ventana de tamaño fijo deslizante. Pregunta: *"¿el PPO funciona en cualquier época, o solo casualmente acertó en el split 80/20?"*. Estándar académico (López de Prado, 2018, cap. 7).

- `train_days = 504` (~2 años)
- `test_days = 252` (~1 año)
- ~4-6 ventanas con 5 años de datos

### Expanding window

Train empieza siempre en día 0 y crece. Pregunta: *"¿el agente mejora con más historia o se satura?"*. Simula reentrenamiento periódico en producción.

- `min_train_days = 504` (~2 años)
- `test_days = 63` (~3 meses)
- ~12-14 ventanas con 5 años de datos

### Sensitivity analysis

Cuatro configuraciones (phi, gamma) entrenadas en igualdad de condiciones (ceteris paribus). Pregunta: *"¿por qué precisamente esa calibración? ¿Es robusta a variaciones?"*.

Resultado generado: `src/reports/sensitivity_analysis.csv` y `.png` con tabla comparativa.

---

## Análisis por régimen de volatilidad

`regime_analysis.py` segmenta el periodo de test en tres regímenes (calma /
transición / crisis) según percentiles de volatilidad rolling de IVV
calibrados sobre train. Para cada estrategia calcula métricas separadas
por régimen.

Responde a: *"¿el PPO bate a las baselines en TODOS los regímenes o solo
en los favorables?"*.

> Nota: módulo distinto de `regime_hmm.py`. Este último usa GMM con
> suavizado para que el agente especulativo TOME decisiones; el primero
> ETIQUETA días post-hoc para evaluación. Son piezas complementarias,
> no duplicadas.

---

## Artefactos generados

Todos en `backend/src/reports/`:

| Fichero | Generado por | Uso en la memoria |
|---|---|---|
| `training_diagnostics.png` | `train_academic` | Verificar entropía, KL, clip fraction estables |
| `overfitting_analysis.png` | `OverfitDetectorCallback` | Curvas train vs eval (no hay sobreajuste) |
| `walk_forward_results.csv` + `.png` | `walk_forward_validation` | Robustez temporal por ventanas fijas |
| `expanding_window_results.csv` + `.png` | `expanding_window_validation` | Robustez con train creciente |
| `sensitivity_analysis.csv` + `.png` | `run_sensitivity_analysis` | Robustez frente a phi/gamma |
| `regime_analysis.png` + `regime_metrics.csv` | `analyze_regimes` | Métricas separadas por régimen |
| `training_progress.png` | `plot_training_progress` | Curva de aprendizaje del PPO |

---

## Tecnologías

**Backend**
- Python 3.11
- FastAPI + Uvicorn (API REST)
- SQLAlchemy + SQLite (persistencia)
- Stable-Baselines3 (PPO)
- Gymnasium (entorno de RL)
- scikit-learn (GMM, K-Means, PCA, StandardScaler)
- pandas, numpy
- yfinance + curl_cffi (datos de mercado)
- matplotlib (gráficas para la memoria)
- bcrypt + python-jose (autenticación JWT)

**Frontend**
- Angular 21 (standalone components)
- TypeScript 5
- Plotly.js (gráficas interactivas)

**Visualización**
- Streamlit (dashboard académico)
- TensorBoard (curvas de entrenamiento)

---

## Referencias académicas

- **Markowitz, H. (1952)**. *Portfolio Selection*. Journal of Finance.
  Base teórica del baseline Markowitz MV.
- **Sharpe, W. (1964)**. *Capital Asset Pricing Model*. Marco teórico
  para la métrica Sharpe Ratio y el uso de IVV como proxy del mercado.
- **Fama, E. (1970)**. *Efficient Capital Markets*. Justifica la
  selección de activos líquidos en el screener.
- **Hamilton, J. (1989)**. *Regime-switching models*. Econometrica.
  Trabajo seminal sobre detección de regímenes financieros.
- **Jegadeesh, N., & Titman, S. (1993)**. *Returns to Buying Winners
  and Selling Losers*. Journal of Finance. Base del filtro Sharpe
  rolling positivo en el screener.
- **Rabiner, L. (1989)**. *Tutorial on Hidden Markov Models*. Modelo
  cuya simplificación usamos en `regime_hmm.py` (GMM + suavizado).
- **Schulman, J. et al. (2017)**. *Proximal Policy Optimization
  Algorithms*. arXiv:1707.06347. Algoritmo PPO usado.
- **DeMiguel, V. et al. (2009)**. *Optimal versus Naive
  Diversification*. Review of Financial Studies. Justifica la fortaleza
  de Equal Weight como baseline.
- **Almgren, R., & Chriss, N. (2001)**. *Optimal Execution*. Justifica
  el filtro de liquidez del screener.
- **Ang, A., & Bekaert, G. (2002)**. *Regime Switches in Interest
  Rates*. Journal of Business & Economic Statistics. Confirma la
  detectabilidad empírica de regímenes.
- **López de Prado, M. (2018)**. *Advances in Financial Machine
  Learning*. Wiley. Cap. 7 — fundamenta la metodología walk-forward.

---

## Estado del proyecto

| Componente | Estado |
|---|---|
| Pipeline de datos | Operativo |
| Entrenamiento PPO con 4 perfiles | Operativo |
| Validación walk-forward + expanding | Operativo |
| Sensitivity analysis (4 configs) | Operativo |
| Agente especulativo GMM + K-Means | Operativo |
| API FastAPI con JWT y BD | Operativa |
| Frontend Angular (8 páginas) | Operativo |
| Dashboard Streamlit (8 secciones) | Operativo |
| Análisis por régimen | Operativo |
| App móvil + frontend Vue | **Trabajo futuro** |
| Datos en tiempo real (`LiveSource`) | **Trabajo futuro** |
| Reentrenamiento por ventana en walk-forward | **Trabajo futuro** (TODO académico documentado) |

---

## Licencia y autoría

Trabajo Final de Máster — Universitat Oberta de Catalunya (UOC).
Autor: Marcos Marqués Primo.

Para feedback o consultas: ver `mailto` en `package.json` del frontend o
el campo `created_by` en BD para usuarios del sistema.
