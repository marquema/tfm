# TFM — Optimización Dinámica de Carteras Híbridas con ETFs Cripto y Activos Tradicionales mediante Deep Reinforcement Learning

**Autor**: Marcos Marqués Primo
**Tutor (PDC)**: Ruben Perez Ibañez
**Profesor Responsable de Asignatura (PRA)**: David Masip
**Titulación**: Máster Universitario en Ciencia de Datos — UOC
**Fecha entrega**: Junio 2026

---

## Resumen

Este TFM diseña, implementa y evalúa un sistema de gestión dinámica de carteras híbridas mediante *Deep Reinforcement Learning*. El sistema combina activos tradicionales (renta variable y renta fija) con ETFs regulados sobre criptoactivos (IBIT y ETHA) para evaluar cómo se integra esta nueva clase de activos digitales en la inversión institucional.

El algoritmo principal de diseño es **Proximal Policy Optimization (PPO)**, justificado por argumentos teóricos en el capítulo 2 de la memoria. Como análisis posterior de robustez algorítmica se contrasta empíricamente contra **A2C** y **SAC** con N=5 semillas por configuración, siguiendo el protocolo de Henderson et al. (2018).

La función de recompensa es multiobjetivo:

```
r_t = clip( Sharpe_rolling_20d  −  φ · MDD_t  −  γ · turnover_t , −1 , 1 )
```

La validación es estricta *out-of-sample* mediante esquemas *walk-forward* (2 años train / 1 año test) y *expanding window* (2 años train mínimo, 3 meses test acumulativo). El universo final consta de 17 activos seleccionados mediante screener cuantitativo con **cutoff temporal estricto 2024-03-08** (anterior al periodo de test), evitando *data leakage*.

---

## Resultados principales (universo honest n=17, test 2024-03-09 a 2026-04-30)

| Estrategia | Sharpe | Retorno % | MDD % |
|---|---|---|---|
| **Random Uniform** (baseline aleatoria) | **1.157** | +120.05 | −30.13 |
| Buy & Hold | 0.994 | +80.51 | −25.27 |
| Momentum Top-3 | 0.942 | +105.89 | −33.30 |
| Equal-Weight | 0.926 | +73.74 | −26.91 |
| Speculative GMM+K-Means | 0.868 | +81.97 | −31.74 |
| Cartera 60/40 | 0.785 | +28.82 | −11.25 |
| Markowitz MV | 0.723 | +41.60 | −20.62 |
| **PPO Optuna LT** (media N=5) | 0.474 ± 0.103 | — | — |
| **A2C LT** (media N=5) | 0.569 ± 0.287 | — | — |
| **SAC LT** (media N=5) | **0.883 ± 0.296** | — | — |
| **SAC LT mejor semilla (seed 4)** | **1.280** | **+119.93** | **−22.59** |

**Hallazgos clave**:

1. La mejor semilla individual de SAC (Sharpe 1.280) **supera a Random Uniform** y demuestra que existe señal aprendible en el universo honest.
2. Ningún algoritmo DRL **supera en media** a Random Uniform sobre 5 semillas. La señal existe pero su captura sistemática depende fuertemente de la inicialización.
3. La optimización automática de hiperparámetros con Optuna (51 trials TPE + MedianPruner) **NO mejora** sobre la calibración manual. El mejor ensayo (Sharpe validación 1.32) es el #0, generado por muestreo aleatorio antes de que TPE consolidara su modelo previo. Resultado negativo documentado.
4. SAC obtiene mejor Sharpe medio que PPO y A2C en este dominio; su naturaleza *off-policy* con replay buffer captura más información por paso.
5. La varianza inter-semilla supera en magnitud a las diferencias entre algoritmos, impidiendo afirmar dominación estocástica con N=5 semillas (Henderson 2018).

---

## Estructura del repositorio

```
.
├── README.md                          (este fichero)
│
├── backend/                           Núcleo Python + FastAPI
│   ├── main.py                        Aplicación FastAPI (auth, fase1-4, simulación)
│   ├── requirements.txt               Dependencias Python congeladas
│   │
│   ├── src/                           Módulos del trabajo
│   │   ├── auth/                      Autenticación JWT + roles + persistencia usuarios
│   │   ├── pipeline_getdata/          Ingesta Yahoo Finance (curl_cffi) + screener honest
│   │   ├── feature_engineering/       Z-score por ventana, RSI/MACD/momentum/correlaciones
│   │   ├── training_drl/              PortfolioEnv (Gymnasium) + train_academic + callbacks
│   │   ├── hpo/                       Búsqueda Optuna (espacio 10-dim, TPE+MedianPruner)
│   │   ├── benchmarking/              6 baselines + compute_metrics canónico
│   │   ├── unsupervised/              GMM régimen + K-Means clustering + speculative agent
│   │   ├── investor/                  Endpoint /investor/simular para rol inversor
│   │   └── reports/                   Generadores de figuras + JSONs canónicos de métricas
│   │
│   ├── scripts/                       Scripts standalone reproducibles (CRÍTICO)
│   │   ├── run_screener_and_prepare.py        Genera data/ desde Yahoo
│   │   ├── run_optuna.py                      Búsqueda Optuna 51 trials
│   │   ├── retrain_optuna_best.py             PPO Optuna × 5 seeds × 1.5M
│   │   ├── retrain_a2c_multiseed.py           A2C × 5 seeds × 2 perfiles × 1.5M
│   │   ├── retrain_sac_multiseed.py           SAC × 5 seeds × 2 perfiles × 500k
│   │   ├── retrain_ppo_optuna_dual.py         Variante dual reward
│   │   ├── retrain_speculative_honest.py      GMM+K-Means honest retrain
│   │   ├── eval_baselines_honest.py           6 baselines test
│   │   ├── run_walkforward_expanding_honest.py Validación temporal
│   │   ├── recompute_metrics.py               Re-eval modelos guardados
│   │   ├── show_optuna_best.py                Inspección study SQLite
│   │   └── diagnose_ppo*.py                   Diagnóstico pesos PPO
│   │
│   ├── data/                          Datos generados por screener (cutoff 2024-03-08)
│   │   ├── normalized_features.csv    Features Z-score entrada agente (n=17)
│   │   ├── original_prices.csv        Precios reales (cálculo recompensa)
│   │   ├── original_features.csv      Features sin normalizar
│   │   ├── dividend_features.csv      Dividendos
│   │   ├── sp500_universe.csv         Universo seleccionado
│   │   └── universe_config.json       Configuración screener
│   │
│   ├── hpo/
│   │   └── optuna_study_ppo.db        SQLite study Optuna (51 trials, evidencia)
│   │
│   └── models/                        Modelos canónicos serializados (.zip SB3)
│       ├── best_model_academic_OPTUNA_seed{0..4}     PPO Optuna × 5 semillas
│       ├── best_model_academic_OPTUNA_dual           PPO Optuna + dual reward
│       ├── best_model_academic_a2c_low_turnover_seed{0..4}
│       ├── best_model_academic_a2c_aggressive_seed{0..4}
│       ├── best_model_academic_sac_low_turnover_seed{0..4}
│       ├── best_model_academic_sac_aggressive_seed{0..4}
│       └── speculative_gmm.pkl        Modelo GMM+K-Means honest
│
└── frontend-angular/                  Capa de presentación (anexo D memoria)
    ├── package.json                   Dependencias Node
    ├── angular.json                   Configuración Angular CLI
    ├── tsconfig*.json                 Configuración TypeScript
    ├── public/                        Assets estáticos
    └── src/
        ├── index.html
        ├── main.ts
        └── app/
            ├── components/            UI compartida
            ├── guards/                Auth guards
            ├── interceptors/          JWT interceptor HTTP
            ├── pages/                 9 pantallas (login, dashboard, universo, ...)
            └── services/              Clientes API
```

---

## Setup local — Backend

**Requisitos**: Python 3.11, Windows o Linux.

```powershell
cd backend

# 1. Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate       # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar servidor FastAPI (puerto 8000)
.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000

# 4. Probar dashboard Streamlit (opcional)
.venv\Scripts\python.exe -m streamlit run src/reports/app_dashboard.py
```

La API queda accesible en `http://localhost:8000/docs` (Swagger UI auto-generado).

---

## Setup local — Frontend

**Requisitos**: Node 18+, Angular CLI 17.

```bash
cd frontend-angular

# 1. Instalar dependencias
npm install

# 2. Lanzar en modo desarrollo (puerto 4200)
ng serve

# 3. Abrir navegador: http://localhost:4200
```

El frontend asume que el backend está corriendo en `localhost:8000`. Configurable en `src/environments/`.

---

## Reproducir resultados del TFM

Los modelos canónicos están en `backend/models/` listos para evaluar. Para regenerarlos desde cero (~28 horas de cómputo CPU sobre Intel i7 portátil sin GPU):

```powershell
cd backend
.venv\Scripts\activate

# Paso 1: regenerar features desde Yahoo Finance (1-2 min)
.venv\Scripts\python.exe scripts/run_screener_and_prepare.py

# Paso 2: PPO Optuna multi-seed (~4 horas)
.venv\Scripts\python.exe scripts/retrain_optuna_best.py

# Paso 3: A2C multi-seed × 2 perfiles (~10 horas)
.venv\Scripts\python.exe scripts/retrain_a2c_multiseed.py

# Paso 4: SAC multi-seed × 2 perfiles (~12 horas)
.venv\Scripts\python.exe scripts/retrain_sac_multiseed.py

# Paso 5: PPO Optuna + dual reward (~30 min)
.venv\Scripts\python.exe scripts/retrain_ppo_optuna_dual.py

# Paso 6: GMM+K-Means speculative (~1 min)
.venv\Scripts\python.exe scripts/retrain_speculative_honest.py

# Paso 7: evaluar 6 baselines honest (~2 min)
.venv\Scripts\python.exe scripts/eval_baselines_honest.py

# Paso 8: validación temporal WF + EW (~1 hora)
.venv\Scripts\python.exe scripts/run_walkforward_expanding_honest.py
```

Para inspeccionar el study Optuna (51 trials TPE + MedianPruner):

```powershell
.venv\Scripts\python.exe scripts/show_optuna_best.py
```

Resultados se persisten en `backend/src/reports/`:
- `baselines_honest_results.json` — 6 baselines
- `optuna_retrain_results.json` — PPO Optuna 5 semillas
- `a2c_multiseed_results.json` — A2C 10 ejecuciones
- `sac_multiseed_results.json` — SAC 10 ejecuciones
- `speculative_honest_result.json` — GMM+K-Means
- `walk_forward_results.csv` / `expanding_window_results.csv` — validación temporal

---

## Configuración experimental común

| Componente | Valor |
|---|---|
| Universo | 17 tickers (4 obligatorios IVV/BND/IBIT/ETHA + 13 screener S&P 500) |
| Periodo dataset | 2017-11-09 a 2026-04-30 (≈ 8.5 años, 2891 pasos diarios) |
| Cutoff screener honest | 2024-03-08 (estrictamente anterior al test) |
| Partición train/test | 80% / 20% (test: 579 días, 2024-03-09 → 2026-04-30) |
| Capital inicial | 10.000 $ |
| Comisión | 0.10% sobre turnover |
| Política | `MlpPolicy` con `net_arch=[256, 256]` |
| Perfil principal | `low_turnover` (φ=0.02, γ=0.02) |
| Inferencia | Determinista (`deterministic=True`) |

**Hiperparámetros Optuna best** (trial #0):
`lr=8.47e-5`, `n_steps=512`, `batch_size=256`, `clip_range=0.20`, `ent_coef=0.013`, `gae_lambda=0.902`, `vf_coef=0.979`, `max_grad_norm=0.883`, `varphi=0.0094`, `gamma=0.0020`.

---

## Endpoints API principales

| Verbo | Path | Descripción |
|---|---|---|
| POST | `/auth/login` | OAuth2, devuelve JWT |
| POST | `/admin/fase3/entrenar-academico` | Entrena modelo en background (lock auto) |
| POST | `/admin/fase3/walk-forward` | Validación temporal rolling |
| POST | `/admin/fase3/expanding-window` | Validación expanding window |
| POST | `/admin/fase3/sensitivity` | Análisis sensibilidad 4 perfiles |
| POST | `/admin/fase4/simular` | Backtest interactivo (rol inversor) |
| GET  | `/estado` | Estado del pipeline (polling) |

Documentación completa en `http://localhost:8000/docs` tras lanzar el servidor.

---

## Limitaciones reconocidas

1. **Proxy BTC-USD/ETH-USD pre-2024**: IBIT cotiza desde enero 2024 y ETHA desde julio 2024. Se sustituye por subyacente en periodo anterior con escalado multiplicativo. Tracking error despreciable frente a volatilidad diaria.
2. **Inestabilidad entre ventanas**: Sharpe robusto WF +0.349 ± 1.904, EW +0.654 ± 2.399. El agente no generaliza bien a transiciones de régimen no vistas durante entrenamiento.
3. **Restricción long-only**: pesos en $[0, 1]$ con suma = 1. Sin shorts ni apalancamiento, coherente con gestión institucional UCITS regulada.
4. **Régimen de test único**: marzo 2024 a abril 2026 cubre ciclo alcista tech + lateral BTC + bajista ETH. No incluye crisis amplia tipo 2008.
5. **Costes idealizados**: comisión lineal 0.1% sobre turnover, sin modelo de impacto de mercado.
6. **Survivorship bias adicional**: el screener selecciona entre componentes ACTUALES del S&P 500, no históricos point-in-time.
7. **Multi-seed N=5 insuficiente para tests de hipótesis formales**: requiere N≥10 según Henderson 2018.

Detalle completo en el capítulo 5 de la memoria.

---

## Líneas futuras identificadas

1. Régimen GMM como característica explícita del estado
2. Arquitecturas recurrentes (`MlpLstmPolicy` en sb3-contrib, Transformer)
3. *Curriculum learning* para estabilidad inter-régimen
4. Optuna sobre A2C y SAC (no realizado por presupuesto)
5. Extensión a posiciones cortas con costes realistas (Almgren-Chriss)
6. Validación sobre regímenes históricos de crisis (2008, COVID Q1 2020, cripto 2022)
7. Despliegue productivo con re-entrenamiento periódico (MLOps completo)
8. Costes de transacción dependientes del tamaño de orden
9. Vectorización del entorno para A2C (`SubprocVecEnv` con callbacks adaptados)

---

## Stack tecnológico

| Categoría | Tecnología |
|---|---|
| Lenguaje backend | Python 3.11 |
| Framework DRL | Stable-Baselines3 / Gymnasium |
| Análisis numérico | NumPy, Pandas, PyTorch |
| Optimización HPO | Optuna 3.x (TPE + MedianPruner) |
| API REST | FastAPI + Uvicorn |
| Persistencia | SQLite + SQLAlchemy ORM |
| Autenticación | JWT (HS256) |
| Ingesta datos | Yahoo Finance vía `curl_cffi` |
| Frontend | Angular 17 + TypeScript + Plotly |
| Dashboards exploratorios | Streamlit |
| Documentación | LaTeX (TFUOC.cls) |

---

## Contacto

Marcos Marqués Primo — Estudiante UOC Máster Ciencia de Datos
Para dudas sobre el código o reproducción: ver Anexo D de la memoria.
