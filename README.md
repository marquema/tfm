# TFM — Optimización Dinámica de Carteras Híbridas de ETFs mediante Deep Reinforcement Learning

Implementación completa del sistema de gestión de carteras basado en PPO (Proximal Policy Optimization) comparado contra cuatro estrategias clásicas: Equal Weight, Buy & Hold, Cartera 60/40 y Markowitz Media-Varianza.

---

## Requisitos

- Python 3.10+
- Windows / Linux / macOS

---

## 1. Instalación

```bash
# Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# Instalar dependencias
pip install -r requirements.txt
```

---

## 2. Arrancar la API

```bash
uvicorn main:app --reload
```

La API queda disponible en `http://localhost:8000`.  
Dejar esta terminal abierta durante todo el flujo.

---

## 3. Descarga y preparación de datos

Descarga precios OHLCV desde Yahoo Finance, calcula features técnicas y normaliza con Z-Score sin lookahead bias.

```bash
curl -X POST "http://localhost:8000/fase1/preparar-datos" \
     -H "Content-Type: application/json" \
     -d '{
           "tickers": ["IVV","BND","IBIT","MO","JNJ","SCU","AWK","CB"],
           "start_date": "2014-01-01",
           "end_date": "2026-03-01"
         }'
```

**Archivos generados:**
- `data/normalized_features.csv` — features normalizadas para el agente
- `data/original_prices.csv` — precios de cierre sin normalizar

---

## 4. Verificar datos

```bash
curl http://localhost:8000/fase2/validar-datos
```

La respuesta debe mostrar `filas > 0`, `columnas > 0` y `nan = 0`.

---

## 5. Entrenamiento académico

Entrena el agente PPO con validación out-of-sample continua, early stopping y detección de sobreajuste.

```bash
curl -X POST "http://localhost:8000/fase3/entrenar-academico?steps=1000000"
```

El proceso puede tardar entre 30 minutos y varias horas según la CPU.  
Monitorizar el progreso en la terminal de uvicorn.

**Archivos generados:**
- `models/best_model_academic/best_model.zip` — modelo en el punto de máxima generalización ✅
- `models/ppo_academic_final.zip` — modelo al final del entrenamiento
- `src/reports/training_diagnostics.png` — métricas internas PPO (entropía, value loss, explained variance)
- `src/reports/overfitting_analysis.png` — curvas train vs eval para detección de sobreajuste

> **Nota:** usar siempre `best_model.zip`, no `ppo_academic_final.zip`. El best model se guarda en el momento de máximo reward de evaluación, antes de cualquier degradación por sobreentrenamiento.

---

## 6. Walk-Forward Validation

Validación temporal equivalente al k-fold cross-validation. Entrena y evalúa el agente en múltiples ventanas temporales independientes para demostrar robustez en distintos regímenes de mercado.

```bash
curl -X POST "http://localhost:8000/fase3/walk-forward?steps_por_ventana=100000"
```

**Archivos generados:**
- `src/reports/walk_forward_analysis.png` — Sharpe, retorno y MDD por ventana temporal
- `src/reports/walk_forward_results.csv` — tabla de resultados por ventana

> Se puede lanzar en paralelo con el entrenamiento si hay recursos suficientes, o después de que termine.

---

## 7. Verificar estado del sistema

```bash
curl http://localhost:8000/estado
```

Todos los campos deben estar en `true` antes de ejecutar el backtest.

---

## 8. Backtest por consola — results_viewer

Ejecuta el backtest completo e imprime la tabla comparativa en consola. No requiere la API.

```bash
python results_viewer.py
```

**Archivos generados en `src/reports/`:**
| Archivo | Contenido |
|---|---|
| `backtest_principal.png` | Curvas de equity de todas las estrategias |
| `metrics_table.csv` | Tabla de métricas exportable para la memoria del TFM |
| `training_progress.png` | Curva de reward durante el entrenamiento |

> `results_viewer.py` usa `models/best_model_academic/best_model.zip` por defecto.

---

## 9. Dashboard interactivo — Streamlit

```bash
streamlit run app_dashboard.py
```

Se abre en `http://localhost:8501`.

**Pasos dentro del dashboard:**
1. Sidebar → seleccionar `models/best_model_academic/best_model.zip`
2. Ajustar comisión por operación (por defecto 0.1%), capital inicial y split train/test
3. Pulsar **▶ Ejecutar Backtest Completo**
4. Explorar las secciones:
   - **Métricas de rendimiento** — tabla comparativa con glosario explicativo
   - **Equity curves** — evolución del capital con fechas reales
   - **Drawdown** — peor caída en cada momento del período de test
   - **Asset allocation** — cartera final del agente + evolución de pesos
   - **Diagnóstico del entrenamiento** — validación académica del proceso de aprendizaje

---

## Orden de ejecución completo

```
uvicorn main:app --reload
│
├── POST /fase1/preparar-datos          # descarga y features
├── GET  /fase2/validar-datos           # verificar integridad
├── POST /fase3/entrenar-academico      # entrenar PPO (1M pasos)
└── POST /fase3/walk-forward            # validación temporal

python results_viewer.py               # backtest + tabla por consola
streamlit run app_dashboard.py         # dashboard interactivo
```

---

## Estructura del proyecto

```
├── main.py                          # API FastAPI
├── app_dashboard.py                 # Dashboard Streamlit
├── results_viewer.py                # Backtest por consola
├── data/
│   ├── normalized_features.csv      # features normalizadas (generado)
│   └── original_prices.csv          # precios de cierre (generado)
├── models/
│   └── best_model_academic/
│       └── best_model.zip           # modelo PPO final (generado)
├── src/
│   ├── environment_trading.py       # entorno Gymnasium para el agente
│   ├── training_analysis.py         # callbacks académicos y walk-forward
│   ├── train.py                     # configuración PPO
│   ├── pipeline_getdata/
│   │   ├── data_downloader.py       # pipeline de descarga y features
│   │   └── data_source.py           # abstracción de fuentes de datos
│   ├── feature_ingeneering/
│   │   └── data_features.py         # cálculo de indicadores técnicos
│   ├── benchmarking/
│   │   └── baselines.py             # estrategias de referencia
│   └── reports/                     # gráficas y CSVs generados
└── requirements.txt
```

---

## Universo de activos

| Ticker | Activo | Categoría |
|--------|--------|-----------|
| IVV | iShares Core S&P 500 ETF | Renta variable EE.UU. |
| BND | Vanguard Total Bond Market ETF | Renta fija |
| IBIT | iShares Bitcoin Trust ETF | Activo digital |
| MO | Altria Group | Dividendo alto |
| JNJ | Johnson & Johnson | Defensivo salud |
| SCU | Sculptor Capital Management | Alternativo |
| AWK | American Water Works | Utilities |
| CB | Chubb Limited | Seguros |

---

## Función de recompensa

$$R_t = \text{Sharpe}_{20d}(t) - \phi \cdot \text{MDD}(t) - \gamma \cdot \text{Turnover}(t)$$

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| φ (phi) | 0.02 | Penalización por drawdown |
| γ (gamma) | 0.01 | Penalización por rotación de cartera |
| Ventana Sharpe | 20 días | Horizonte para Sharpe rolling |