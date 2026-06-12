# CHEATSHEET DEFENSA — VERSIÓN BLINDADA FINAL

**Marcos Marqués Primo · UOC Máster Ciencia de Datos · 15/06/2026**
*Optimización Dinámica de Carteras Híbridas con ETFs Cripto y Activos Tradicionales mediante Deep Reinforcement Learning*

---

## PLAN 8 DÍAS REALES (07/06/2026 → 15/06/2026)

Hoy = 07/06 (domingo). Defensa = 15/06 (lunes). 8 días reales prep + día defensa.

| Día | Fecha | Día sem | Tarea |
|---|---|---|---|
| **D-8** | 07/06 | Dom | Compilar PDF tex (resolver inputenc clash si reaparece) + revisión visual portada UOC (item 16). Confirmar logística defensa: sala física / link virtual, hora exacta, soporte técnico UOC. |
| **D-7** | 08/06 | Lun | Fix tex residuales post-compile (typos, refs rotas, floats descolocados). Re-compile clean. PDF DEFINITIVO. |
| **D-6** | 09/06 | Mar | Revisión slides pptx contra PDF final. Ajustar números si discrepan. Lectura activa cheatsheet — marcar números débiles. |
| **D-5** | 10/06 | Mié | **Ensayo 1** cronometrado completo (slides + voz, sin parar). Apuntar tiempo total + bloques largos. |
| **D-4** | 11/06 | Jue | Refinar slides débiles detectados ensayo 1. Memorizar 5 números clave (flashcards). |
| **D-3** | 12/06 | Vie | **Ensayo 2** + GRABAR audio. Escuchar grabación, detectar muletillas. Q&A: contestar voz alta preguntas-tipo + gotchas. |
| **D-2** | 13/06 | Sáb | **Ensayo 3 final** (idealmente delante alguien). Backup PDF: USB + email + nube. Push final git con tag entrega. Q&A: preguntas cobertura 360. |
| **D-1** | 14/06 | Dom | **Día ligero**. Relectura cheatsheet + flashcards. Ropa preparada. Dormir 8h. NO ensayar nada nuevo. |
| **D-0** | 15/06 | Lun | **DEFENSA**. Conectar/llegar 30 min antes. Agua + cheatsheet impreso a mano. |

---

## 5 NÚMEROS CRÍTICOS A MEMORIZAR

1. **SAC LT mejor seed = Sharpe 1.280** (retorno +119.93%, MDD −22.59%). Único DRL que supera Random Uniform.
2. **SAC LT media N=5 = 0.883 ± 0.296** — mejor algoritmo DRL en media.
3. **PPO Optuna LT media N=5 = 0.474 ± 0.103** — NO supera Random Uniform.
4. **Random Uniform = Sharpe 1.157, retorno +120.05%** — la baseline a batir.
5. **Universo HONEST n=17, cutoff screener 2024-03-08, test 579 días** (2024-03-09 → 2026-04-30).

## FLASHCARDS COMPLETAS (12 números)

| # | Pregunta | Respuesta |
|---|---|---|
| 1 | Sharpe del mejor algoritmo DRL en media | SAC LT = **0.883 ± 0.296** sobre 5 semillas |
| 1b | Mejor semilla individual | SAC LT seed 4 = **1.280**, retorno **+119.93%**, MDD **−22.59%** |
| 2 | Sharpe del PPO Optuna en media | **0.474 ± 0.103**, no supera Random Uniform |
| 2b | Semillas y pasos PPO | 5 semillas × 1.500.000 pasos cada una |
| 3 | Random Uniform | Sharpe **1.157**, retorno **+120.05%**, MDD −30.13% |
| 4 | Universo | **n=17** activos, cutoff **2024-03-08**, test **579 días** |
| 5 | Mejor Sharpe ensayo Optuna | Trial **#0**, Sharpe validación **1.32** (muestreo aleatorio) |
| 6 | Validación temporal WF | 9 ventanas (7 sig.), Sharpe robusto **+0.349 ± 1.904**, mediana 1.178 |
| 6b | Validación temporal EW | 37 ventanas (26 sig.), Sharpe robusto **+0.654 ± 2.399**, 62% positivas |
| 7 | Núcleo obligatorio universo | **IVV** (S&P 500), **BND** (bonos), **IBIT** (BTC ETF), **ETHA** (ETH ETF) |
| 8 | Perfil principal | `low_turnover`: **φ=0.02, γ=0.02** |
| 9 | Mejor ventana walk-forward | V7 (mayo 2023 → enero 2024), Sharpe **2.258**, retorno **+78%** |

---

## CAPÍTULO 1 — INTRODUCCIÓN (3 preguntas)

**P1.1 — ¿Por qué este problema y por qué ahora?**
> Enero de 2024 marca un cambio cualitativo: la SEC aprueba los primeros ETFs spot sobre Bitcoin en Estados Unidos. El iShares Bitcoin Trust, IBIT, de BlackRock, fue uno de ellos. Por primera vez los gestores patrimoniales pudieron incorporar Bitcoin a sus carteras a través de un instrumento regulado, custodiado por un prime broker convencional y negociado en bolsas organizadas. La pregunta deja de ser hipotética y se vuelve operativa: cómo gestionar dinámicamente una cartera híbrida que combine activos tradicionales con esta nueva clase de activos digitales. Los marcos clásicos —Markowitz y derivados— asumen estacionariedad, normalidad y ausencia de fricciones, supuestos que los criptoactivos rompen frontalmente.

**P1.2 — ¿Cuál es el objetivo principal?**
> Desarrollar un agente DRL basado en PPO que optimice la asignación de capital en una cartera híbrida de ETFs, mejorando el ratio de Sharpe/Sortino frente a estrategias de referencia, incorporando control explícito de drawdown y costes operativos en la función de recompensa.

**P1.3 — ¿Por qué PPO como algoritmo principal si SAC obtuvo mejor Sharpe medio?**
> PPO se eligió por argumentos teóricos antes de la comparativa empírica, expuestos en el capítulo 2: on-policy, mecanismo de clipping para estabilidad, eficiencia muestral aceptable, bien estandarizado en Stable-Baselines3. La comparativa multi-algoritmo de la iteración 3 funciona como verificación a posteriori de esa decisión, no como cambio de objetivo. El resultado empírico muestra que SAC captura mejor la señal en media (0.883 vs 0.474), lo que se documenta como hallazgo relevante. PPO sigue siendo el algoritmo principal del diseño del sistema porque permitió toda la fase de calibración manual + Optuna de forma estable.

---

## CAPÍTULO 2 — ESTADO DEL ARTE (3 preguntas)

**P2.1 — ¿Por qué PPO frente a DQN, A2C, DDPG, TRPO y SAC en la justificación teórica?**
> DQN requiere espacio de acción discreto, incompatible con el simplex continuo de pesos. DDPG es notoriamente frágil ante la inicialización: dos entrenamientos con datos idénticos pueden producir agentes muy diferentes. TRPO usa una restricción explícita de divergencia KL con segunda derivada, mucho más complejo de implementar sin mejora consistente sobre PPO. A2C carece del mecanismo de clipping de PPO. SAC es off-policy con replay buffer, prometedor pero con coste computacional mayor y sintonización más exigente. PPO combina simplicidad de implementación, estabilidad del entrenamiento y eficiencia muestral razonable.

**P2.2 — ¿Qué aporta FinRL y qué deja sin resolver?**
> FinRL estandariza ingesta de datos, preprocesamiento y backtesting sobre el estándar Gymnasium. Lo que no aborda: ETFs regulados sobre criptoactivos como IBIT/ETHA, funciones de recompensa multiobjetivo con penalización explícita por drawdown y turnover, validación temporal estricta out-of-sample. El TFM extiende esa infraestructura en estos frentes, documentado en el Anexo A.

**P2.3 — ¿Cuál es el hueco en la literatura que cubre el trabajo?**
> No existe en la literatura revisada un marco integrado que combine simultáneamente: (i) un agente DRL basado en PPO, (ii) recompensa multiobjetivo con MDD y turnover, (iii) un universo híbrido con ETFs regulados sobre Bitcoin y Ethereum, (iv) un protocolo de validación temporal estricto out-of-sample. Cubrir ese vacío es la motivación central del TFM.

---

## CAPÍTULO 3 — MATERIALES Y MÉTODOS (8 preguntas)

**P3.1 — ¿Por qué long-only?**
> Mantiene la comparabilidad con la gestión institucional regulada: fondos UCITS y mutual funds bajo el Investment Company Act de 1940 prohíben las posiciones cortas en la mayoría de vehículos. Evita modelar apalancamiento, costes de préstamo del activo, riesgo de liquidación y restricciones regulatorias específicas. Extender a posiciones cortas con $[-1, 1]$ queda como línea futura.

**P3.2 — ¿Cómo gestionas la corta historia de IBIT/ETHA?**
> IBIT cotiza desde enero 2024 y ETHA desde julio 2024. Para entrenar con suficiente histórico se sustituye cada ETF por su activo subyacente (BTC-USD, ETH-USD) antes de la fecha de lanzamiento, aplicando un escalado multiplicativo que empalma el último precio del subyacente con el primer precio real del ETF. El tracking error de un ETF spot es del orden del 0.25% anual frente a una volatilidad diaria de Bitcoin del 3-5%; la distorsión es despreciable. Se reconoce explícitamente como limitación 1 del capítulo 5.

**P3.3 — ¿Por qué 17 activos?**
> Compromiso entre tres factores: dimensionalidad del espacio de acción manejable para PPO/A2C/SAC, diversidad sectorial (máximo 3 activos por sector GICS) y dilución de peso por activo. Cuatro son obligatorios por razón estructural (IVV, BND, IBIT, ETHA — exposición a renta variable, renta fija y criptoactivos), trece se seleccionan vía screener cuantitativo con cutoff 2024-03-08.

**P3.3b — TUTOR: El título habla de "ETFs Cripto y Activos Tradicionales" pero tu universo incluye acciones individuales del S&P 500. ¿No es una contradicción?**
> Observación documentada explícitamente en la sección 3.3 del trabajo. El término "ETFs" en el título se interpreta en sentido amplio como acceso institucional a clases de activos —incluyendo la novedad principal del trabajo, los ETFs spot sobre criptoactivos IBIT y ETHA. La cartera es realmente híbrida en dos sentidos: combina instrumentos institucionales (los ETFs del núcleo obligatorio IVV, BND, IBIT, ETHA) con equity directa (los 13 componentes del S&P 500 seleccionados vía screener). Una variante natural del trabajo, declarada como línea futura, consistiría en sustituir las acciones individuales por ETFs sectoriales o temáticos (XLK tech, XLV salud, XLE energía) manteniendo el universo en formato 100% ETF. La razón de no haberlo hecho en este TFM es que las acciones individuales dan más granularidad al espacio de decisión del agente y permiten observar comportamientos sectoriales más finos.

**P3.4 — Función de recompensa: ¿por qué multiobjetivo y no solo Sharpe?**
> La recompensa entrega en una sola señal por paso tres dimensiones que importan al gestor: Sharpe rolling de 20 días (rentabilidad ajustada a riesgo), penalización por MDD (control de caídas) y penalización por turnover (coste operativo). Una recompensa basada solo en retorno induciría al agente a ignorar el riesgo; una basada solo en penalización de caídas le llevaría a no operar. El clip a $[-1, 1]$ estabiliza el entrenamiento, asumiendo el trade-off de limitar el refuerzo en días excepcionales.

**P3.5 — ¿Por qué inferencia determinista?**
> PPO aprende una política Gaussiana diagonal sobre el simplex de pesos. En entrenamiento la estocasticidad es imprescindible para explorar; en evaluación se devuelve la media de la distribución (`deterministic=True` en SB3). Tres razones: reproducibilidad exacta de la curva de equity, realismo operativo (un gestor patrimonial necesita la mejor estimación, no una muestra aleatoria distinta cada día) y separación limpia entre entrenamiento y evaluación.

**P3.6 — ¿Cómo evitas el data leakage en la normalización Z-score?**
> Una primera versión calculaba media y desviación típica sobre el conjunto de entrenamiento global, lo que introducía lookahead residual en las ventanas tempranas del walk-forward. La corrección académicamente sólida —función `_normalize_window` en `training_analysis.py`— recalcula media y desviación usando únicamente los datos estrictamente anteriores al inicio de cada ventana. Todos los resultados del capítulo 4 corresponden a esta versión corregida.

**P3.7 — ¿Por qué Sharpe rolling de 20 días y no otro horizonte?**
> 20 días hábiles equivale a un mes de cotización, ventana habitual en finanzas para detectar tendencias de medio plazo sin contaminar con ruido intradiario. Ventanas más cortas (5 días) son hipersensibles a outliers; ventanas largas (60 días, 252 días) suavizan en exceso y retrasan la respuesta a cambios de régimen. 20 días es el estándar en la literatura DRL financiero, compatible con el horizonte de decisión diario del agente.

**P3.8 — ¿Por qué dataset arranca exactamente el 9 de noviembre de 2017?**
> Primer día con datos disponibles para todos los activos del universo, incluyendo ETH-USD (Ethereum cotiza desde julio 2017 pero hay días sin volumen hasta noviembre). Anclar el inicio al primer día con cobertura completa evita rellenos artificiales por imputación, que distorsionarían las correlaciones rolling de las features. El dataset cubre aproximadamente 8.5 años (2891 pasos diarios), suficiente para múltiples regímenes.

**N.2 — NOTA 10: ¿Por qué solo IBIT (BTC) y ETHA (ETH)? ¿No deberías incluir Solana, Cardano, BNB?**
> Restricción regulatoria. En 2024-2026 los únicos ETFs spot regulados aprobados por SEC sobre criptoactivos son Bitcoin (enero 2024) y Ethereum (julio 2024). Solana, Cardano, BNB no tienen ETF spot regulado en el periodo del estudio. El trabajo se centra deliberadamente en la institucionalización vía ETFs regulados, no en el universo cripto completo. Cuando se aprueben más ETFs cripto (Solana spot está pendiente SEC review), el universo se ampliaría naturalmente. Es línea futura.

**Z.1 — NOTA 10: Tu universo viene del S&P 500 actual. ¿No tienes survivorship bias adicional al data leakage?**
> Observación correcta y reconocida implícitamente. El S&P 500 actual contiene los supervivientes —empresas que no quebraron, no fueron adquiridas ni eliminadas del índice. La consecuencia: el screener selecciona entre ganadores estructurales, lo que sesga el universo hacia retornos positivos esperados aunque cumpla el cutoff temporal 2024-03-08 para el data leakage. Mitigación parcial: el screener filtra por volatilidad + Sharpe rolling, no por retorno absoluto, lo que reduce el sesgo hacia "ganadores extremos" pero no lo elimina. Una corrección completa requeriría histórico point-in-time del S&P 500 (composición real en cada fecha histórica, incluyendo empresas eliminadas posteriormente), no disponible vía Yahoo Finance gratuito. Limitación honesta del trabajo, complementaria a la limitación 6 del capítulo 5 (data leakage del screener).

---

## CAPÍTULO 4 — RESULTADOS (8 preguntas)

**P4.1 — ¿Cuál es el resultado principal del trabajo?**
> SAC es el algoritmo con mayor Sharpe medio: 0.883 ± 0.296 sobre 5 semillas. La mejor semilla de SAC, con Sharpe 1.280, es la única configuración DRL del estudio que supera a Random Uniform (1.157). PPO Optuna obtiene 0.474 ± 0.103 en media, por debajo de Random Uniform por más de 0.68 puntos de Sharpe. Hay señal aprendible en el universo, pero su captura es débil y depende fuertemente de la semilla.

**P4.2 — ¿Por qué Random Uniform domina la tabla?**
> El periodo de test (marzo 2024 a abril 2026) cubre un régimen mixto: ciclo alcista del sector tecnológico y de semiconductores, fase lateral en Bitcoin, mercado bajista en Ethereum. En un mercado con tendencia alcista generalizada en varios activos, cualquier estrategia diversificada captura ese impulso agregado. Batir a Random Uniform exige discriminar activamente cuándo entrar y cuándo salir, que es la dificultad real a la que se enfrenta el agente DRL. El screener honest no anticipa qué activo subirá durante el test, solo filtra por liquidez y volatilidad previas, pero el universo resultante contiene varios ganadores del periodo.

**P4.3 — ¿Por qué la mejor semilla de SAC supera a Random pero la media no?**
> Henderson 2018 documenta que la varianza inter-semilla en DRL puede superar las diferencias entre algoritmos. El rango de SAC LT es de 0.748 puntos (de 0.532 a 1.280), más de 10 veces la desviación estándar de PPO. Por eso se reporta media ± std sobre N=5 semillas: un único entrenamiento puede ser engañoso. La señal existe (la mejor semilla lo demuestra) pero su captura sistemática no está resuelta con los recursos disponibles.

**P4.4 — ¿Por qué la iteración 1 está sobre universo viejo?**
> La iteración 1 es cronológicamente el primer bloque experimental del proyecto, ejecutada sobre el universo inicial con n=15 antes de detectar el sesgo de selección en el screener original. Tras la detección, se corrige el screener (cutoff 2024-03-08), se vuelve a entrenar todo sobre el universo n=17 (configuración base de la iteración 2 y comparativa multi-algoritmo de la iteración 3) y se mantiene la iteración 1 como evidencia del proceso experimental. Sus números absolutos (Sharpe 1.875, 1.624, 1.650) no son comparables con los del universo final; solo las conclusiones cualitativas (la recompensa dual añade ruido, el perfil agresivo concentra en cripto) se conservan como hipótesis estructurales.

**P4.4b — TUTOR: Si SAC funciona mejor que PPO, ¿no estás cambiando el objetivo del TFM a mitad de la memoria?**
> No. PPO es el algoritmo PRINCIPAL DE DISEÑO del sistema desde el primer capítulo. Toda la justificación teórica del capítulo 2 (sección 2.2.1, "PPO vs alternativas"), toda la calibración manual del capítulo 3 (sección 3.7), toda la búsqueda Optuna (sección 3.8) y toda la configuración base del capítulo 4 son sobre PPO. La iteración 3 funciona como verificación experimental a posteriori de la decisión teórica, no como cambio de rumbo. Su propósito explícito —declarado en la introducción de la sección 4.7— es cuantificar cuánto del resultado depende del algoritmo concreto y cuánto del diseño del entorno y de la recompensa. El hallazgo de que SAC obtiene mejor Sharpe medio es un resultado empírico relevante que se documenta con honestidad, no una rectificación del objetivo del trabajo. PPO sigue siendo el algoritmo central de diseño porque permitió toda la fase de calibración + Optuna de forma estable y reproducible; SAC se reporta como referencia comparativa.

**P4.5 — ¿Por qué SAC mejor que PPO?**
> SAC es off-policy: reutiliza experiencias del replay buffer en múltiples actualizaciones, lo que mejora la eficiencia muestral. La regularización por entropía fuerza exploración sostenida. En un dominio con baja relación señal-ruido y un universo con varios activos volátiles, esa capacidad de extraer más información por paso parece la clave de la diferencia. PPO on-policy con clipping conservador produce un aprendizaje estable pero menos agresivo capturando potencial alcista.

**P4.6 — ¿Por qué Optuna no mejora la calibración manual?**
> 51 trials sobre un espacio de búsqueda de 10 dimensiones es subdimensionado para que TPE construya un modelo previo robusto. El MedianPruner es severo con ensayos ruidosos: 39 de 51 se eliminaron antes de completar evaluación. El problema mismo tiene baja relación señal-ruido. El mejor ensayo (Sharpe validación 1.32) es el #0, generado por muestreo aleatorio antes de que TPE tuviera información. Es un resultado negativo documentado que define los límites prácticos de la búsqueda automática en este tipo de problema.

**P4.7 — Validación temporal: ¿qué muestra?**
> El agente genera valor en promedio (Sharpe robusto positivo en ambos esquemas, 62% de ventanas expanding con Sharpe positivo) pero es inestable entre ventanas. La ventana 7 del walk-forward (mayo 2023 a enero 2024, fase de recuperación tras la crisis cripto de 2022) alcanza Sharpe 2.258. La ventana 5 (diciembre 2021 a septiembre 2022, colapso de Terra-Luna) obtiene −1.273. La política no es robusta frente a transiciones de régimen, lo que es coherente con la naturaleza no estacionaria de los mercados.

**P4.8 — ¿Por qué seed 4 de SAC y no otra?**
> Las 5 semillas se asignan aleatoriamente (0, 1, 2, 3, 4) por convención SB3. La seed 4 obtiene Sharpe 1.280; la seed 1 también supera 1.0 (1.191); las otras tres quedan en 0.532-0.712. No hay propiedad especial de la seed 4 más allá de la favorable interacción entre inicialización aleatoria y la trayectoria estocástica del entrenamiento. Esto es precisamente lo que Henderson 2018 advierte: en DRL una semilla no es elegible a priori; reportar media ± std evita el sesgo de selección post-hoc.

**N.1 — NOTA 10: ¿Cómo garantizas que tu agente no sobreajustó al periodo de test?**
> Tres mecanismos. (i) El periodo de test es completamente disjunto del de entrenamiento (cutoff 2024-03-08), sin solapamiento temporal. (ii) El screener se ejecuta SOLO con datos pre-cutoff (corrección data leakage). (iii) La validación temporal con walk-forward y expanding window evalúa sobre múltiples ventanas distintas; si el agente solo aprendiera memorizando el periodo test, su Sharpe colapsaría en las ventanas WF/EW. El callback `OverfitDetectorCallback` monitoriza el gap entre train reward y eval reward durante entrenamiento, deteniendo si supera umbral. Limitación: no usamos early stopping basado en validación pura porque PPO/SAC no convergen monotónicamente —el "mejor checkpoint" es ambiguo en DRL.

---

## CAPÍTULO 5 — CONCLUSIONES (5 preguntas)

**P5.1 — ¿Cuáles son las contribuciones principales?**
> Cuatro: (i) un agente PPO sobre un MDP adaptado a cartera híbrida con ETFs regulados sobre criptoactivos, (ii) recompensa multiobjetivo configurable con perfiles de riesgo, (iii) protocolo de validación temporal estricto con walk-forward, expanding window y análisis de sensibilidad, (iv) implementación como arquitectura modular con persistencia, autenticación y trazabilidad de cada modelo entrenado. Adicionalmente, la comparativa empírica multi-algoritmo con N=5 semillas siguiendo Henderson 2018 y el hallazgo negativo documentado de Optuna.

**P5.2 — ¿Cuál es el significado del fix de data leakage?**
> El screener original ejecutaba el criterio Sharpe rolling sobre el rango completo 2014-2026, incluyendo el periodo de test. Los activos seleccionados estaban sesgados hacia ganadores conocidos del periodo de evaluación. La corrección restringe el screener a datos estrictamente anteriores al cutoff 2024-03-08. El Sharpe del PPO Optuna cae de 1.875 (universo viejo, una semilla) a 0.474 ± 0.103 (universo honest, media de 5 semillas). La caída no significa que el agente empeore, significa que las métricas anteriores estaban infladas por el leakage. Los valores del universo honest son los académicamente correctos.

**P5.3 — ¿Cumplió los objetivos del TFM?**
> Objetivo principal de diseñar y evaluar un agente DRL sobre cartera híbrida con criptoactivos: cumplido. Tres agentes funcionales (PPO/A2C/SAC), recompensa multiobjetivo, inferencia determinista. Benchmarking comparativo: cumplido, 6 baselines clásicas + 1 adicional GMM+K-Means. Optimización de recompensa: cumplido vía análisis de sensibilidad y ablación. Evaluación multi-semilla: cumplido con N=5 seeds. Validación temporal preliminar: cumplido con walk-forward y expanding window; la robustez frente a crisis históricas amplias queda como línea futura.

**P5.4 — Tres líneas futuras de mayor impacto.**
> Primera: incorporar el régimen detectado por el GMM como característica explícita del estado (cambio acotado en pipeline). Segunda: arquitecturas con memoria temporal (`MlpLstmPolicy` en sb3-contrib o Transformer ligero) para capturar dependencias temporales de medio plazo. Tercera: curriculum learning para mejorar estabilidad inter-régimen entrenando primero sobre ventanas de baja volatilidad y escalando hacia regímenes adversos.

**P5.5 — ¿Cómo se vincula tu trabajo con los ODS?**
> Cuatro objetivos: ODS 7 (Energía asequible y no contaminante) — transparencia explícita sobre huella computacional, aproximadamente 28h CPU efectivo equivalentes a 0.3 kgCO₂. ODS 8 (Trabajo decente y crecimiento económico) — gestión patrimonial con riesgo integrado en la función objetivo, más resistente ante eventos adversos. ODS 9 (Industria, innovación e infraestructura) — DRL aplicado a finanzas con arquitectura MLOps reproducible. ODS 12 (Producción y consumo responsables) — asignación de capital trazable y simulable, reduce dependencia de juicio discrecional.

---

## GOTCHAS — 10 PREGUNTAS TRAMPA

**G.1 — "Tu PPO no supera ni al azar, ¿qué aporta el trabajo?"**
> Es un hallazgo negativo documentado, no un fracaso. El TFM aporta: (i) marco metodológico integrado que no existía en la literatura, (ii) protocolo de evaluación honesto con multi-seed y validación temporal estricta, (iii) evidencia empírica de que SAC sí captura señal con su mejor semilla, (iv) cuantificación de los límites de Optuna en problemas de baja relación señal-ruido. La transparencia sobre los resultados negativos es un requisito metodológico básico en investigación científica reproducible.

**G.2 — "La mejor semilla de SAC es solo cherry-picked"**
> Por eso se reporta siempre media ± std sobre N=5 semillas siguiendo el protocolo de Henderson 2018. La media de SAC LT (0.883) sigue siendo el mejor DRL del estudio, por encima de A2C (0.569) y PPO Optuna (0.474). La mejor semilla (1.280) se reporta como evidencia de que la señal existe, no como resultado central. Los intervalos de confianza al 95% se solapan entre SAC y A2C, lo que se reconoce explícitamente como limitación.

**G.3 — "¿Por qué la iteración 1 está en la memoria si no es entregable?"**
> Documenta el proceso experimental real. La caja de atención al inicio de la sección 4.7 declara explícitamente su carácter exploratorio. Sus conclusiones cualitativas (la recompensa dual añade ruido frente a Sharpe puro, el perfil agresivo concentra en cripto) se conservan como propiedades estructurales del agente, independientes del universo. Sus números absolutos no se utilizan como conclusión. Revalidar la ablación sobre el universo n=17 queda fuera del presupuesto experimental.

**G.4 — "Sharpe robusto +0.349 en WF con std 1.9 es decepcionante"**
> La std elevada (1.904) está dominada por la ventana 3 (Sharpe −3.160), que tiene volatilidad apenas por encima del umbral del filtro (3.41% frente al 1% mínimo). Está en la frontera del filtro de robustez. De las 7 ventanas significativas, 5 son positivas y 2 negativas. La ventana 7 obtiene Sharpe 2.258. La ventana 5 (crisis cripto 2022, Sharpe −1.273) muestra que el agente no generaliza a regímenes adversos no vistos durante entrenamiento. Es coherente con la naturaleza no estacionaria de los mercados, reconocido como limitación 2 del capítulo 5. Si se subiera el umbral a 5%, la ventana 3 también se excluiría y el Sharpe robusto medio subiría a +0.94 sobre 6 ventanas.

**G.5 — "El proxy BTC-USD/ETH-USD pre-2024 introduce sesgo"**
> Reconocido como limitación 1. El tracking error de un ETF spot frente al subyacente es del orden del 0.25% anual; la volatilidad diaria de Bitcoin es del 3-5%. La distorsión económica es despreciable a efectos de aprendizaje. Lo que el proxy ignora —comisiones de gestión, spreads, tracking error residual— se sumaría como un coste pequeño y aproximadamente constante. Cuando IBIT/ETHA acumulen mayor histórico real, conviene rehacer el experimento sin el proxy.

**G.6 — "¿Por qué SAC con menos pasos (500K vs 1.5M)?"**
> Justificación en Haarnoja et al. 2018: SAC es off-policy con replay buffer, lo que multiplica la utilización de cada experiencia. Su eficiencia muestral es mayor por construcción. Igualar a 1.5M en SAC saturaría el replay buffer (capacidad 100K) y reciclaría las experiencias muchas veces, lo que puede degradar el aprendizaje. No obstante, la asimetría de presupuesto se reconoce explícitamente como limitación 4 de la iteración 3: introduce un sesgo metodológico favorable a SAC que conviene declarar.

**G.7 — "Si A2C seed 1 obtiene 0.954, ¿por qué no usar A2C?"**
> Porque seleccionar un algoritmo por una semilla concreta es precisamente lo que Henderson 2018 desaconseja. La media de A2C LT es 0.569 ± 0.287, por debajo de SAC. La semilla 1 de A2C es una ejecución favorable; las semillas 0 y 3 obtienen 0.248 y 0.279 respectivamente. Un sistema de gestión patrimonial debe operar con expectativas robustas, no con resultados puntuales.

**G.8 — "Los costes de transacción del 0.1% son optimistas"**
> Es una limitación reconocida (línea futura 8). El modelo asume comisión lineal sobre el turnover, sin modelar impacto de mercado, libro de órdenes ni dependencia del tamaño de orden. Extender a un modelo de costes más realista (Almgren-Chriss, por ejemplo) es una mejora natural para un sistema con vocación productiva.

**G.9 — "¿Por qué no usaste GPU?"**
> La arquitectura MLP de dos capas de 256 unidades es ligera. El cuello de botella del entrenamiento no es el forward/backward de la red sino el cómputo del entorno Gymnasium (cálculo del Sharpe rolling, MDD, turnover, comisiones) por cada paso. Sobre un Intel i7 de portátil, cada entrenamiento de 1.5M pasos completa en 30-45 min sin GPU. Una GPU no mejoraría sustancialmente este perfil y aumentaría la huella energética innecesariamente.

**G.10 — "¿Por qué el frontend Angular si no es contribución?"**
> Tal como indicó el tutor, no se contabiliza como aportación central. Se incluye únicamente como anexo D, a título informativo, para documentar la totalidad del artefacto entregado y facilitar la inspección de los experimentos sin manejar peticiones HTTP a mano. El núcleo metodológico del TFM es el DRL aplicado a carteras híbridas, no la capa de presentación.

---

## METODOLOGÍA EXTRAS (10 preguntas)

**M.1 — ¿Diferencia entre walk-forward y expanding window?**
> Walk-forward usa ventana fija de 2 años de train + 1 año de test que se desliza adelante (sin solape). Expanding window mantiene el inicio del train fijo (2017-11-09) y lo acumula hasta el inicio de cada ventana, con test de 63 días. Expanding simula mejor un sistema en producción que se reentrena periódicamente con todo el histórico disponible.

**M.2 — ¿Por qué Z-score y no min-max?**
> Z-score es robusto a outliers (sigue siendo bien definido aunque haya valores extremos), invariante a escala de la feature original y matemáticamente coherente con el supuesto de distribución aproximadamente gaussiana en cada feature normalizada. Min-max comprime todos los valores al rango [0,1] pero un único outlier extremo distorsiona la escala del resto.

**M.3 — ¿Cómo se evita data leakage en el screener?**
> El screener se restringe a datos estrictamente anteriores al cutoff 2024-03-08. Ningún criterio del screener (volatilidad rolling, Sharpe rolling, sector GICS, deduplicación de tickers hermanos) usa información del periodo de test.

**M.4 — ¿Qué es GAE y por qué se usa?**
> Generalized Advantage Estimation. Combina estimaciones de la ventaja con distintos horizontes temporales mediante un promedio ponderado exponencial. Reduce la varianza del gradiente de política sin introducir sesgo significativo. Es el estándar en implementaciones modernas de PPO/A2C.

**M.5 — ¿Por qué clip a [−1, 1] y no [−5, +5]?**
> Las redes neuronales se entrenan con gradientes acotados. Un Sharpe rolling de −15 en un día catastrófico explotaría el gradiente y desestabilizaría la política durante muchos pasos. El rango [−1, 1] mantiene la escala del Sharpe rolling, MDD y turnover en órdenes de magnitud comparables, evitando que un componente domine la señal.

**M.6 — ¿Qué garantiza la inferencia determinista?**
> Reproducibilidad exacta de la curva de equity para un modelo dado. Misma red, mismos datos de test, misma curva. Permite comparar entre perfiles, configuraciones y baselines sin ruido estocástico del muestreo de la política gaussiana del actor.

**M.7 — ¿Cómo manejas el rebalanceo de las baselines?**
> Equal-Weight y 60/40 rebalancean mensualmente. Buy & Hold no rebalancea. Markowitz reestima con ventana de 252 días, rebalancea mensual. Momentum Top-3 actualiza con ventana de 60 días, rebalancea mensual. Random Uniform muestrea pesos nuevos mensualmente de una Dirichlet(1). DRL puede rebalancear a diario.

**M.8 — ¿Por qué Dirichlet(1) para Random Uniform?**
> Dirichlet(1) genera muestras uniformes sobre el simplex de pesos (vector de pesos no negativos que suman 1). Una uniforme independiente por componente, normalizada después, no es uniforme sobre el simplex —sesga hacia el centro.

**M.9 — ¿Qué papel juega la regularización por entropía en SAC?**
> SAC añade un término de entropía al objetivo del actor, ponderado por un coeficiente que el algoritmo ajusta automáticamente (`ent_coef='auto'`). Fuerza al agente a mantener una política con suficiente variabilidad —no colapsar a una asignación determinista demasiado pronto—, lo que mejora la exploración y produce políticas más suaves en problemas continuos.

**M.10 — ¿Por qué `min_periods=5` en el Sharpe rolling?**
> Antes del paso 5 no hay suficiente histórico de retornos para calcular un Sharpe rolling mínimamente fiable. En los primeros 4 pasos se usa el log-retorno diario escalado como señal de transición. A partir del paso 5 se activa el Sharpe sobre la ventana incremental disponible (5-19 días). Una vez alcanzados los 20 días, ventana fija deslizante.

---

## ARQUITECTURA TÉCNICA (5 preguntas)

**A.1 — ¿Por qué MLP [256, 256] y no más capas?**
> Valor recomendado por SB3 para problemas continuos de dimensionalidad media. Una red más pequeña ([64, 64]) podría no capturar la dinámica del estado. Una red mucho mayor ([512, 512, 512]) corre riesgo de sobreajustar con los pocos episodios efectivos del entrenamiento.

**A.2 — ¿Por qué FastAPI?**
> Tipado estático con Pydantic, generación automática de OpenAPI, soporte nativo async/await y rendimiento comparable a Node.js. Es la opción moderna estándar para microservicios Python en 2024-2025.

**A.3 — ¿Por qué SQLite y no PostgreSQL?**
> SQLite es suficiente para el alcance del TFM: usuarios de autenticación y metadatos de modelos entrenados. Cero overhead de despliegue (un único archivo). PostgreSQL sería la siguiente etapa para entorno productivo con concurrencia alta y auditoría completa.

**A.4 — ¿Por qué `curl_cffi` para la ingesta?**
> Yahoo Finance bloquea las cabeceras HTTP estándar de la librería `requests` en entornos corporativos con proxies SSL. `curl_cffi` emula la huella TLS de un navegador Chrome moderno, permite acceder a fuentes públicas sin modificar las políticas de seguridad de red. La ingesta se encapsula tras una interfaz Strategy para que el resto del pipeline sea agnóstico al proveedor.

**A.5 — ¿Cómo se persisten los modelos?**
> Cada modelo se serializa en formato `.zip` de Stable-Baselines3 en `backend/models/`. Sus metadatos (algoritmo, perfil, semilla, fecha, pasos de entrenamiento) se registran en la tabla `modelos` de SQLite. Las métricas de evaluación por experimento se persisten como JSON en `backend/src/reports/`.

---

## DISEÑO EXPERIMENTAL (5 preguntas)

**D.1 — ¿Por qué N=5 semillas y no N=10?**
> N=5 es el mínimo absoluto recomendado por Henderson 2018; la propia publicación advierte que N≥10 sería preferible para intervalos de confianza estrechos. Compromiso entre robustez estadística y presupuesto computacional: cada semilla de PPO/A2C consume 1.5M pasos = 30-45 min en CPU; cinco semillas × 3 algoritmos × 2 perfiles ≈ 28 horas. N=10 doblaría a 56 horas, inviable en un TFM. La extensión a N=10 está declarada como línea futura.

**D.2 — ¿Por qué solo `low_turnover` y `aggressive` en la iteración 3?**
> Para acotar el espacio de comparación. `low_turnover` es el perfil principal del TFM (justificado en el análisis de sensibilidad). `aggressive` es el perfil opuesto (penalizaciones mínimas, máxima libertad). El contraste entre ambos cubre el rango operativo del catálogo. `balanced` y `conservative` son intermedios y no aportan información discriminativa adicional sobre el comportamiento algorítmico.

**D.3 — ¿Por qué Random Uniform y Momentum Top-3?**
> Random Uniform responde a la pregunta de cordura mínima: ¿el agente realmente aprende o sus resultados son consistentes con un resultado aleatorio? Si el DRL no bate a la asignación aleatoria, no está aprendiendo nada del estado. Momentum Top-3 responde a la pregunta competitiva: ¿el DRL aporta algo por encima de un factor cuantitativo clásico y operativamente sencillo? Si no, la aportación incremental debe matizarse.

**D.4 — ¿Cómo se decidió el cutoff 2024-03-08?**
> Coincide con el cierre del último día anterior al inicio del periodo de test (2024-03-09 es el primer día de test). Se eligió el split 80/20 sobre el dataset 2017-11-09 → 2026-04-30, lo que sitúa el corte automáticamente en esa fecha. El screener se restringe a datos hasta el cierre del 2024-03-08 para garantizar zero solapamiento entre criterio de selección y periodo de evaluación.

**D.5 — ¿Qué pasaría si quitas IBIT y ETHA del universo?**
> El universo perdería su naturaleza híbrida y el trabajo dejaría de responder a su pregunta de investigación. Para validar que los criptoactivos aportan información, una ablación natural sería comparar el rendimiento del agente con y sin cripto en el universo. No se ha realizado por presupuesto experimental y se reconoce como línea futura implícita.

**N.5 — NOTA 10: ¿Por qué walk-forward y no k-fold cross-validation?**
> Porque k-fold mezcla pasado y futuro en cada fold y eso es data leakage temporal. En finanzas el orden importa: entrenar con datos de 2025 para predecir 2020 viola la causalidad. Walk-forward y expanding window respetan estrictamente el orden temporal —train siempre anterior a test. López de Prado 2018 desaconseja explícitamente k-fold para series financieras y propone purged k-fold con embargo como única variante aceptable; no la usamos porque walk-forward + expanding ya cubren la robustez temporal con menor complejidad.

**N.7 — NOTA 10: ¿Por qué decisiones diarias y no intradiarias?**
> Tres razones. (i) Los ETFs IBIT/ETHA tienen liquidez intradiaria buena pero los activos S&P 500 seleccionados varían; operar intradiario complicaría el modelo de slippage. (ii) Yahoo Finance API no provee tick data fiable; pasar a intradiario requiere proveedor de pago (Polygon.io, Refinitiv). (iii) La pregunta de investigación es asignación de cartera, no microestructura de mercado —los hedge funds que sí operan intradiario lo hacen sobre estrategias de market making o latencia, no asignación dinámica. El horizonte diario es el estándar de la literatura DRL aplicada a portfolio management (FinRL, Jiang 2017).

**N.8 — NOTA 10: ¿Por qué MLP y no LSTM, Transformer o CNN sobre features?**
> MLP es la arquitectura de referencia en SB3 para problemas continuos y permite una comparación limpia entre algoritmos (PPO/A2C/SAC con misma red). Arquitecturas recurrentes (LSTM, Transformer) añaden memoria temporal explícita pero también complejidad e hiperparámetros adicionales; el agente MLP ya recibe features con horizonte temporal embebido (momentum 5d/20d/60d, volatilidad rolling). CNN sobre features tendría sentido si las features tuvieran estructura espacial 2D (heatmap correlaciones), no es el caso. La extensión a `MlpLstmPolicy` (RecurrentPPO en sb3-contrib) es la primera línea futura del trabajo (sección 5.4), justamente porque es la mejora arquitectónica más natural.

---

## CRÍTICA Y LIMITACIONES (5 preguntas)

**C.1 — ¿Realmente probaste que SAC es mejor que PPO?**
> En este universo y en este periodo de test, SAC obtiene mejor Sharpe medio que PPO con N=5 semillas. Pero los intervalos de confianza al 95% se solapan parcialmente, y el rango inter-semilla de SAC (0.748) supera a la diferencia entre medias (0.41). Una comparativa estadísticamente concluyente requeriría N≥10 y test de hipótesis (Mann-Whitney, por ejemplo). El trabajo lo declara explícitamente: comparación indicativa, no definitiva.

**C.2 — ¿Por qué no usaste un test de hipótesis?**
> Con N=5 semillas por configuración, los tests no paramétricos como Mann-Whitney tienen muy poca potencia estadística. Hubiera arrojado p-valores no significativos para casi todas las comparaciones, sin aportar información. Reportar media±std + rango es más informativo en esta escala muestral. Pasar a tests formales requiere N≥10 (mínimo recomendado por Henderson) y está declarado como línea futura.

**C.3 — La iteración 3 favorece a SAC por el presupuesto asimétrico.**
> Reconocido explícitamente como limitación 4 de la iteración 3. PPO y A2C entrenan 1.5M pasos, SAC solo 500K. La justificación es la mayor eficiencia muestral de SAC como algoritmo off-policy con replay buffer (Haarnoja 2018). Igualar a 1.5M en SAC saturaría el replay buffer (capacidad 100K). No obstante, conviene reconocer el sesgo metodológico.

**C.4 — ¿Y si tu test fuera otro régimen de mercado?**
> El periodo de test cubre marzo 2024 a abril 2026: ciclo alcista tech, fase lateral en Bitcoin, mercado bajista en Ethereum. No hay recesión amplia, crisis financiera tipo 2008 ni black swan cripto. Las métricas son válidas para este régimen, no se garantizan para otros. La validación temporal walk-forward y expanding window —que sí cubren el shock cripto 2022— mostraron precisamente que el agente pierde capital en transiciones de régimen no vistas durante entrenamiento. Se declara como limitación 5.

**C.5 — ¿El proxy BTC-USD/ETH-USD invalida las conclusiones?**
> No. El tracking error de un ETF spot es del orden del 0.25% anual; la volatilidad diaria de Bitcoin es del 3-5%. La distorsión es despreciable a efectos de aprendizaje. Lo que el proxy ignora —comisiones de gestión, spreads, tracking error residual— se sumaría como un coste pequeño y aproximadamente constante. Cuando IBIT y ETHA acumulen mayor histórico real, conviene rehacer el experimento sin el proxy. Es la limitación 1 del trabajo.

**N.3 — NOTA 10: ¿Qué te separa de un sistema de un hedge fund cuantitativo real?**
> Cuatro brechas. (i) Datos: hedge funds usan order book L2 + datos alternativos (sentiment, satellite, credit card); yo uso solo OHLCV diario público. (ii) Modelado de costes: ellos modelan impacto de mercado con Almgren-Chriss; yo asumo comisión lineal 0.1%. (iii) Latencia: ellos operan en microsegundos; yo en cierre diario. (iv) Universo: ellos cubren miles de activos globales; yo 17. Un hedge fund con DRL produciría Sharpe superior porque cada una de esas dimensiones añade información o reduce coste. Mi trabajo es un prototipo metodológico, no un competidor a Renaissance o Two Sigma.

**Z.3 — NOTA 10: ¿Has detectado reward hacking? ¿El agente encuentra atajos no deseados?**
> Sí, una vez. En el universo legacy n=15, el perfil aggressive (φ=0.01, γ=0.005) producía concentración extrema en IBIT (>40% del peso medio), explotando la baja penalización por MDD para maximizar Sharpe en el régimen alcista cripto del periodo. No es estrictamente reward hacking porque la recompensa funcionaba como definida, pero sí un caso de "el agente encuentra el comportamiento de máxima recompensa que el diseñador no había anticipado". Mitigación aplicada: el perfil `low_turnover` (γ=0.020, doble penalización turnover) fuerza diversificación implícita y elimina ese patrón. Otra señal de posible hacking: las primeras ventanas del walk-forward donde el agente colapsa a casi-liquidez con vol <1%, capturado por el filtro de robustez (sección 4.3). Es un comportamiento aprendido —no rotar protege de penalización por turnover en periodos sin señal clara.

---

## PERSONALES Y PROYECTO (5 preguntas)

**P.1 — ¿Cuánto tiempo te llevó el trabajo?**
> Cuatro meses según la planificación inicial del diagrama de Gantt: mes 1 ingesta y exploración, mes 2 entorno Gymnasium, mes 3 entrenamiento DRL, mes 4 evaluación, validación temporal y redacción. La iteración 1 (universo viejo) consumió el tercer mes; la detección del data leakage y la iteración 2 + 3 sobre universo honest, junto con la búsqueda Optuna, ocuparon el cuarto mes y parte del quinto.

**P.2 — ¿Qué fue lo más difícil del proyecto?**
> Detectar el data leakage en el screener original a mitad del proyecto y aceptar que las métricas iniciales (Sharpe 1.875) eran académicamente incorrectas. Rehacer todo sobre el universo honest implicó renunciar a resultados visualmente impactantes a cambio de resultados honestos pero más modestos (Sharpe 0.474-0.883 según algoritmo). La decisión metodológica correcta era clara, pero supuso reescribir gran parte del capítulo de resultados.

**P.3 — ¿Qué cambiarías si lo empezases hoy?**
> Cuatro cosas. Primero, implementar desde el inicio el screener con cutoff temporal estricto, no descubrirlo a mitad del proyecto. Segundo, integrar el régimen GMM como feature del estado del agente desde la primera versión del entorno. Tercero, usar VecEnv vectorizado para A2C desde el principio, adaptando los callbacks académicos a la agregación multi-entorno. Cuarto, ampliar el alcance de Optuna a 200+ trials con pruner menos agresivo, o saltar directamente a CMA-ES como optimizador alternativo.

**P.4 — ¿Cuál es la aplicación real?**
> Como punto de partida metodológico para un sistema productivo de gestión patrimonial con exposición a criptoactivos regulados. La infraestructura modular permitiría reentrenamiento periódico, simulación personalizada por inversor y trazabilidad de cada decisión. No es un producto comercial sino un prototipo funcional con orientación MLOps que demuestra la viabilidad técnica del enfoque.

**P.4b — TUTOR: ¿Este sistema sirve para hacerse rico o ganar dinero de forma automática?**
> No. Es un trabajo académico, no un sistema de inversión productivo. Tres razones operativas: (i) los resultados sobre universo honest sin sesgo de selección muestran que ni siquiera el mejor algoritmo (SAC) supera a Random Uniform en media, lo que implica que un inversor podría obtener resultados comparables muestreando pesos al azar; (ii) los costes reales (slippage variable, impacto de mercado, comisiones de gestión del ETF, fiscalidad) están idealizados a un 0.1% lineal en el TFM, lo que infravalora el coste operativo; (iii) los mercados son no estacionarios y la validación temporal muestra que el agente pierde capital en transiciones de régimen que no ha visto durante entrenamiento. El valor del trabajo es metodológico —protocolo de evaluación honesto, marco integrado—, no operativo. Cualquier promesa de rentabilidad automática en finanzas debe ser tratada con extrema cautela.

**P.5 — ¿Por qué te interesa este tema?**
> Convergen tres líneas de mi interés: la frontera del machine learning (DRL aplicado a problemas reales), la institucionalización de los criptoactivos (un cambio cualitativo de mercado de los últimos dos años) y la gestión cuantitativa de carteras (un dominio con tradición matemática sólida desde Markowitz 1952). El TFM me permitió integrar las tres en un único trabajo aplicado.

**Z.6 — NOTA 10: ¿Qué asignaturas del máster han contribuido directamente a este TFM?**
> Tres núcleos. Primero, "Aprendizaje computacional" y "Aprendizaje por refuerzo" aportaron los fundamentos de DRL: MDP, policy gradient, actor-crítico. Segundo, "Métodos estadísticos avanzados" y "Estadística aplicada" sustentan la metodología de validación (multi-seed, intervalos de confianza, tests de hipótesis no paramétricos). Tercero, "Diseño y desarrollo de aplicaciones web" + "Bases de datos avanzadas" aportaron la arquitectura FastAPI + SQLite + Angular. El TFM integra estas tres líneas en un único trabajo aplicado al dominio financiero, que es donde aporto conocimiento previo profesional ajeno al máster.

---

## COBERTURA 360 — 8 PREGUNTAS BLINDADAS

**B.1 — Interpretabilidad: ¿puedes explicar POR QUÉ tu agente toma una decisión concreta?**
> Razón. La política PPO/SAC es una red neuronal: no hay regla explícita. Mitigaciones aplicadas: (i) inferencia determinista (curva de equity reproducible bit a bit), (ii) trazabilidad de pesos día a día (`diagnose_ppo_weights.csv`), (iii) análisis estructural en sección 4.5 (asignación promedio, exposición cripto, sectores). Lo que NO ofrece: justificación causal de cada acción individual. Línea futura natural: aplicar SHAP o Integrated Gradients sobre la red para atribuir importancia de features por decisión.

**B.2 — Métricas: ¿por qué reportas Sharpe, Sortino, CAGR, Vol y MDD?**
> Cada métrica responde una pregunta distinta. Retorno bruto ignora riesgo. Sharpe (retorno excedente / volatilidad anualizada) penaliza volatilidad simétrica. Sortino solo penaliza volatilidad a la baja, más alineado con la percepción del inversor. CAGR anualiza el retorno para comparabilidad entre periodos de distinta duración. Vol anualizada describe el riesgo diario amplificado. MDD captura la peor caída sostenida, más relevante para retención de inversores que la volatilidad media. El conjunto da una imagen completa; reportar solo una sería caer en cherry-picking métrico.

**B.3 — Reproducibilidad técnica: ¿cómo garantizas reproducibilidad más allá de la inferencia determinista?**
> Cuatro mecanismos: (i) semillas fijas en `np.random.seed`, `torch.manual_seed`, `env.seed`, (ii) versiones congeladas en `requirements.txt`, (iii) datos CSV persistidos en `backend/data/` con commit git, (iv) JSONs de métricas por experimento con hiperparámetros completos para re-entrenamiento. Limitación reconocida: el entrenamiento DRL es estocástico por construcción (exploración del actor); lo reproducible es la evaluación de un modelo guardado, no el proceso de entrenamiento desde cero.

**B.4 — Generalización: ¿tu sistema funcionaría en otro mercado (europeo, asiático, FX)?**
> Conceptualmente sí: el MDP y la recompensa son agnósticos al universo. Prácticamente requiere: (i) reentrenar con datos del nuevo mercado, (ii) ajustar el screener al universo correspondiente (Eurostoxx, Nikkei, pares FX), (iii) recalibrar el proxy si hay activos sin histórico completo, (iv) validar los supuestos de fricciones (las comisiones europeas difieren de USA). La arquitectura modular FastAPI + Strategy pattern facilita el cambio de proveedor de datos. Queda como línea natural futura.

**B.5 — Literatura: ¿hay paper con Sharpe comparable sobre universo similar?**
> La comparación directa es difícil porque la literatura DRL financiero usa universos heterogéneos: Jiang 2017 trabajaba sobre criptomonedas spot puro (sin tradicional), FinRL reporta sobre Dow Jones 30 sin cripto, otros papers sobre futuros o FX. Cada elección de universo y periodo de test genera Sharpe muy distintos, lo que dificulta comparaciones directas. Lo que sí compartimos con la literatura es la advertencia metodológica de Henderson 2018 sobre varianza inter-semilla, que aplicamos sistemáticamente.

**B.6 — Coste energético / ODS 7: ¿qué huella de carbono tiene tu trabajo?**
> Aproximadamente 28 horas CPU efectivo en Intel i7 portátil sin GPU. Con consumo medio 75W ≈ 2.1 kWh. Factor mix eléctrico español 2024 (0.15 kgCO₂/kWh, Red Eléctrica de España) ≈ 0.3 kgCO₂, equivalente a 1.5 km de coche convencional. Una búsqueda Optuna sobre A2C/SAC con N≥10 semillas multiplicaría esto por un orden de magnitud. La transparencia sobre el coste computacional se considera un requisito metodológico básico en DRL responsable. Cubierto en sección 5.4 ODS 7.

**B.7 — Ingeniería SW: ¿tienes tests? ¿CI? ¿Cómo verificas que el entorno funciona?**
> Tests smoke críticos: `python -c "from src.training_drl.training_analysis import ..."` valida imports. Validación del entorno mediante baselines simples (Equal-Weight, Random Uniform) que sirven como agentes canario: si el entorno tiene bugs en el cálculo de recompensa o comisiones, estos baselines también arrojarían resultados anómalos. CI no implementado (alcance TFM); el trabajo es prototipo funcional, no producto. Tests unitarios formales con pytest = línea futura para etapa productiva.

**B.8 — Speculative GMM+K-Means: ¿qué papel juega?**
> Es una baseline ADICIONAL al catálogo de 6 clásicas. Combina GMM (detector de régimen volatilidad alta/baja) con K-Means (clustering de activos por similitud de retornos). Asigna pesos según el cluster dominante en cada régimen. Sharpe 0.868, retorno +81.97%, por debajo de Random Uniform pero por encima de varias clásicas. Demuestra que un enfoque de clustering simple no batería al azar en este universo, lo que refuerza la dificultad del problema.

**N.4 — NOTA 10: Un regulador (CNMV, ESMA) ¿aprobaría tu sistema para gestión patrimonial?**
> No tal cual. MiFID II y el KID (Key Information Document) requieren transparencia sobre la lógica de inversión que un agente DRL black-box no provee. Para uso regulado se necesitaría: (i) capa de explicabilidad (SHAP/Integrated Gradients sobre cada decisión), (ii) límites duros sobre exposición por activo/sector (no solo penalizaciones suaves en recompensa), (iii) circuit breakers ante drawdowns extremos, (iv) auditoría continua del drift de la política. El TFM es prototipo de I+D, no producto desplegable bajo MiFID II actual. La AI Act europea (2024) clasificaría el sistema como "high-risk AI" si se usase para asesoramiento financiero, lo que añadiría requisitos adicionales de documentación y testing.

**N.6 — NOTA 10: ¿Por qué Sharpe y Sortino y no Calmar, Omega, Tail Risk?**
> Compromiso entre interpretabilidad y completitud. Sharpe y Sortino son las dos métricas que un comité de inversión institucional reconoce inmediatamente. Calmar (CAGR/MDD) está implícitamente cubierto reportando CAGR y MDD por separado. Omega ratio captura la asimetría de la distribución pero es menos estándar. CVaR/Expected Shortfall sería relevante para una crítica de tail risk pero requiere asumir una distribución paramétrica o estimación no paramétrica con suficientes datos en la cola, no robusto con 579 días de test. Añadir 5+ métricas dispersaría la atención sin aportar discriminación adicional entre estrategias. Es elección consciente, no omisión.

---

## BONUS — PREGUNTAS INESPERADAS

**X.1 — ¿Por qué no usaste un LLM (GPT, Claude) en el TFM?**
> Fuera del alcance metodológico. Los LLMs no son agentes de decisión secuencial con espacio de acción continuo; son modelos generativos de texto. Aplicar un LLM a asignación de cartera requeriría un wrapper específico (RLHF con función de recompensa financiera, in-context learning sobre historial de mercado) y dejaría de ser DRL clásico. Es un enfoque distinto, línea futura interesante pero no compatible con el alcance "Proximal Policy Optimization sobre Gymnasium" del TFM.

**X.2 — ¿Cuál es el límite de tu trabajo: hasta dónde lo defendería en una conferencia académica?**
> Defendería en cualquier foro la metodología: marco integrado, validación temporal estricta, multi-seed, transparencia sobre resultados negativos. Lo que NO defendería en una conferencia: afirmar que SAC es definitivamente superior a PPO en cartera híbrida (N=5 insuficiente para conclusión estadística) ni que el sistema está listo para producción (faltan tests formales, modelado realista de costes, validación sobre crisis históricas).

**X.3 — Si tuvieras que reducir el trabajo a una frase, ¿cuál sería?**
> "Construí un marco metodológico integrado para evaluar agentes DRL en carteras híbridas con criptoactivos institucionales, y documenté con honestidad tanto los hallazgos positivos (SAC mejor semilla supera Random Uniform) como los negativos (PPO Optuna no supera Random Uniform en media)."

---

## IMPLEMENTACIÓN / CÓDIGO (12 preguntas)

**I.1 — ¿Cómo está organizado el repositorio?**
> Estructura modular: `backend/main.py` expone los endpoints FastAPI (auth, fase1-3, simulación). `backend/src/training_drl/` contiene `environment_trading.py` (PortfolioEnv Gymnasium), `training_analysis.py` (train_academic + 3 callbacks PPO/A2C/SAC), `risk_profiles.py` (4 perfiles). `backend/src/hpo/` contiene Optuna (`objective_ppo.py`, `space.py`, `eval_metrics.py`). `backend/src/benchmarking/baselines.py` (6 baselines + compute_metrics canónico). `backend/src/unsupervised/speculative_agent.py` (GMM+K-Means). `backend/src/reports/` contiene generadores de figuras + JSONs de métricas. `backend/data/` precios y features. `backend/models/` modelos serializados. `backend/scripts/` scripts standalone para retrain.

**I.2 — ¿Qué hace `PortfolioEnv.step(action)` exactamente?**
> Recibe vector de acción crudo del actor. Pasos: (i) clipping a no-negativos + normalización a simplex (suma 1). (ii) Salvaguarda equipesos si suma <ε=10⁻³. (iii) Calcula diff de pesos vs día anterior → turnover. (iv) Aplica comisión = `commission_rate × turnover` sobre el valor cartera. (v) Avanza un paso en serie de precios, calcula nuevo valor cartera con pesos aplicados. (vi) Calcula Sharpe rolling 20d, MDD, retorno desde valor previo. (vii) Computa recompensa `clip(Sharpe − φ·MDD − γ·turnover, −1, 1)`. (viii) Construye nueva observación (features + pesos + última reward). (ix) Detecta done: fin de episodio o caída >90% capital. (x) Devuelve `(obs, reward, done, truncated, info)`.

**I.3 — ¿Cómo implementaste el rebalanceo con comisiones?**
> Comisión proporcional al turnover absoluto: `cost = commission_rate × Σ|wᵢ,ₜ − wᵢ,ₜ₋₁|`, donde `commission_rate = 0.001` (0.1%). El valor de la cartera se decrementa por ese coste ANTES de aplicar los retornos del día, modelando que la operación se ejecuta al cierre con el precio del día previo. Slippage parametrizable adicional incluido pero deshabilitado por defecto en los experimentos del trabajo (0%). El cálculo del turnover es el L1-norm de la diferencia de pesos, escalado a [0, 2] máximo (rotación completa).

**I.4 — ¿Qué hace `_normalize_window(df_raw, train_end_idx)`?**
> Recalcula media y desviación típica por feature SOLO sobre los datos `df_raw[:train_end_idx]` (estrictamente anteriores al inicio de la ventana de test). Aplica Z-score sobre TODO el dataframe (train + test) usando esos estadísticos. Garantiza que ningún estadístico del periodo de test contamine la normalización del periodo de entrenamiento. Se invoca automáticamente en `walk_forward_validation` y `expanding_window_validation` siempre que esté presente el CSV de features sin normalizar. Sin esta función, la versión inicial sufría lookahead residual documentado como limitación 3 del capítulo 5.

**I.5 — ¿Cómo se carga un modelo entrenado desde la API FastAPI?**
> El endpoint `/admin/fase3/entrenar-academico` recibe el algoritmo (PPO/A2C/SAC) y el perfil de riesgo, lanza el training en background y persiste el modelo en `backend/models/best_model_academic_{ALGO}_{PERFIL}_seed{N}/best_model.zip`. El endpoint `/admin/fase4/simular` instancia `SAC.load()` o `PPO.load()` desde Stable-Baselines3 leyendo el path, ejecuta inferencia determinista (`deterministic=True`) sobre el entorno de test y devuelve la curva de equity + métricas. Los metadatos del modelo (algoritmo, perfil, seed, pasos, fecha) viven en tabla SQLite `modelos`.

**I.6 — ¿Qué patrón de diseño usaste para los algoritmos PPO/A2C/SAC?**
> Strategy + Factory. La función `train_academic(algorithm, ...)` actúa como factory que instancia la clase SB3 correspondiente (`PPO`, `A2C`, `SAC`) y le inyecta el callback monitor adecuado (`AcademicMonitorCallback` para PPO, `A2CMonitorCallback`, `SACMonitorCallback`). Los tres comparten el mismo entorno `PortfolioEnv` y la misma `MlpPolicy` con `net_arch=[256, 256]`. La diferenciación entre algoritmos se reduce a los hiperparámetros específicos (clip_range PPO, replay_buffer SAC, n_steps A2C) y al callback. Para añadir un cuarto algoritmo (TD3, DDPG) bastaría con una entrada nueva en el `if/elif` de la factory y un callback compatible.

**I.7 — ¿Cómo se persisten las simulaciones del rol inversor?**
> Cada llamada a `/investor/simular` genera un objeto con parámetros (capital inicial, comisión, split train/test) + curva de equity + métricas. Persistencia mixta: parámetros y métricas en tabla SQLite `historial_simulaciones`, curva de equity completa serializada como JSON o CSV en disco bajo `backend/simulations/{user_id}/{sim_id}.json` para evitar inflar la BD con series temporales largas. El frontend Angular lee la BD para el listado histórico y carga el JSON bajo demanda al abrir una simulación concreta.

**I.8 — ¿Por qué un lock file `models/.training.lock`?**
> Tres razones operativas: (i) el endpoint `/estado` lo lee para reportar al frontend que hay training en curso sin necesidad de inspeccionar procesos; (ii) sirve como mutex: si llega una segunda petición de training mientras hay uno corriendo, el endpoint la rechaza con 409 Conflict en lugar de lanzar dos trainings concurrentes que se pisarían el modelo; (iii) protege contra ediciones de `src/training_drl/*` o `main.py` durante el entrenamiento —el server reload de uvicorn mataría el proceso de BackgroundTasks y se perdería el modelo a medio entrenar. El lock se crea al inicio del background task y se elimina en el `finally` para garantizar cleanup incluso ante excepciones.

**I.9 — ¿Qué monitoriza `OverfitDetectorCallback`?**
> El gap entre la reward media en entrenamiento y la reward media en evaluación periódica sobre el conjunto de validación. Si el gap supera un umbral configurado durante N evaluaciones consecutivas, dispara `early stop` deteniendo el entrenamiento. Su función es académica más que práctica: en DRL el "overfitting" tradicional es ambiguo (la política puede converger a soluciones distintas según semilla), pero un gap creciente entre train y eval es indicador de que el agente memoriza el rollout en lugar de aprender la dinámica. En los entrenamientos finales del trabajo, el callback raramente disparó porque PPO con clip_range conservador es naturalmente estable.

**I.10 — ¿Qué loguea `AcademicMonitorCallback`?**
> Por episodio: reward media, retorno final, MDD, turnover medio, Sharpe del episodio, número de pasos. Por evaluación periódica (cada N steps): mismas métricas sobre el conjunto eval determinista. Persiste todo en `src/reports/training_log_{algo}.csv` para post-mortem y generación de `training_diagnostics_{algo}.png` y `overfitting_analysis_{algo}.png`. La versión específica para A2C (`A2CMonitorCallback`) ajusta la frecuencia de log a la cadencia más rápida de actualizaciones del algoritmo. La versión para SAC (`SACMonitorCallback`) trackea adicionalmente el coeficiente de entropía auto-tuned.

**I.11 — ¿Cómo lanza FastAPI un training de 1.5M pasos sin bloquear la API?**
> Mediante `BackgroundTasks` nativo de FastAPI. El endpoint recibe la petición, valida parámetros, crea el lock file y devuelve respuesta HTTP 202 Accepted inmediata. El training real se delega a una función que `BackgroundTasks` ejecuta tras devolver la respuesta. Limitación: si el proceso uvicorn muere (cierre de terminal, reload), el training también muere —no hay persistencia de procesos huérfanos. Para sobrevivir cierre de terminal en mi setup, lanzo el server con `Start-Process -WindowStyle Hidden`. Para producción real se necesitaría Celery + Redis o equivalente; queda como línea de evolución hacia despliegue MLOps completo.

**I.12 — ¿Qué tests/smoke checks tienes implementados?**
> Tests smoke críticos del tipo `python -c "from src.training_drl.training_analysis import train_academic, AcademicMonitorCallback"` que validan que los imports principales funcionan tras cualquier edit. Validación implícita del entorno mediante baselines simples: `Equal_Weight` y `Random_Uniform` actúan como agentes canario; si el entorno tuviera bugs en cálculo de recompensa/comisiones, estos baselines también arrojarían resultados anómalos. Validación cruzada de números: los JSONs de `src/reports/` se cruzan contra las tablas del `tfm.tex` mediante un script de chequeo. NO hay pytest formal ni CI; reconocido como deuda técnica para etapa productiva (línea futura B.7).

**Z.2 — NOTA 10: ¿Cómo se define un episodio en tu entorno? ¿Cuándo termina?**
> Cada episodio cubre el periodo completo train o test —desde el primer día disponible hasta el último— como un único rollout continuo. Condiciones de terminación (done=True): (i) `current_step >= len(features)`, fin natural del periodo, o (ii) `portfolio_value < 0.1 * initial_capital`, caída del 90% del capital (salvaguarda contra runs catastróficos que distorsionarían el aprendizaje). Cada entrenamiento ejecuta múltiples episodios completos sobre el mismo periodo train hasta agotar `total_timesteps` (1.5M pasos PPO/A2C, 500K SAC). El reset reinicia `portfolio_value` al capital inicial y los pesos a equipesos. No hay random start del episodio en mid-period, lo que simplifica reproducibilidad a costa de menos diversidad de trayectorias.

**Z.4 — NOTA 10: ¿Qué valor usas para el discount factor γ del MDP?**
> γ = 0.99 (valor por defecto SB3 para PPO/A2C/SAC). Esto da horizonte efectivo de aproximadamente 100 pasos = ~5 meses de mercado, coherente con horizontes de inversión institucional medio plazo. γ más bajo (0.9) habría producido un agente miope que solo optimiza el siguiente día; γ más alto (0.999) habría diluido la señal de recompensa diaria contra retornos lejanos inciertos. No realicé sensibilidad sobre γ porque el valor 0.99 está consolidado en literatura DRL y la calibración del trabajo se centró en hiperparámetros con mayor potencial discriminativo (clip_range, learning_rate, batch_size). Atención conceptual: γ del MDP (discount factor) NO es el mismo γ de la penalización por turnover en la recompensa. La memoria usa la misma letra por convención de notación; son parámetros distintos.

**Z.5 — NOTA 10: ¿Usas precios ajustados por splits y dividendos?**
> Sí. Yahoo Finance API devuelve `Adj Close` por defecto, que incorpora ajustes retroactivos por splits y dividendos. Esto garantiza que un retorno calculado como `log(P_t / P_{t-1})` refleje el cambio económico real para el inversor, no artefactos corporativos (un split 2:1 no es una caída del 50%). Sin esta corrección, los retornos calculados en días de split arrojarían valores anómalos que distorsionarían el aprendizaje del agente. Verificable en `backend/data/original_prices.csv`: los precios históricos de activos como NVDA muestran el ajuste retroactivo por el split 10:1 de junio 2024 visible como salto suave, no como discontinuidad.

---

## CIERRE — MENSAJE PRINCIPAL (memoriza literal)

> "Este trabajo construye un sistema de gestión dinámica de carteras híbridas con DRL. La principal contribución metodológica es un marco integrado que combina recompensa multiobjetivo, validación temporal estricta sobre universo honest sin sesgo de selección, y protocolo multi-seed siguiendo Henderson 2018. Los resultados muestran que existe señal aprendible —la mejor semilla de SAC supera a Random Uniform con Sharpe 1.280— pero su captura sistemática es el reto no resuelto: la varianza inter-semilla con N=5 semillas impide afirmar dominancia estadística. Documentamos como hallazgo negativo que Optuna no mejora sobre la calibración manual en este dominio. Las limitaciones se reconocen explícitamente y las líneas futuras —régimen como feature, arquitecturas recurrentes, curriculum learning— quedan bien definidas."

---

## ÍNDICE FINAL

| Bloque | Preguntas |
|---|---|
| Cap 1 Introducción | 3 |
| Cap 2 Estado del Arte | 3 |
| Cap 3 Materiales y Métodos | 11 (incluye P3.3b TUTOR, N.2 cripto, Z.1 survivorship) |
| Cap 4 Resultados | 10 (incluye P4.4b TUTOR, N.1 overfitting) |
| Cap 5 Conclusiones | 5 |
| Gotchas | 10 |
| Metodología extras | 10 |
| Arquitectura técnica | 5 |
| Diseño experimental | 8 (incluye N.5 k-fold, N.7 intradiario, N.8 arquitecturas) |
| Crítica y limitaciones | 7 (incluye N.3 hedge fund, Z.3 reward hacking) |
| Personales y proyecto | 7 (incluye P.4b TUTOR hacerse rico, Z.6 asignaturas) |
| Cobertura 360 | 10 (incluye N.4 regulación, N.6 métricas) |
| Bonus inesperado | 3 |
| Implementación / Código | 15 (incluye Z.2 episodio, Z.4 discount γ, Z.5 adjusted close) |
| **TOTAL** | **107** |

---

## CHECKLIST PRE-DEFENSA

### Técnico
- [ ] PDF compilado limpio (sin warnings críticos de bibtex, sin refs `??`)
- [ ] Revisión visual portada UOC
- [ ] Todas las tablas/figuras numeradas correctamente
- [ ] Bibliografía 23 entradas verificadas
- [ ] Repositorio git: tag de versión entrega creado
- [ ] Backup PDF en 3 ubicaciones (USB, email enviado a ti mismo, nube)

### Slides
- [ ] Slides revisados: motivación, problema, método, resultados clave, limitaciones, líneas futuras
- [ ] Cronometrar: 20-25 minutos de presentación, dejar 10-15 min para Q&A
- [ ] Capturar screenshot del dashboard funcional como respaldo visual
- [ ] Transición de slides ensayada (sin pausas largas)

### Ensayos
- [ ] Ensayo 1: completo, cronometrado, sin parar (D-5)
- [ ] Ensayo 2: grabar audio, detectar muletillas (D-3)
- [ ] Ensayo 3: final, frente a alguien si es posible (D-2)

### Q&A
- [ ] Repasar las 78 preguntas en voz alta (al menos las 25 más críticas)
- [ ] Tener listas las respuestas a los 10 gotchas
- [ ] Memorizar los 5 números clave

### Logística defensa
- [ ] Confirmar día, hora, lugar/enlace con la UOC
- [ ] Probar audio/video si es virtual con 24h de antelación
- [ ] Ropa preparada con 2 días de antelación
- [ ] Llegar/conectarse 30 min antes
- [ ] Dormir bien la noche del 14/06
