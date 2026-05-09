"""
Entorno de trading basado en Gymnasium para gestión de carteras con DRL.

Resumen del modulo:
    Es el "tablero de juego" del agente PPO. Define las reglas con las que
    el agente interactúa con el mercado: qué ve, qué puede hacer, qué premio
    recibe por cada acción y cuándo termina la partida.

Conceptos clave (glosario rápido para situarnos):

  - Entorno (env): el simulador con el que el agente practica. Aquí, una
    cartera virtual con $10 000 que se actualiza día a día con precios reales.
  - Episodio: una "partida" completa, desde el primer día del periodo hasta
    el último. El agente entrena viendo muchos episodios.
  - Step: un paso temporal = un día de mercado simulado.
  - Observación: lo que el agente ve cada step para decidir (features de
    mercado + estado de su propia cartera). Más detalle en _build_obs().
  - Acción: lo que el agente decide cada step. Aquí: un vector de pesos
    (cuánto invertir en cada activo).
  - Reward (recompensa): la nota que el entorno le da al agente por su
    acción. Aquí compuesta: Sharpe rolling − phi·MDD − gamma·Turnover.
  - Propiedad de Markov: el estado actual debe contener TODA la información
    relevante para tomar la decisión. No vale "lo que pasó hace 3 días".
  - Referencias tomadas de: https://gymnasium.farama.org/


Por qué Gymnasium:
    Gymnasium (sucesor de OpenAI Gym) es la interfaz estándar de la industria
    para RL. Cualquier algoritmo compatible (PPO, A2C, SAC, DQN...) puede
    entrenar contra cualquier entorno Gymnasium sin modificar el algoritmo.
    Eso permite cambiar de algoritmo o de entorno de forma independiente:
    arquitectura desacoplada.

Flujo de un episodio:
    1. reset()→ coloca al agente al inicio con cartera equilibrada.
    2. step(action) × N  → cada llamada avanza un día y devuelve obs/reward.
    3. done = True → fin del periodo o pérdida del 90 % del capital.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class PortfolioEnv(gym.Env):
    """
    Entorno Gymnasium para asignación dinámica de carteras con Deep RL.

    Cada step (= un día de mercado simulado) ocurre lo siguiente:

      1. El agente recibe una OBSERVACIÓN — lo que ve para decidir.
         Combina features de mercado (RSI, MACD, momentum, etc.) con su
         propio estado de cartera (pesos actuales + retorno acumulado).
         Más detalle en _build_obs().
      2. El agente devuelve una ACCIÓN — su decisión.
         Vector de pesos no negativos, uno por activo. El entorno lo
         normaliza a suma 1 (no se admite apalancamiento ni shorts).
         
         Tres reglas que el entorno obliga a cumplir:
            a) "No negativos" = cada peso ≥ 0. No se puedes poner -0.3 en IVV. 
                Eso querría decir "vender IVV en corto" (apostar a que cae). 
                Como nuestro entorno no permite shorts, todos los pesos son positivos o cero. Si el 
                agente intenta poner -0.3, lo recortamos a 0. No hay shorts.

            b) "Suman 1" = los porcentajes deben sumar el 100 %. Si el agente devuelve 
                [0.5, 0.3, 0.2, 0.4, 0.1] (suma 1.5), el entorno divide todo entre 1.5 para que sume 1. 
                El resultado es [0.33, 0.2, 0.13, 0.27, 0.07]. Es decir, resultados en proporciones 
                normalizadas.

            c) "No apalancamiento" = como suman 1, nunca puede haber más del 100 % invertido. Apalancarse 
                sería poner el 200 % (pedir prestado). Aquí no se permite — la cartera siempre maneja 
                exactamente el dinero que tiene, ni más ni menos.            

      3. El entorno simula el día siguiente con precios reales y calcula:
         - Cómo cambia el valor de la cartera con esos pesos.
         - Coste de comisiones por rebalanceo.
         - Una RECOMPENSA compuesta:
               R_t = Sharpe_rolling - phi · MDD(t) - gamma · Turnover(t)

           Sharpe_rolling : premio por rentabilidad ajustada por riesgo de los últimos 20 días.
           MDD: castigo proporcional a la peor caída sostenida desde el último máximo.
           Turnover : castigo proporcional a cuánto cambió la cartera entre el step anterior y el actual.
           Resumen: rentabilidades - castigo(s).

      4. Si el capital cae bajo el 10 % del inicial, el episodio termina
         con reward = −1 (señal fuerte: "acabas de quebrar").

    Parameters
    ----------
    features_path : str
        Ruta al CSV de features normalizadas (índice = fecha). Las features
        son las señales de mercado precomputadas que el agente "ve".
    prices_path : str
        Ruta al CSV de precios originales (índice = fecha). Los precios se
        usan para calcular cómo evoluciona el valor de la cartera, no para
        que el agente los vea directamente.
    initial_balance : float
        Capital inicial de la cartera en USD (por defecto 10 000).
    commission : float
        Comisión proporcional por rebalanceo. 0.001 = 0.1 % del volumen
        operado en cada step.
    start_idx, end_idx : int, optional
        Índices que delimitan el subconjunto temporal del dataset. Permiten
        crear entornos separados de train (0 a split) y test (split a N) sin
        duplicar datos en memoria.
    phi : float
        Penalización por drawdown en la recompensa. Más alto = el agente se
        vuelve más conservador. Ver risk_profiles.py para perfiles
        calibrados (balanced=0.02, conservative=0.05, aggressive=0.01).
    gamma : float
        Penalización por turnover en la recompensa. Más alto = el agente se
        vuelve menos activo (rota menos la cartera). Ver risk_profiles.py.
    """

    def __init__(self, features_path, prices_path, initial_balance=10000,
                 commission=0.001, start_idx=0, end_idx=None, phi=0.02,
                 gamma=0.01, reward_type='sharpe', alpha=0.5, beta=0.5):
        """
        Constructor del entorno PortfolioEnv.

        El parametro `reward_type` selecciona la senal de recompensa positiva
        que se combina con las penalizaciones por MDD y turnover:

          - 'sharpe' (por defecto, configuracion base del TFM):
                r_t = clip(Sharpe_rolling_20 - phi*MDD - gamma*Turnover, -1, 1)
            Premia retorno ajustado por riesgo. Penaliza la volatilidad de
            forma simetrica (incluso cuando es positiva), lo que tiende a
            infraponderar activos volatiles aunque sean ganadores.

          - 'dual' (iteracion 2 del TFM):
                signal = alpha * Sharpe_rolling_20 + beta * log_return_norm
                r_t    = clip(signal - phi*MDD - gamma*Turnover, -1, 1)
            Combina retorno ajustado por riesgo (Sharpe) con retorno absoluto
            (log_return). Permite al agente premiar la captura de outliers
            ganadores volatiles que la version 'sharpe' tiende a infraponderar.
            Por defecto alpha = beta = 0.5 (mezcla equilibrada).

        Para compatibilidad hacia atras, `reward_type='sharpe'` reproduce
        exactamente el comportamiento previo y los modelos entrenados con la
        configuracion base siguen siendo validos sin cambios.
        """
        super().__init__()

        # ── 1. Cargar features 
        # Pandas no elimina ±inf con dropna() — los reemplazamos a NaN primero
        # y después dropna() limpia ambos. Esto evita que un valor infinito
        # contamine la red neuronal y produzca NaNs al propagarse.
        df_f = (pd.read_csv(features_path, index_col=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna())

        #2. Sincronizar precios con las fechas que sobrevivieron 
        # Si hubo días con feature inválida y los descartamos, también hay que
        # descartar esos mismos días en los precios para mantener alineación.
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        #  3. Recorte temporal (split train/test) 
        # Permite crear dos entornos sobre el mismo dataset sin duplicar datos:
        #   train_env = PortfolioEnv(..., start_idx=0,    end_idx=split)
        #   test_env  = PortfolioEnv(..., start_idx=split, end_idx=None)
        if end_idx is None:
            end_idx = len(df_f)

        self.df_features = (df_f.iloc[start_idx:end_idx]
                               .reset_index(drop=True)
                               .fillna(0.0)         # NaN residual → 0 (neutral)
                               .astype(np.float32))

        # ── 4. Limpieza defensiva de precios 
        # Cadena ffill → bfill → fillna(1.0) cubre tres casos:
        #   - ffill: si el precio falta hoy, propaga el último válido conocido.
        #   - bfill: si el activo no existía aún (ej. IBIT antes de 2024),
        #            usa el primer precio futuro válido para no dejar NaN.
        #   - fillna(1.0): blinda contra divisiones por cero (NUNCA 0:
        #            generaría retornos infinitos).
        self.df_prices = (df_p.iloc[start_idx:end_idx]
                             .reset_index(drop=True)
                             .replace([np.inf, -np.inf], np.nan)
                             .ffill()
                             .bfill()
                             .fillna(1.0)
                             .astype(np.float32))

        # ── 5. Verificación de integridad 
        # Si pese a las protecciones anteriores quedan NaN/inf, los neutralizamos
        # y avisamos. Seguridad: una sola feature corrupta
        # puede romper el entrenamiento entero sin dar error visible.
        n_nan = self.df_features.isnull().values.sum()
        n_inf = np.isinf(self.df_features.values).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"[AVISO] features con NaN={n_nan}, inf={n_inf} — se sustituirán por 0")
            self.df_features = self.df_features.fillna(0.0)
            self.df_features.replace([np.inf, -np.inf], 0.0, inplace=True)

        print(f"Entorno creado con {len(self.df_features)} pasos "
              f"(del índice {start_idx} al {end_idx}).")

        self.n_assets        = len(self.df_prices.columns)
        self.initial_balance = initial_balance
        self.commission      = commission

        # ── 6. Coeficientes de penalización del reward 
        # phi y gamma están calibrados para que ambas penalizaciones tengan
        # peso comparable al premio (log_return diario típico ~0.001).
        # Ver risk_profiles.py para los valores exactos de cada perfil.
        #
        # Nota histórica de calibración (no borrar — ayuda a entender los valores):
        #   phi=0.5  → MDD 20 % penalizaba 0.1, ~100× el retorno típico.
        #              El agente aprendía a "no hacer nada" para evitar el castigo.
        #   phi=0.02 → MDD 20 % penaliza 0.004, mismo orden que un día bueno.
        #              El agente equilibra premio y riesgo de forma natural.
        #
        #   gamma=0.001 → rotación completa costaba 0.002 (~2× el retorno diario).
        #                 Salía barato operar mucho → turnover excesivo y comisiones altas.
        #   gamma=0.01  → rotación completa cuesta 0.02 (~20 días de retorno positivo).
        #                 El agente solo opera si la mejora esperada compensa el coste.
        self.phi= phi
        self.gamma = gamma

        # ── 6.bis Configuracion de la senal de recompensa positiva
        # `reward_type`: 'sharpe' (configuracion base) o 'dual' (Sharpe + log_return).
        # Validacion defensiva: cualquier valor distinto cae al default 'sharpe'.
        if reward_type not in ('sharpe', 'dual'):
            print(f"[WARN] reward_type='{reward_type}' no reconocido. "
                  f"Usando 'sharpe' (configuracion base).")
            reward_type = 'sharpe'
        self.reward_type = reward_type
        self.alpha = float(alpha)  # peso de Sharpe en reward 'dual'
        self.beta  = float(beta)   # peso de log_return en reward 'dual'

        # ── 7. Buffer para el Sharpe rolling
        # Guardamos los últimos 20 retornos para calcular la media/desviación
        # de la fórmula del Sharpe. 20 días ≈ 1 mes de trading: suficiente para
        # captar el régimen actual sin arrastrar mucho ruido del pasado.
        self._ret_buffer= []
        self._sharpe_window = 20

        # ── 8. Definición del observation space (lo que el agente ve) 
        # Es la "observación AUMENTADA": features de mercado + estado interno
        # de la cartera. El extra es esencial: sin saber qué tiene en cartera
        # ni cuánto ha ganado, el agente no puede aprender comportamientos
        # condicionados ("si voy perdiendo, refugiarme en BND"). Detalle en
        # _build_obs().
        n_market_features = self.df_features.shape[1]
        n_portfolio_state = self.n_assets + 1   # n pesos + 1 retorno acumulado
        self.n_market_features = n_market_features

        # Action space: un peso por activo en [0, 1]. Long-only (sin shorts).
        # Box es el tipo de Gymnasium para vectores de números reales acotados.
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        # Observation space: vector real sin acotar. La normalización de las
        # features se hace aguas arriba (data_downloader genera z-scores).
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_market_features + n_portfolio_state,),
            dtype=np.float32
        )

    def _build_obs(self) -> np.ndarray:
        """
        Construye la observación AUMENTADA que el agente verá este step.

        ¿Qué es una "observación" en RL?
            Lo que el agente ve para decidir. Sin observación, el agente
            decide a ciegas. Es el equivalente al "estado del juego" en una
            partida de ajedrez: la posición de las piezas en cada turno.

        ¿Qué es "aumentada"?
            La observación BÁSICA serían solo las features de mercado:
            indicadores técnicos del día (RSI, MACD, momentum, volatilidad…).

            La observación AUMENTADA añade dos cosas más sobre el propio
            estado interno de la cartera:
              1. Pesos actuales — cuánto tiene el agente en cada activo.
              2. Retorno acumulado — si va ganando o perdiendo desde el inicio.

            Ejemplo de por qué hace falta:
                Imaginemos dos días con exactamente el mismo mercado (mismas
                features). En uno el agente va +25 % y en otro va −15 %.
                ¿Debe decidir lo mismo? No: perdiendo conviene
                refugiarse en activos defensivos (BND), ganando puede asumir
                más riesgo. Sin observar su propio estado, el agente sería
                ciego a esa diferencia.

            Esta es la condición conocida como "propiedad de Markov":
            el estado debe contener TODA la información relevante para
            decidir. Si el agente necesita recordar el pasado para decidir
            bien, el estado está incompleto.

        Estructura de la observación final:
            [feature_1, ..., feature_M, w_1, ..., w_N, retorno_acumulado]
                  ↑                          ↑                ↑
            mercado (M reales)        pesos por activo   posición global
                                      (N reales en [0,1])

        Returns
        -------
        np.ndarray
            Vector de dimensión (M + N + 1,) donde
                M = n_market_features
                N = n_assets
        """
        # Datos de mercado del día actual. nan_to_num es defensivo: si una
        # feature trae NaN/inf por algún motivo no detectado, lo neutralizamos
        # antes de que llegue a la red neuronal (NaN se propaga y rompe todo).
        market = np.nan_to_num(
            self.df_features.iloc[self.current_step].values,
            nan=0.0, posinf=0.0, neginf=0.0
        )

        # Retorno acumulado normalizado:
        #    0   → sin cambio respecto al inicio
        #   +1   → la cartera dobló su valor
        #   −1   → pérdida total (clip evita que números absurdos lleguen a la red)
        # Limitamos a [−1, +5] para tolerar grandes ganancias en train.
        
        # ¿Por qué hacemos clip a [-1, +5]?
        # Cota inferior -1: matemáticamente no se puede perder más del 100 % (el valor de 
        # cartera nunca es negativo en este entorno). El -1 es el "infierno" absoluto. Si 
        # por un bug llegara un valor menor (-1.5, por ejemplo), lo recortamos.
        # Cota superior +5 (= +500 %, multiplicar por 6): dejamos margen amplio porque 
        # durante el entrenamiento el agente puede encontrar trayectorias muy rentables en 
        # mercados alcistas. No es realista pero no queremos ahogar la señal si pasa. Sin 
        # embargo, valores como +50 (multiplicar por 51) sí los recortamos — son 
        # numéricamente improbables y romperían la escala que ve la red.
        # Resumen: convertimos "cuánto vale tu cartera ahora" en un número entre −1 y 
        # +5 fácil de digerir por la red. Si la realidad va más allá de esos topes, la 
        # achatamos para no distorsionar al agente.
        
        #todo: no podríamos más allá del 500%? tal vez al 900%?
        
        cumulative_return = np.clip(
            (self.portfolio_value - self.initial_balance) / (self.initial_balance + 1e-8),
            -1.0, 5.0
        )

        # Estado de cartera = pesos actuales + retorno acumulado, en un solo vector.
        portfolio_state = np.append(
            self.weights.astype(np.float32), np.float32(cumulative_return)
        )

        # Observación final: concatenamos mercado + estado de cartera.
        # Este vector es la "fotografía completa" del momento que la red
        # neuronal del PPO recibe como entrada para decidir la acción.
        return np.concatenate([market, portfolio_state]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial para empezar un nuevo episodio.

        Es lo equivalente a "empezar una nueva partida": el agente vuelve al
        día 0, con el capital inicial completo, cartera repartida a partes
        iguales entre todos los activos (estado neutro), y los buffers
        internos (Sharpe rolling, marca de agua del MDD) limpios.

        Gymnasium llama a reset() automáticamente al inicio de cada episodio
        durante el entrenamiento. PPO entrena viendo decenas de miles de
        episodios para que la política converja.

        Parameters
        ----------
        seed : int, optional
            Semilla para el generador aleatorio (lo gestiona la clase base).
            Útil para reproducibilidad si se introduce aleatoriedad en el
            entorno (no es el caso actual).
        options : dict, optional
            Opciones adicionales según el contrato de Gymnasium. No usadas.

        Returns
        -------
        tuple[np.ndarray, dict]
            (observación inicial, info vacío). El segundo elemento existe
            por contrato Gymnasium pero aquí no contiene nada relevante.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        # Cartera inicial equilibrada: 1/N en cada activo. Es el punto de
        # partida más neutral; el agente decidirá cómo redistribuir.
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._ret_buffer = []

        # Máximo del valor de cartera: la usaremos para
        # calcular el Maximum Drawdown ("¿cuánto he caído desde mi mejor
        # momento?"). Importante resetearla AQUÍ y no en step(), porque
        # en cada episodio nuevo el "mejor momento" es el inicial, porque contiene el
        # mejor dato alcanzado en el episodio anterior.
        self.max_portfolio_value = self.initial_balance

        return self._build_obs(), {}

    def step(self, action):
        """
        Avanza un día (un step) del simulador.

        Es el método más importante del entorno: en él se materializa toda
        la lógica de gestión de cartera (rebalanceo, comisiones, valor
        actualizado) y se calcula la recompensa que guía el aprendizaje.

        Esquema de lo que ocurre cada step:
          1. Tomar referencia del valor de cartera ANTES de actuar.
          2. Limpiar y normalizar la acción del agente (pesos a suma 1).
          3. Cobrar comisiones por el rebalanceo respecto al step anterior.
          4. Avanzar un día y aplicar los precios de "mañana" a la cartera.
          5. Calcular la recompensa: Sharpe rolling − φ·MDD − γ·Turnover.
          6. Comprobar condición de quiebra (capital < 10 % del inicial).
            todo: poner como condicion el 10% del capital conseguido?? U otras condiciones 
            mas favorables para no llegar a esa quiebra absoluta?
          7. Construir la nueva observación y devolver todo a SB3/PPO.

        Parameters
        ----------
        action : np.ndarray
            Vector de pesos deseados por activo. SB3 puede mandar valores
            ligeramente fuera de [0, 1] (por la distribución gaussiana de
            la política PPO); el step los limpia antes de usarlos.

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict]
            (observación, recompensa, terminado, truncado, info).
            'truncado' siempre es False — no usamos límite por timeout.
            'info' lleva: value, drawdown, weights, turnover (útil para
            métricas y reporting fuera del entorno).
        """
        # ── 1. Estado previo: precios de hoy y valor de referencia 
        prices_today = self.df_prices.iloc[self.current_step].values
        # Capturamos el valor de cartera ANTES de actuar para luego poder
        # calcular el retorno del step (cuánto subió/bajó la cartera).
        baseline_value = max(self.portfolio_value, 1e-6)

        # ── 2. Limpiar y normalizar la acción del agente 
        # PPO emplea una política gaussiana que puede generar valores
        # ligeramente fuera del [0, 1] declarado en action_space → clip.
        # Tras eso, normalizamos para que los pesos sumen 1 (cartera completa
        # invertida; sin liquidez sobrante ni apalancamiento).
        action_clipped = np.clip(action, 0.0, 1.0)
        total_weight = np.sum(action_clipped)
        if total_weight < 1e-3:
            # Caso raro/especial: el agente devolvió todos los pesos casi a 0.
            # Caemos al equilibrado 1/N como fallback razonable.
            # Es decir: si el agente está tan perdido que dice "no quiero invertir en nada" 
            # (todos los pesos casi a cero), lo metemos en una cartera equilibrada de 
            # oficio. Equivale a decir: "tranquilo, divide igual entre todos hasta que 
            # vuelvas a tener una opinión clara".
            new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        else:
            new_weights = action_clipped / total_weight

        # ── 3. Comisiones por rebalanceo 
        # diff_weights mide cuánto hay que mover entre activos. Esa cantidad
        # × valor cartera × comisión = coste total. Esto desincentiva
        # rebalanceos erráticos y simula los costes reales del broker.
        diff_weights = np.abs(new_weights - self.weights)
        cost = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost

        # ── 4. Progresión temporal 
        # Si ya estamos en el último día, terminamos sin calcular más.
        if self.current_step >= len(self.df_features) - 1:
            done = True
            return (self._build_obs(), 0.0, done, False,
                    {"value": self.portfolio_value, "drawdown": 0})

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1

        # ── 5. Evolución del valor con los precios de mañana 
        prices_tomorrow = self.df_prices.iloc[self.current_step].values

        # np.fmax ignora NaN, np.maximum los propaga. Necesario porque IBIT
        # y ETHA pueden tener huecos antes de su fecha de listado (2024).
        prices_today_safe    = np.fmax(prices_today,    1e-6)
        prices_tomorrow_safe = np.fmax(prices_tomorrow, 1e-6)

        # ── Recorte defensivo de retornos diarios 
        # Retorno diario = precio_hoy / precio_ayer (cuánto multiplica el
        # precio de hoy al de ayer):
        #   retorno = 1.02 → +2 %
        #   retorno = 0.98 → -2 %
        #   retorno = 0.5  → -50 % (un día catastrófico real)
        #   retorno = 2.0  → +100 % (un día imposible en activos cotizados)
        #
        # Capamos cualquier valor fuera de [0.5, 2.0] para protegernos de
        # DATOS CORRUPTOS, no de eventos reales de mercado. La distinción
        # importa: ningún activo del universo aceptado por nuestro screener
        # (S&P 500 + IBIT + ETHA, sea cual sea la composición concreta del
        # día — pueden ser 9, 15 o 20 activos según el screener) se mueve un
        # ±100 % en un día, ni siquiera en el peor crash histórico.
        #
        # Lo que SÍ produce ratios absurdos en los datos crudos:
        #   - Splits accionariales (2-por-1 → precio cae 50 % por estructura,
        #     no por mercado). Yahoo Finance suele ajustar, pero a veces se
        #     cuela un día sin ajustar.
        #   - Errores puntuales del feed de datos.
        #   - Fechas mal alineadas en el CSV.
        #
        # El daño que evitamos: si IBIT apareciese con retorno 5.0 por un
        # error y el agente tiene 30 % en IBIT, la cartera virtualmente se
        # multiplicaría por 1.6 ese día. El agente "aprendería a amar" IBIT
        # por una razón ficticia. El recorte hace que un dato basura cause
        # como mucho un movimiento creíble (+100 %), no destruye el episodio.
        asset_returns = np.clip(prices_tomorrow_safe / prices_today_safe, 0.5, 2.0)

        # Valor de cartera mañana = Σ (capital_asignado_a_cada_activo × retorno_del_activo)
        new_value = np.sum((self.portfolio_value * new_weights) * asset_returns)
        # Clip final defensivo: evita overflow → NaN en operaciones posteriores.
        self.portfolio_value = float(np.clip(new_value, 1e-6, 1e9))

        self.weights = new_weights

        # ── 6. REWARD SHAPING — el corazón académico del entorno 
        # La forma de la recompensa es lo que determina qué aprende el agente.
        # Aquí construimos una recompensa COMPUESTA con tres ingredientes:

        # 6.A — Retorno logarítmico del step
        # log(V_nuevo / V_viejo). Usamos log en vez de retorno simple porque
        # los retornos logarítmicos son aditivos en el tiempo (suman) y
        # estables numéricamente (acotados implícitamente).
        log_return = float(np.log(self.portfolio_value / baseline_value + 1e-8))

        # 6.B — Sharpe rolling: la señal de calidad principal
        # En lugar de premiar retorno bruto (que incentiva asumir riesgo
        # ilimitado), premiamos retorno AJUSTADO por riesgo. Para eso
        # mantenemos un buffer de los últimos 20 días y calculamos el Sharpe
        # de esa ventana. 20 días ≈ 1 mes: suficiente para captar la dinámica
        # actual del mercado sin arrastrar régimenes pasados ya superados.
        self._ret_buffer.append(log_return)
        if len(self._ret_buffer) > self._sharpe_window:
            self._ret_buffer.pop(0)

        if len(self._ret_buffer) >= 5:
            rets = np.array(self._ret_buffer)
            # Sharpe anualizado: media/std × √252.
            # Lo normalizamos dividiendo por 3 y recortando a [−1, 1] para
            # que entre en la misma escala que las penalizaciones — así
            # ningún término del reward domina al resto.
            rolling_sharpe = float(
                np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
            )
            sharpe_norm = float(np.clip(rolling_sharpe / 3.0, -1.0, 1.0))
        else:
            # Warm-up: en los primeros 5 steps no hay suficiente historia
            # para un Sharpe fiable. Usamos directamente el log_return como
            # señal aproximada hasta llenar el buffer.
            sharpe_norm = float(np.clip(log_return * 100, -1.0, 1.0))

        # 6.C — Maximum Drawdown actual (penalización)
        # Drawdown = (mejor valor visto − valor actual) / mejor valor visto.
        # Mide cuánto se ha caído desde el máximo histórico ("¿qué dolor
        # psicológico produciría esta racha en un inversor real?").
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = float(np.clip(
            (self.max_portfolio_value - self.portfolio_value)
            / (self.max_portfolio_value + 1e-8),
            0.0, 1.0
        ))

        # 6.D — Composición final del reward:
        #
        # Configuracion 'sharpe' (base):
        #   R_t = Sharpe_rolling − φ · MDD(t) − γ · Turnover(t)
        #   Premia retorno ajustado por riesgo. Penaliza la volatilidad de
        #   forma simetrica.
        #
        # Configuracion 'dual' (iteracion 2):
        #   signal = α · Sharpe_rolling + β · log_return_norm
        #   R_t    = signal − φ · MDD(t) − γ · Turnover(t)
        #   Combina retorno ajustado por riesgo con retorno absoluto. El
        #   componente de log_return premia la captura de outliers ganadores
        #   (que en 'sharpe' quedan infraponderados al penalizarse su
        #   volatilidad incluso cuando es positiva).
        #
        # En ambos casos, las penalizaciones por MDD y turnover son identicas:
        #   - φ · MDD       : "no caigas mucho desde tu mejor momento".
        #   - γ · Turnover  : "no operes por operar; cada movimiento cuesta".
        turnover         = float(np.sum(diff_weights))
        risk_penalty     = self.phi   * current_drawdown
        turnover_penalty = self.gamma * turnover

        if self.reward_type == 'dual':
            # log_return_norm: misma escala que sharpe_norm (acotado en [-1, 1])
            # para que el alpha/beta sean ponderaciones interpretables sobre
            # senales del mismo orden de magnitud.
            log_return_norm = float(np.clip(log_return * 100.0, -1.0, 1.0))
            reward_signal = self.alpha * sharpe_norm + self.beta * log_return_norm
        else:  # 'sharpe' (default, configuracion base del TFM)
            reward_signal = sharpe_norm

        reward = float(np.clip(
            reward_signal - risk_penalty - turnover_penalty, -1.0, 1.0
        ))

        # ── 7. Condición de quiebra 
        # Si el agente ha perdido el 90 % del capital, el episodio termina con
        # señal fuerte de penalización (−1). Esto enseña al agente que perder
        # tanto es un fallo catastrófico que debe evitar a toda costa.
        if self.portfolio_value < (self.initial_balance * 0.1):
            done = True
            reward = -1.0

        # info: información auxiliar para reporting fuera del entorno.
        # No se usa para entrenar — Stable-Baselines3 solo lee la reward.
        info = {
            "value":    self.portfolio_value,
            "drawdown": current_drawdown,
            "weights":  self.weights,
            "turnover": turnover,
        }

        return self._build_obs(), float(reward), done, False, info


# ---------------------------------------------------------------------------
# Compatibilidad hacia atrás: alias del atributo renombrado
# ---------------------------------------------------------------------------
# El atributo se llamaba originalmente `df_precios` (en español) y se renombró
# a `df_prices` por consistencia con el resto del código (mezclar idiomas
# dificultaba la lectura). Para no romper código externo o scripts antiguos
# que aún acceden a `env.df_precios`, monkey-patcheamos el __init__ para
# crear el alias automáticamente. Cuando se confirme que ningún módulo usa
# el nombre antiguo, este bloque puede eliminarse limpiamente.
_original_init = PortfolioEnv.__init__

def _patched_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.df_precios = self.df_prices  # alias retrocompatible

PortfolioEnv.__init__ = _patched_init
