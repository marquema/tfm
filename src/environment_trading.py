import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):

    def __init__(self, features_path, prices_path, initial_balance=10000, commission=0.001,
                 start_idx=0, end_idx=None, phi=0.5, gamma=0.001):
        super().__init__()

        # Cargar datos: primero reemplazar ±inf (dropna NO los elimina) y luego NaNs
        df_f = (pd.read_csv(features_path, index_col=0)
                  .replace([np.inf, -np.inf], np.nan)
                  .dropna())

        # Sincronizar precios con las features que han quedado vivas tras el dropna
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        # Establecer el límite final del subconjunto (train o test)
        if end_idx is None:
            end_idx = len(df_f)

        # Subconjunto temporal: permite crear entornos de train (0..split) y test (split..N)
        # fillna(0) como segunda línea de defensa ante NaNs residuales introducidos en el slice
        self.df_features = (df_f.iloc[start_idx:end_idx]
                               .reset_index(drop=True)
                               .fillna(0.0)
                               .astype(np.float32))
        # Para precios: forward-fill para propagar el último precio válido,
        # luego backward-fill por si hay NaN al inicio, y finalmente 0 como último recurso.
        # Nunca usar 0 directamente porque genera retornos infinitos (división por cero).
        self.df_precios  = (df_p.iloc[start_idx:end_idx]
                               .reset_index(drop=True)
                               .replace([np.inf, -np.inf], np.nan)
                               .ffill()
                               .bfill()
                               .fillna(1.0)
                               .astype(np.float32))

        # Verificación de integridad: avisar si quedan NaN/inf tras el saneamiento
        n_nan = self.df_features.isnull().values.sum()
        n_inf = np.isinf(self.df_features.values).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"[AVISO] features con NaN={n_nan}, inf={n_inf} — se sustituirán por 0")
            self.df_features = self.df_features.fillna(0.0)
            self.df_features.replace([np.inf, -np.inf], 0.0, inplace=True)

        print(f"Entorno creado con {len(self.df_features)} pasos (del índice {start_idx} al {end_idx}).")

        self.n_assets        = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission      = commission

        # Coeficientes de la función de recompensa compuesta:
        #   R_t = r_p(t) - [phi·MDD(t) + gamma·Turnover(t)]
        # phi:   penalización por Maximum Drawdown (riesgo de pérdida extrema)
        # gamma: penalización por rotación excesiva (estabilidad de la estrategia)
        self.phi   = phi
        self.gamma = gamma

        # Espacios de Gymnasium: acciones continuas en [0,1] para cada activo
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df_features.shape[1],), dtype=np.float32
        )

    #def __init__(self, features_path, prices_path, initial_balance=10000, commission=0.001):
    def __init2__(self, features_path, prices_path, initial_balance=10000, commission=0.001, start_idx=0, end_idx=None):
        super().__init__()
        # Carga de datos
        #self.df_features = pd.read_csv(features_path, index_col=0)
        #self.df_precios = pd.read_csv(prices_path, index_col=0)

        self.df_features = pd.read_csv(features_path, index_col=0).dropna()
        self.df_precios = pd.read_csv(prices_path, index_col=0).loc[self.df_features.index]

        if end_idx is None:
            end_idx = len(self.df_features)

        
        # 1. Eliminar cualquier fila que tenga algún NaN (especialmente al inicio)
        # Esto sincroniza precios y features
        #self.df_features = self.df_features.dropna().astype(np.float32)
        self.df_features.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        
        # 2. Asegurarnos de que los precios tengan las mismas fechas que las features
        #self.df_precios = self.df_precios.loc[self.df_features.index].astype(np.float32)
        self.df_precios = self.df_precios.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        
        # 3. Resetear índices para que coincidan perfectamente (0, 1, 2...)
        self.df_features = self.df_features.reset_index(drop=True)
        self.df_precios = self.df_precios.reset_index(drop=True)        
        
        self.n_assets = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission = commission # 0.1% de comisión por operación

        # Espacios de Gymnasium
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df_features.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = 0
        self.balance         = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights         = np.zeros(self.n_assets)

        # Inicializar la marca de agua máxima aquí, no en step(),
        # para garantizar que se resetea correctamente en cada episodio
        self.max_portfolio_value = self.initial_balance

        obs = np.nan_to_num(
            self.df_features.iloc[self.current_step].values,
            nan=0.0, posinf=0.0, neginf=0.0
        )
        return obs.astype(np.float32), {}

    def step(self, action):
        """
            agente minimiza el Maximum Drawdown y optimiza el Sharpe
            $$R_t = r_p(t) - \phi \cdot \text{Drawdown}(t)$$        
        """
        # todo: personalizar el factor $\phi$ para que sea dinámico según la correlación detectada en el mercado
        # 1. Precios y estado previo
        precios_hoy = self.df_precios.iloc[self.current_step].values
        # Referencia de valor para calcular el retorno del paso
        valor_base_paso = max(self.portfolio_value, 1e-6)

        # 2. Normalizar pesos (sumatoria = 1)
        # Primero recortar al espacio declarado [0, 1] — SB3 usa Gaussian que puede salirse
        action_clipped = np.clip(action, 0.0, 1.0)
        peso_total = np.sum(action_clipped)
        if peso_total < 1e-3:
            # Si el agente intenta asignar ~0 a todo, distribuir uniforme (evita /0 y pesos explosivos)
            new_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        else:
            new_weights = action_clipped / peso_total
        
        # 3. Gestión de Costes:coste de mover dinero, lo que evita rebalanceos erráticos por ruido
        diff_weights = np.abs(new_weights - self.weights)
        cost = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost
        
        # 4. Progresión Temporal
        if self.current_step >= len(self.df_features) - 1:
            done = True
            obs = np.nan_to_num(
                self.df_features.iloc[self.current_step].values,
                nan=0.0, posinf=0.0, neginf=0.0
            )
            return obs.astype(np.float32), 0.0, done, False, {"value": self.portfolio_value, "drawdown": 0}

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        # 5. Evolución del Mercado (Impacto de los precios de mañana)
        precios_manana = self.df_precios.iloc[self.current_step].values

        # np.fmax ignora NaN (a diferencia de np.maximum que lo propaga)
        # Necesario porque IBIT no existía antes de 2024 y puede quedar NaN residual
        precios_hoy_safe    = np.fmax(precios_hoy,    1e-6)
        precios_manana_safe = np.fmax(precios_manana, 1e-6)

        # Recortar retornos diarios al rango [0.5, 2.0] (−50% / +100%)
        # para evitar que un dato erróneo en el CSV destruya el portfolio_value
        retornos_activos = np.clip(precios_manana_safe / precios_hoy_safe, 0.5, 2.0)

        nuevo_valor = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        # Recortar a un rango finito para evitar overflow → NaN en operaciones posteriores
        self.portfolio_value = float(np.clip(nuevo_valor, 1e-6, 1e9))

        self.weights = new_weights

        # --- REWARD SHAPING: EL CORAZÓN DE LA ESTRATEGIA ---

        # A. Retorno Logarítmico (La base del beneficio)
        log_return = float(np.log(self.portfolio_value / valor_base_paso + 1e-8))

        # B. Cálculo de la "Marca de Agua" y Maximum Drawdown (MDD)
        # max_portfolio_value se inicializa en reset() para garantizar coherencia entre episodios
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        # np.clip evita NaN cuando portfolio_value == max_portfolio_value == inf
        current_drawdown = float(np.clip(
            (self.max_portfolio_value - self.portfolio_value) / (self.max_portfolio_value + 1e-8),
            0.0, 1.0
        ))

        # C. Función de recompensa compuesta (según la fórmula del TFM):
        #   R_t = r_p(t) - [α·Riesgo(t) + γ·Turnover(t)]
        # Los costes de transacción β·Costes(t) ya están implícitos en r_p(t)
        # a través de la deducción directa sobre portfolio_value (línea anterior).

        # Penalización por riesgo: desincentiva estrategias con grandes caídas
        risk_penalty = self.phi * current_drawdown

        # Penalización por rotación: desincentiva rebalanceos excesivos
        # turnover = Σ|w_nuevo - w_anterior| (ya calculado como diff_weights)
        turnover         = float(np.sum(diff_weights))
        turnover_penalty = self.gamma * turnover

        # Recortar la recompensa a un rango acotado para evitar gradientes explosivos
        reward = float(np.clip(log_return - risk_penalty - turnover_penalty, -10.0, 10.0))

        # 7. Condición de quiebra (Game Over)
        if self.portfolio_value < (self.initial_balance * 0.1):
            done = True
            reward = -10.0 # Penalización masiva por colapso de cartera
        
        obs = np.nan_to_num(
            self.df_features.iloc[self.current_step].values,
            nan=0.0, posinf=0.0, neginf=0.0
        )

        # Enviamos el drawdown en la info para monitorizarlo en Tensorboard
        # Incluir turnover en el info para monitorización en TensorBoard
        info = {
            "value":    self.portfolio_value,
            "drawdown": current_drawdown,
            "weights":  self.weights,
            "turnover": turnover,
        }
        
        return obs.astype(np.float32), float(reward), done, False, info


    def step_2(self, action):
            #1. Precios y estado previo
            precios_hoy = self.df_precios.iloc[self.current_step].values
            valor_antes_movimiento = max(self.portfolio_value, 1e-6)

            #2. Normalizar pesos (deben sumar 1)
            new_weights = action / (np.sum(action) + 1e-6)#panico en el mercado? Not for me
            #3. Costes - comisiones
            diff_weights = np.abs(new_weights - self.weights)
            cost = np.sum(diff_weights) * self.portfolio_value * self.commission
            self.portfolio_value -= cost
                        
            # Si ya estamos en el último registro disponible, no hay un "mañana"
            if self.current_step >= len(self.df_features) - 1:
                done = True
                obs = self.df_features.iloc[self.current_step].values
                return obs.astype(np.float32), 0.0, done, False, {"value": self.portfolio_value}
                                
            self.current_step += 1
            done = self.current_step >= len(self.df_features) - 1
            
            #Evol del mercado (de hoy a mañana)
            try:
                precios_manana = self.df_precios.iloc[self.current_step].values
            except Exception as e:
                print(e.args)
            retornos_activos = precios_manana / precios_hoy
            self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
            self.portfolio_value = max(self.portfolio_value, 1e-6) # Protección
            
            self.weights = new_weights 

            # 6. Recompensa (Log Returns)
            reward = float(np.log(self.portfolio_value / valor_antes_movimiento + 1e-8))

            # 7. Condición de quiebra (Game Over)
            if self.portfolio_value < (self.initial_balance * 0.1):
                done = True
                reward = -10.0 
            
            obs = self.df_features.iloc[self.current_step].values
            return obs.astype(np.float32), reward, done, False, {"value": self.portfolio_value}

    def step_simplex(self, action):
        # 1. Normalizar pesos (deben sumar 1)
        new_weights = action / (np.sum(action) + 1e-6)
        
        # 2. Precios
        precios_hoy = self.df_precios.iloc[self.current_step].values
        
        # 3. Calcular costes de transacción
        # Diferencia entre lo que tenemos y lo que queremos (rebalanceo)
        diff_weights = np.abs(new_weights - self.weights)
        cost = np.sum(diff_weights) * self.portfolio_value * self.commission
        
        # 4. Actualizar valor tras costes
        self.portfolio_value -= cost
        valor_antes_movimiento = self.portfolio_value
        
        # 5. Avanzar el tiempo
        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        # 6. Calcular nuevo valor basado en el cambio de precio de los activos
        precios_manana = self.df_precios.iloc[self.current_step].values
        retornos_activos = precios_manana / precios_hoy
        self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        
        self.weights = new_weights # Guardamos los nuevos pesos

        # 6.1 Aseguramos que el valor no sea negativo ni cero absoluto
        self.portfolio_value = max(self.portfolio_value, 1e-6)
        valor_antes_movimiento = max(valor_antes_movimiento, 1e-6)

        # 7. Cálculo de la recompensa con protección (épsilon)
        # Añadimos un pequeño 1e-8 dentro del log por seguridad
        reward = np.log(self.portfolio_value / valor_antes_movimiento + 1e-8)

        ## 7. RECOMPENSA (Log Returns)
        ## Usamos LaTeX para la fórmula: $$R_t = \ln\left(\frac{V_t}{V_{t-1}}\right)$$
        ## reward = np.log(self.portfolio_value / valor_antes_movimiento)

        # 8. Terminación por quiebra (Game Over)
        # Si la IA pierde el 90% del dinero, paramos el episodio
        if self.portfolio_value < (self.initial_balance * 0.1):
            done = True
            reward = -10 # Penalización fuerte por arruinarse        
        
        obs = self.df_features.iloc[self.current_step].values
        return obs.astype(np.float32), float(reward), done, False, {"value": self.portfolio_value}