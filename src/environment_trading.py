import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):

    def __init__(self, features_path, prices_path, initial_balance=10000, commission=0.001, start_idx=0, end_idx=None):
        super().__init__()
        
        #Cargar datos básicos y eliminar NaNs
        df_f = pd.read_csv(features_path, index_col=0).dropna()
        
        #2. Sincronizar precios con las features que han quedado vivas
        df_p = pd.read_csv(prices_path, index_col=0).loc[df_f.index]

        #3. Establecer el límite final si no se ha pasado
        if end_idx is None:
            end_idx = len(df_f)

        #conjuntos de Train o Test
        self.df_features = df_f.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
        self.df_precios = df_p.iloc[start_idx:end_idx].reset_index(drop=True).astype(np.float32)
                
        print(f"Entorno creado con {len(self.df_features)} pasos (del índice {start_idx} al {end_idx}).")

        self.n_assets = len(self.df_precios.columns)
        self.initial_balance = initial_balance
        self.commission = commission 

        #Gymnasium
        ### 2. Gestión de la Señal-Ruido mediante Espacios Continuos
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
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.weights = np.zeros(self.n_assets) 
        
        obs = self.df_features.iloc[self.current_step].values
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
        #"Panic? Not for me"
        new_weights = action / (np.sum(action) + 1e-6)
        
        # 3. Gestión de Costes:coste de mover dinero, lo que evita rebalanceos erráticos por ruido
        diff_weights = np.abs(new_weights - self.weights)
        cost = np.sum(diff_weights) * self.portfolio_value * self.commission
        self.portfolio_value -= cost
        
        # 4. Progresión Temporal
        if self.current_step >= len(self.df_features) - 1:
            done = True
            obs = self.df_features.iloc[self.current_step].values
            return obs.astype(np.float32), 0.0, done, False, {"value": self.portfolio_value, "drawdown": 0}

        self.current_step += 1
        done = self.current_step >= len(self.df_features) - 1
        
        # 5. Evolución del Mercado (Impacto de los precios de mañana)
        precios_manana = self.df_precios.iloc[self.current_step].values
        retornos_activos = precios_manana / precios_hoy
        self.portfolio_value = np.sum((self.portfolio_value * new_weights) * retornos_activos)
        self.portfolio_value = max(self.portfolio_value, 1e-6) # Protección contra valores negativos
        
        self.weights = new_weights 

        # --- REWARD SHAPING: EL CORAZÓN DE LA ESTRATEGIA ---
        
        # A. Retorno Logarítmico (La base del beneficio)
        log_return = float(np.log(self.portfolio_value / valor_base_paso + 1e-8))

        # B. Cálculo de la "Marca de Agua" y Drawdown (Gestión de Riesgo Extremo)
        # Necesitas inicializar self.max_portfolio_value en el método reset()
        if not hasattr(self, 'max_portfolio_value'): 
            self.max_portfolio_value = self.initial_balance
            
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value

        # C. Penalización por Riesgo (Factor de sensibilidad al Drawdown)
        # Un phi=0.5 es un equilibrio agresivo/conservador.
        phi = 0.5 
        risk_penalty = phi * current_drawdown

        # Recompensa Final: $$R_t = \text{log\_return} - \text{risk\_penalty}$$
        reward = log_return - risk_penalty

        # 7. Condición de quiebra (Game Over)
        if self.portfolio_value < (self.initial_balance * 0.1):
            done = True
            reward = -10.0 # Penalización masiva por colapso de cartera
        
        obs = self.df_features.iloc[self.current_step].values
        
        # Enviamos el drawdown en la info para monitorizarlo en Tensorboard
        info = {
            "value": self.portfolio_value,
            "drawdown": current_drawdown,
            "weights": self.weights
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