import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.environment_trading import PortfolioEnv

def entrenar_modelo(total_timesteps=100000):
    print("--- Iniciando Entrenamiento de la IA ---")
    
    # 1. Crear el entorno
    env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv')    
    #Verificar si hay NaNs antes de empezar
    if env.df_features.isnull().values.any():
        print("ERROR: ¡Todavía hay NaNs en las features!")
        return    
    
    #2. Configurar el algoritmo PPO
    #MlpPolicy: Red neuronal estándar (Multi-layer Perceptron): barandilla de seguridad
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    
    #3. Aprender
    print(f"Entrenando por {total_timesteps} pasos...")
    model.learn(total_timesteps=total_timesteps)
    
    #4. Guardar el "cerebro"
    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_portfolio_manager_50000")
    print("Modelo guardado en models/ppo_portfolio_manager_50000.zip")
    
    return "Entrenamiento completado"




def entrenar_con_validacion(total_timesteps=100000, split_pct=0.8):
    print("--- Configurando Entrenamiento con Validación Temporal ---")
    
    # 1. Cargar datos para calcular el punto de corte
    df_f = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)
    
    # 2. Crear entorno de ENTRENAMIENTO (80% inicial)
    # Pasaremos un parámetro 'end_idx' a tu PortfolioEnv para recortar los datos
    train_env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv', end_idx=split_idx)
    
    # 3. Crear entorno de VALIDACIÓN (20% final)
    eval_env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv', start_idx=split_idx)
    
    # 4. Configurar el Callback de Evaluación
    # Evaluará cada 5000 pasos y guardará el mejor modelo en 'models/best_model'
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_model/',
        log_path='./logs/results/', 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )

    # 5. Inicializar PPO
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./logs/")
    
    # 6. ¡Lanzar los 100.000 pasos!
    print(f"Iniciando entrenamiento de {total_timesteps} pasos...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # 7. Guardar el modelo final (aunque el 'best_model' suele ser el mejor)
    model.save("models/ppo_final_100k")
    print("✅ Entrenamiento finalizado.")