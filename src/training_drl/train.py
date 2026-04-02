import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.training_drl.environment_trading import PortfolioEnv
from src.training_drl.training_analysis import entrenar_academico, walk_forward_validation

def entrenar_modelo(total_timesteps=100000):
    print("--- Iniciando Entrenamiento de la IA ---")
    
    # 1. Crear el entorno
    env = PortfolioEnv('data/normalized_features.csv', 'data/original_prices.csv')    
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
    df_f = pd.read_csv('data/normalized_features.csv')
    split_idx = int(len(df_f) * split_pct)
    
    # 2. Crear entorno de ENTRENAMIENTO (80% inicial)
    # Pasaremos un parámetro 'end_idx' a tu PortfolioEnv para recortar los datos
    train_env = PortfolioEnv('data/normalized_features.csv', 'data/original_prices.csv', end_idx=split_idx)
    
    # 3. Crear entorno de VALIDACIÓN (20% final)
    eval_env = PortfolioEnv('data/normalized_features.csv', 'data/original_prices.csv', start_idx=split_idx)
    
    # 4. Configurar el Callback de Evaluación
    # Evaluará cada 5000 pasos y guardará el mejor modelo en 'models/best_model'
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best_model/',
        log_path='./logs/results/',
        eval_freq=10000,       # evaluar cada 10k pasos (antes 5k era demasiado frecuente)
        n_eval_episodes=3,     # promedio de 3 episodios para estabilizar la métrica
        deterministic=True,
        render=False
    )

    # 5. Inicializar PPO con arquitectura y hiperparámetros optimizados para portfolios
    #
    # Red más profunda (256-256): con 171 features el default (64-64) es demasiado pequeño
    # para capturar las relaciones entre indicadores técnicos, correlaciones y regímenes.
    #
    # n_steps=1024: horizonte de rollout reducido (default 2048) — el episodio de train
    # tiene ~2460 pasos, con 2048 el agente ve solo ~1 episodio por update, lo que ralentiza
    # la convergencia. Con 1024 ve ~2 episodios y actualiza más frecuentemente.
    #
    # ent_coef=0.01: coeficiente de entropía — incentiva exploración durante el entrenamiento,
    # evita que el agente colapse a "siempre igual peso" desde el inicio.
    #
    # max_grad_norm=0.5: recorte de gradientes — previene actualizaciones explosivas si
    # algún batch contiene transiciones con rewards muy distintos.
    #
    # vf_coef=0.5: peso del value function loss — equilibra policy loss y value loss.
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # 6. ¡Lanzar los 100.000 pasos!
    print(f"Iniciando entrenamiento de {total_timesteps} pasos...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    
    # 7. Guardar el modelo final (aunque el 'best_model' suele ser el mejor)
    model.save("models/ppo_final_100k")
    print("Entrenamiento finalizado.")