import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.environment_trading import PortfolioEnv

# Hiperparámetros compartidos ajustados para el espacio de observación de 81 features.
# Con más señal de entrada, la red necesita más capacidad y actualizaciones más
# conservadoras para no sobreajustar durante las primeras iteraciones.
PPO_KWARGS = dict(
    policy       = "MlpPolicy",
    clip_range   = 0.2,        # Recorte de Schulman: limita el cambio de política por update
    learning_rate= 3e-4,       # Adam lr estándar; se puede reducir a 1e-4 si el value loss oscila
    n_steps      = 2048,       # Pasos por rollout antes de actualizar (↑ = estimaciones más estables)
    batch_size   = 256,        # Mini-batch: 256 para 81 features da gradientes más estables que 64
    n_epochs     = 10,         # Épocas por update; 10 es el default de SB3
    policy_kwargs = dict(
        net_arch = [256, 256]  # Red más profunda que el default [64,64] para procesar 81 features
    ),
    verbose          = 1,
    tensorboard_log  = "./logs/",
)


def _make_envs(split_pct=0.8, reward_mode='sharpe_drawdown'):
    """Crea entornos de train y eval con el split temporal indicado."""
    df_f      = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)

    train_env = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        end_idx=split_idx, reward_mode=reward_mode
    )
    eval_env = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        start_idx=split_idx, reward_mode=reward_mode
    )
    return train_env, eval_env, split_idx


def entrenar_con_validacion(total_timesteps=100000, split_pct=0.8):
    """
    Entrenamiento principal con reward_mode='sharpe_drawdown'.
    Guarda el mejor modelo (por reward de validación) en models/best_model/.
    """
    print("--- Entrenamiento principal: sharpe_drawdown ---")
    train_env, eval_env, _ = _make_envs(split_pct, reward_mode='sharpe_drawdown')

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = './models/best_model/',
        log_path             = './logs/results/',
        eval_freq            = 5000,
        deterministic        = True,
        render               = False,
    )

    model = PPO(env=train_env, **PPO_KWARGS)
    print(f"Entrenando {total_timesteps} pasos | obs_dim={train_env.observation_space.shape[0]}")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("models/ppo_final_100k")
    print("Modelo guardado: models/ppo_final_100k.zip")
    return "Entrenamiento completado"


def entrenar_ablacion(total_timesteps=100000, split_pct=0.8):
    """
    Entrena el modelo baseline con reward_mode='log_return' (sin penalización de drawdown).
    Guarda en models/ablation_log_return/ para comparar contra el modelo principal.

    El ablation study responde: ¿cuánto aporta el risk-shaping (φ·drawdown)?
    Se compara en ver_resultados.py con ejecutar_backtest_comparativo().
    """
    print("--- Ablation study: log_return (sin penalización de drawdown) ---")
    train_env, eval_env, _ = _make_envs(split_pct, reward_mode='log_return')

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = './models/ablation_log_return/',
        log_path             = './logs/ablation/',
        eval_freq            = 5000,
        deterministic        = True,
        render               = False,
    )

    model = PPO(env=train_env, **PPO_KWARGS)
    print(f"Entrenando {total_timesteps} pasos | reward_mode=log_return")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("models/ablation_log_return/model_final")
    print("Modelo ablation guardado: models/ablation_log_return/")
    return "Ablation completado"


def entrenar_modelo(total_timesteps=100000):
    """Entrenamiento simple sin validación (útil para pruebas rápidas)."""
    print("--- Entrenamiento rápido sin validación ---")
    env = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        reward_mode='sharpe_drawdown'
    )
    model = PPO(env=env, **PPO_KWARGS)
    model.learn(total_timesteps=total_timesteps)
    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_quick")
    print("Modelo guardado: models/ppo_quick.zip")
    return "Entrenamiento rápido completado"
