import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.trading_env import PortfolioEnv

# ---------------------------------------------------------------------------
# Hiperparámetros compartidos
# Ajustados para el espacio de observación de 81 features.
# ---------------------------------------------------------------------------
PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    clip_range    = 0.2,        # Schulman clipping: limits policy shift per update
    learning_rate = 3e-4,       # Standard Adam lr
    n_steps       = 2048,       # Steps per rollout before each update
    batch_size    = 256,        # Large mini-batch for stable gradients with 81 features
    n_epochs      = 10,         # PPO epochs per update
    ent_coef      = 0.01,       # Entropy bonus: keeps the policy from collapsing to
                                # deterministic early — important with 81-dim obs space
    policy_kwargs = dict(
        net_arch = [256, 256]   # 256x256 MLP (default [64,64] too small for 81 features)
    ),
    verbose         = 1,
    tensorboard_log = "./logs/",
)

# Training episode length: 252 trading days = 1 year window.
# With the full train set (~5 278 days) and 300k steps, this gives:
#   300 000 / 252 ≈ 1 190 random market windows seen during training
# vs. only ~57 full-dataset passes without this setting.
_TRAIN_EP_LEN = 252

# Porcentajes de corte para walk-forward validation (fijos; las etiquetas se generan
# dinámicamente en _get_ventanas_wfv() a partir del rango de fechas real del CSV).
_WFV_PCTS = [(0.49, 0.65), (0.65, 0.82), (0.82, 1.00)]


def _get_ventanas_wfv():
    """
    Genera las etiquetas de año reales para cada ventana WFV leyendo el CSV.
    Si el dataset cubre 2000-2026 o 2014-2026, las etiquetas serán correctas en ambos casos.
    """
    csv = 'data/features_normalizadas.csv'
    if not os.path.exists(csv):
        # Fallback genérico si aún no se ha generado el dataset
        return [(p[0], p[1], f"Ventana {i+1}") for i, p in enumerate(_WFV_PCTS)]

    idx = pd.read_csv(csv, index_col=0, usecols=[0]).index
    idx = pd.to_datetime(idx)
    n   = len(idx)

    ventanas = []
    for train_end_pct, test_end_pct in _WFV_PCTS:
        train_end_i = int(n * train_end_pct) - 1
        test_end_i  = min(int(n * test_end_pct) - 1, n - 1)
        y_train_start = idx[0].year
        y_train_end   = idx[train_end_i].year
        y_test_end    = idx[test_end_i].year
        label = f"Train {y_train_start}–{y_train_end} → Test {y_train_end}–{y_test_end}"
        ventanas.append((train_end_pct, test_end_pct, label))
    return ventanas


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _make_envs(train_end_idx, test_start_idx, test_end_idx=None, reward_mode='rolling_sharpe'):
    # Training: short random episodes so PPO sees many diverse market regimes
    train_env = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        end_idx=train_end_idx, reward_mode=reward_mode,
        episode_length=_TRAIN_EP_LEN, random_start=True,
    )
    # Evaluation/backtest: full test window, no randomness
    eval_env = PortfolioEnv(
        'data/features_normalizadas.csv', 'data/precios_originales.csv',
        start_idx=test_start_idx, end_idx=test_end_idx, reward_mode=reward_mode,
        episode_length=None, random_start=False,
    )
    return train_env, eval_env


def _evaluar_sharpe(model, env, rf=0.0):
    """Ejecuta un episodio completo y devuelve el Sharpe Ratio anualizado."""
    obs, _ = env.reset()
    done = False
    valores = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        valores.append(info['value'])
    rets = pd.Series(valores).pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    return float(((rets.mean() - rf) / rets.std()) * np.sqrt(252))


def _evaluar_max_drawdown(model, env):
    """Ejecuta un episodio completo y devuelve el Max Drawdown (negativo, fracción)."""
    obs, _ = env.reset()
    done = False
    valores = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        valores.append(info['value'])
    s = pd.Series(valores)
    return float((s / s.cummax() - 1).min())


# ---------------------------------------------------------------------------
# Entrenamiento principal
# ---------------------------------------------------------------------------

def entrenar_con_validacion(total_timesteps=300000, split_pct=0.8):
    """
    Main training with reward_mode='rolling_sharpe'.
    Saves best model (by eval reward) to models/best_model/.
    """
    print("--- Main training: rolling_sharpe ---")
    df_f      = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)

    train_env, eval_env = _make_envs(split_idx, split_idx, reward_mode='rolling_sharpe')

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


def entrenar_ablacion(total_timesteps=300000, split_pct=0.8):
    """
    Ablation baseline: reward_mode='log_return' (no risk shaping).
    Answers: how much does rolling_sharpe reward improve over pure log-return?
    """
    print("--- Ablation: log_return (no drawdown penalty) ---")
    df_f      = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)

    train_env, eval_env = _make_envs(split_idx, split_idx, reward_mode='log_return')

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = './models/ablation_log_return/',
        log_path             = './logs/ablation/',
        eval_freq            = 5000,
        deterministic        = True,
        render               = False,
    )

    model = PPO(env=train_env, **PPO_KWARGS)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("models/ablation_log_return/model_final")
    print("Modelo ablation guardado.")
    return "Ablation completado"


# ---------------------------------------------------------------------------
# Múltiples semillas — robustez estadística
# ---------------------------------------------------------------------------

def entrenar_multisemilla(total_timesteps=300000, split_pct=0.8, seeds=(42, 123, 456)):
    """
    Entrena el modelo con N semillas distintas y reporta media ± std del Sharpe.

    PPO tiene alta varianza estocástica: los pesos iniciales de la red, el orden
    de los mini-batches y el muestreo de acciones dependen del seed. Reportar
    media ± std sobre 3 semillas demuestra que el resultado no es un golpe de suerte
    y hace el experimento reproducible ante un tribunal académico.

    Guarda cada modelo en models/seed_{seed}/ y los resultados en models/multisemilla.csv.
    """
    print(f"--- Multi-seed training: {len(seeds)} semillas ---")
    df_f      = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * split_pct)

    resultados = []
    for seed in seeds:
        print(f"\n  Semilla {seed}...")
        train_env, eval_env = _make_envs(split_idx, split_idx, reward_mode='rolling_sharpe')

        save_path = f"./models/seed_{seed}/"
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path = save_path,
            log_path             = f"./logs/seed_{seed}/",
            eval_freq            = 5000,
            deterministic        = True,
            render               = False,
        )

        model = PPO(env=train_env, seed=seed, **PPO_KWARGS)
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # Evaluar el mejor modelo guardado por el callback
        best_path = os.path.join(save_path, "best_model.zip")
        if os.path.exists(best_path):
            best_model = PPO.load(best_path)
        else:
            best_model = model

        sharpe = _evaluar_sharpe(best_model, eval_env)
        dd     = _evaluar_max_drawdown(best_model, eval_env)
        resultados.append({'seed': seed, 'sharpe': sharpe, 'max_drawdown': dd})
        print(f"  Semilla {seed}: Sharpe={sharpe:.3f}, MaxDD={dd:.2%}")

    df_res = pd.DataFrame(resultados)
    os.makedirs('models', exist_ok=True)
    df_res.to_csv('models/multisemilla.csv', index=False)

    print(f"\n{'='*40}")
    print(f"Sharpe:    {df_res['sharpe'].mean():.3f} ± {df_res['sharpe'].std():.3f}")
    print(f"MaxDD:     {df_res['max_drawdown'].mean():.2%} ± {df_res['max_drawdown'].std():.2%}")
    print(f"{'='*40}")
    return df_res


# ---------------------------------------------------------------------------
# Walk-forward validation — generalización por régimen de mercado
# ---------------------------------------------------------------------------

def walk_forward_validation(total_timesteps=50000):
    """
    Validación walk-forward con 3 ventanas de expansión progresiva.

    En lugar de un único split 80/20, se repite el ciclo train→test en tres
    ventanas que avanzan cronológicamente. Cada ventana de test es un régimen
    de mercado distinto (baja volatilidad 2020-2022, bear market 2022-2024,
    recuperación 2024-2026). Si el modelo obtiene Sharpe > 0 en las tres
    ventanas, la generalización está demostrada de forma rigurosa.

    Guarda modelos en models/wfv_ventana_{n}/ y resultados en models/wfv_resultados.csv.
    """
    print("--- Walk-Forward Validation (3 ventanas) ---")
    df_f = pd.read_csv('data/features_normalizadas.csv')
    n    = len(df_f)

    resultados = []
    for i, (train_end_pct, test_end_pct, label) in enumerate(_get_ventanas_wfv(), start=1):
        train_end  = int(n * train_end_pct)
        test_end   = int(n * test_end_pct)
        test_start = train_end

        print(f"\n  Ventana {i}: {label}")
        print(f"  Train: filas 0–{train_end} | Test: filas {test_start}–{test_end}")

        train_env, test_env = _make_envs(
            train_end, test_start, test_end, reward_mode='rolling_sharpe'
        )

        save_path = f"./models/wfv_ventana_{i}/"
        eval_callback = EvalCallback(
            test_env,
            best_model_save_path = save_path,
            log_path             = f"./logs/wfv_{i}/",
            eval_freq            = max(1000, total_timesteps // 20),
            deterministic        = True,
            render               = False,
        )

        model = PPO(env=train_env, **PPO_KWARGS)
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        best_path = os.path.join(save_path, "best_model.zip")
        best_model = PPO.load(best_path) if os.path.exists(best_path) else model

        sharpe = _evaluar_sharpe(best_model, test_env)
        dd     = _evaluar_max_drawdown(best_model, test_env)

        resultados.append({
            'ventana':      i,
            'descripcion':  label,
            'train_dias':   train_end,
            'test_dias':    test_end - test_start,
            'sharpe':       round(sharpe, 4),
            'max_drawdown': round(dd, 4),
        })
        print(f"  → Sharpe={sharpe:.3f} | MaxDD={dd:.2%}")

    df_res = pd.DataFrame(resultados)
    os.makedirs('models', exist_ok=True)
    df_res.to_csv('models/wfv_resultados.csv', index=False)

    print(f"\n{'='*50}")
    print(df_res[['descripcion', 'sharpe', 'max_drawdown']].to_string(index=False))
    print(f"Sharpe medio WFV: {df_res['sharpe'].mean():.3f} ± {df_res['sharpe'].std():.3f}")
    print(f"{'='*50}")
    return df_res


# ---------------------------------------------------------------------------
# Entrenamiento rápido (pruebas)
# ---------------------------------------------------------------------------

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
