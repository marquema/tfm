"""
Search space para Optuna HPO PPO. Decisión P5 (10 dims).

Justificación rangos: centrados en SB3 defaults + catálogo perfiles TFM,
extendidos en ambos extremos para permitir TPE explorar configuraciones no
consideradas manualmente. Log scale para hyperparams sensibilidad-relativa
(learning_rate, entropía, penalizaciones recompensa). Lin scale para hyperparams
con rango compacto.

Documentación memoria final §3.X "Optimización de Hiperparámetros con Optuna":
ver decisión P5 en project_optuna_decisions.md.
"""

import optuna


def sample_hyperparams(trial: optuna.Trial) -> dict:
    """
    Sample 10 hyperparams del trial.

    8 hyperparams PPO + 2 recompensa (varphi, gamma).
    Justificación de cada rango: comentario inline.

    Returns
    -------
    dict con keys:
        learning_rate, n_steps, batch_size, clip_range, ent_coef,
        gae_lambda, vf_coef, max_grad_norm, varphi, gamma
    """
    return {
        # Learning rate: log [1e-5, 3e-3]. Cubre SB3 default (3e-4) hasta
        # agresivos y conservadores. Sensibilidad multiplicativa → log.
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 3e-3, log=True),

        # n_steps: potencias 2 alrededor del actual TFM (2048). Trayectoria
        # on-policy entre actualizaciones.
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),

        # batch_size: divisores típicos n_steps. Estabilidad gradiente.
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),

        # clip_range (epsilon PPO): [0.05, 0.3]. Cubre paper PPO (0.1-0.3)
        # más conservador (0.05). Lin scale, rango compacto.
        'clip_range': trial.suggest_float('clip_range', 0.05, 0.3),

        # ent_coef: log [1e-4, 1e-1]. Exploración baja (1e-4) a alta (0.1).
        # Default SB3 = 0. TFM usa 0.01.
        'ent_coef': trial.suggest_float('ent_coef', 1e-4, 1e-1, log=True),

        # gae_lambda: [0.9, 0.99]. Varianza estimación ventaja. Rango habitual
        # literatura PPO.
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),

        # vf_coef: [0.3, 1.0]. Peso crítico vs actor loss. Default SB3 = 0.5.
        'vf_coef': trial.suggest_float('vf_coef', 0.3, 1.0),

        # max_grad_norm: [0.3, 1.0]. Clipping gradiente estabilidad.
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),

        # varphi (MDD penalty): log [0.005, 0.1]. Cubre catálogo perfiles:
        # aggressive (0.01) → conservative (0.05), extendido a ambos extremos.
        'varphi': trial.suggest_float('varphi', 0.005, 0.1, log=True),

        # gamma (turnover penalty): log [0.001, 0.05]. Cubre catálogo:
        # aggressive (0.005) → low_turnover (0.020), extendido.
        'gamma': trial.suggest_float('gamma', 0.001, 0.05, log=True),
    }


def validate_batch_size_compatibility(n_steps: int, batch_size: int) -> int:
    """
    Garantiza batch_size <= n_steps. Si no, devuelve min(batch_size, n_steps).

    SB3 PPO requiere n_steps % batch_size == 0 idealmente. Si trial sugiere
    combo incompatible (e.g. n_steps=512, batch_size=256 OK; n_steps=512,
    batch_size=128 OK), no hace ajuste. Si n_steps < batch_size, devuelve n_steps.
    """
    if batch_size > n_steps:
        return n_steps
    return batch_size
