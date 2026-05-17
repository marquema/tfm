"""
Walk-forward (rolling) + expanding window sobre universo honest n=17.

Llama funciones directas sin server FastAPI. PPO hyperparams default
(coherente con flujo memoria §4.3 original — NO usa Optuna best).

Coste estimado:
- Walk-forward: ~2-4h (4-6 ventanas × 100k pasos)
- Expanding:    ~3-5h (más ventanas, 63d test stride)
- Total:        ~6-9h

Genera artefactos en src/reports/:
- walk_forward_results.csv + walk_forward_analysis.png
- expanding_window_results.csv + expanding_window_analysis.png

Uso:
    cd backend
    .venv/Scripts/python.exe scripts/run_walkforward_expanding_honest.py

Opcionalmente solo una:
    .venv/Scripts/python.exe scripts/run_walkforward_expanding_honest.py walkforward
    .venv/Scripts/python.exe scripts/run_walkforward_expanding_honest.py expanding
"""

import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from src.training_drl.training_analysis import (  # noqa: E402
    walk_forward_validation,
    expanding_window_validation,
)


FEATURES_PATH = str(BACKEND_ROOT / 'data' / 'normalized_features.csv')
PRICES_PATH = str(BACKEND_ROOT / 'data' / 'original_prices.csv')
STEPS_PER_WINDOW = 100_000


def run_walkforward():
    print('=' * 70)
    print('WALK-FORWARD (rolling) honest universe n=17')
    print('=' * 70)
    t0 = time.time()
    df = walk_forward_validation(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        train_days=504,
        test_days=252,
        total_timesteps=STEPS_PER_WINDOW,
        reward_type='sharpe',
    )
    elapsed_min = (time.time() - t0) / 60
    print(f'\nWalk-forward COMPLETO en {elapsed_min:.1f} min')
    print(df)
    print('Artefactos: src/reports/walk_forward_results.csv + walk_forward_analysis.png')


def run_expanding():
    print('\n' + '=' * 70)
    print('EXPANDING WINDOW honest universe n=17')
    print('=' * 70)
    t0 = time.time()
    df = expanding_window_validation(
        features_path=FEATURES_PATH,
        prices_path=PRICES_PATH,
        min_train_days=504,
        test_days=63,
        total_timesteps=STEPS_PER_WINDOW,
        reward_type='sharpe',
    )
    elapsed_min = (time.time() - t0) / 60
    print(f'\nExpanding COMPLETO en {elapsed_min:.1f} min')
    print(df)
    print('Artefactos: src/reports/expanding_window_results.csv + expanding_window_analysis.png')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode in ('walkforward', 'wf', 'both'):
        run_walkforward()

    if mode in ('expanding', 'ew', 'both'):
        run_expanding()

    print('\n' + '=' * 70)
    print('DONE')
    print('=' * 70)


if __name__ == '__main__':
    main()
