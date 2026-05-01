"""
Diagnostico del PPO entrenado: reproduce el backtest sobre el periodo
test y emite los pesos diarios + metricas para entender por que el PPO
rinde mal frente a baselines triviales.

Salida:
  - src/reports/diagnose_ppo_weights.csv : pesos diarios del PPO en test
  - src/reports/diagnose_ppo_summary.json: resumen ejecutivo del diagnostico
  - prints en consola con las observaciones clave

Uso:
    .venv/Scripts/python.exe scripts/diagnose_ppo.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(HERE)
sys.path.insert(0, BACKEND_ROOT)

from stable_baselines3 import PPO
from src.training_drl.environment_trading import PortfolioEnv

FEATURES_PATH = os.path.join(BACKEND_ROOT, 'data', 'normalized_features.csv')
PRICES_PATH = os.path.join(BACKEND_ROOT, 'data', 'original_prices.csv')
MODEL_PATH = os.path.join(BACKEND_ROOT, 'models', 'best_model_academic',
                          'best_model.zip')
OUT_WEIGHTS = os.path.join(BACKEND_ROOT, 'src', 'reports',
                           'diagnose_ppo_weights.csv')
OUT_SUMMARY = os.path.join(BACKEND_ROOT, 'src', 'reports',
                           'diagnose_ppo_summary.json')

SPLIT_PCT = 0.8


def main():
    print("\n" + "#" * 70)
    print("# DIAGNOSTICO PPO - reproducir backtest y analizar comportamiento")
    print("#" * 70)

    # ── 1. Cargar datos y modelo ────────────────────────────────────────
    df_p = pd.read_csv(PRICES_PATH, index_col=0)
    df_f = pd.read_csv(FEATURES_PATH, index_col=0)
    n_total = len(df_f)
    split_idx = int(n_total * SPLIT_PCT)
    n_test = n_total - split_idx
    tickers = [c.replace('_Close', '') for c in df_p.columns]

    print(f"\nDataset total: {n_total} dias")
    print(f"Train: 0 .. {split_idx} ({split_idx} dias)")
    print(f"Test:  {split_idx} .. {n_total} ({n_test} dias)")
    print(f"Universo ({len(tickers)}): {tickers}")
    print(f"Test arranca: {df_f.index[split_idx]}")
    print(f"Test termina: {df_f.index[-1]}")

    # ── 2. Sanity check sobre features ──────────────────────────────────
    nan_count = df_f.isna().sum().sum()
    inf_count = np.isinf(df_f.values).sum()
    print(f"\nNaN en features: {nan_count}")
    print(f"Inf en features: {inf_count}")

    # ── 3. Sanity check sobre precios ───────────────────────────────────
    test_prices = df_p.iloc[split_idx:]
    print(f"\nRetornos por activo en test (last/first - 1):")
    for col in df_p.columns:
        ticker = col.replace('_Close', '')
        ret = (test_prices[col].iloc[-1] / test_prices[col].iloc[0] - 1) * 100
        print(f"  {ticker:>6}: {ret:+.2f}%  "
              f"({test_prices[col].iloc[0]:.2f} -> {test_prices[col].iloc[-1]:.2f})")

    # ── 4. Cargar modelo y reproducir backtest ──────────────────────────
    print(f"\nCargando modelo: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    env = PortfolioEnv(
        FEATURES_PATH, PRICES_PATH,
        start_idx=split_idx,
    )
    obs, _ = env.reset()
    done = False
    weights_log = []
    equity_log = [env.initial_balance]
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # Reproducir la normalizacion que hace el entorno: clip a [0,1] y suma=1
        w = np.clip(action, 0, 1)
        w = w / (w.sum() + 1e-8)
        weights_log.append(w)
        obs, _, done, _, info = env.step(action)
        equity_log.append(float(info['value']))
        step += 1

    weights_arr = np.array(weights_log)
    print(f"\nBacktest reproducido: {weights_arr.shape[0]} pasos")

    # ── 5. Analizar comportamiento del PPO ──────────────────────────────
    avg_weights = weights_arr.mean(axis=0)
    std_weights = weights_arr.std(axis=0)
    median_weights = np.median(weights_arr, axis=0)
    max_weights = weights_arr.max(axis=0)
    min_weights = weights_arr.min(axis=0)

    print(f"\nPesos del PPO (media | std | median | min-max) por activo:")
    for i, t in enumerate(tickers):
        print(f"  {t:>6}: {avg_weights[i]*100:>6.2f}% +- {std_weights[i]*100:>5.2f}  "
              f"| med {median_weights[i]*100:>5.2f}%  "
              f"| {min_weights[i]*100:>5.2f}% .. {max_weights[i]*100:>5.2f}%")

    # Turnover total
    if weights_arr.shape[0] > 1:
        daily_turnover = np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)
        avg_turnover = daily_turnover.mean()
        max_turnover = daily_turnover.max()
        print(f"\nTurnover diario medio: {avg_turnover:.4f}  "
              f"(max: {max_turnover:.4f})")
        # Compara con un "rebalanceo mensual" tipico (~1/22 = 0.045)
        print(f"  ref. rebalanceo mensual completo: ~0.090 (90% rotacion al mes)")

    # Detectar concentracion: hay algun activo con >50% del peso medio?
    top_idx = np.argsort(avg_weights)[::-1]
    top3 = [tickers[i] for i in top_idx[:3]]
    top3_pct = sum(avg_weights[i] for i in top_idx[:3]) * 100
    print(f"\nTop-3 activos por peso medio: {top3} (suman {top3_pct:.1f}%)")

    # Cuanto tiempo dedico a cripto?
    crypto_idx = [i for i, t in enumerate(tickers) if t in ('IBIT', 'ETHA')]
    if crypto_idx:
        crypto_avg = sum(avg_weights[i] for i in crypto_idx) * 100
        crypto_pct_per_day = weights_arr[:, crypto_idx].sum(axis=1)
        print(f"\nExposicion a cripto (IBIT+ETHA):")
        print(f"  Peso medio: {crypto_avg:.2f}%")
        print(f"  Peso minimo: {crypto_pct_per_day.min()*100:.2f}%  "
              f"Peso maximo: {crypto_pct_per_day.max()*100:.2f}%")
        # ¿Cuantos dias estuvo "muy expuesto" a cripto (>20%) o "muy poco" (<5%)?
        n_high = (crypto_pct_per_day > 0.20).sum()
        n_low = (crypto_pct_per_day < 0.05).sum()
        print(f"  Dias con >20% en cripto: {n_high} / {n_test}  "
              f"({n_high/n_test*100:.1f}%)")
        print(f"  Dias con <5%  en cripto: {n_low} / {n_test}  "
              f"({n_low/n_test*100:.1f}%)")

    # ── 6. Guardar pesos diarios ────────────────────────────────────────
    df_w = pd.DataFrame(
        weights_arr,
        index=df_f.index[split_idx:split_idx + weights_arr.shape[0]],
        columns=tickers,
    )
    os.makedirs(os.path.dirname(OUT_WEIGHTS), exist_ok=True)
    df_w.to_csv(OUT_WEIGHTS, encoding='utf-8-sig')
    print(f"\nPesos diarios guardados: {OUT_WEIGHTS}")

    # ── 7. Resumen ejecutivo en JSON ────────────────────────────────────
    summary = {
        'tickers': tickers,
        'test_period': {
            'start': str(df_f.index[split_idx]),
            'end': str(df_f.index[-1]),
            'n_days': int(n_test),
        },
        'avg_weights_pct': {t: float(avg_weights[i] * 100)
                            for i, t in enumerate(tickers)},
        'std_weights_pct': {t: float(std_weights[i] * 100)
                            for i, t in enumerate(tickers)},
        'crypto_exposure_pct': {
            'avg': float(sum(avg_weights[i] for i in crypto_idx) * 100)
                if crypto_idx else 0.0,
            'min': float(crypto_pct_per_day.min() * 100) if crypto_idx else 0.0,
            'max': float(crypto_pct_per_day.max() * 100) if crypto_idx else 0.0,
        } if crypto_idx else None,
        'avg_daily_turnover': float(avg_turnover) if weights_arr.shape[0] > 1 else 0.0,
        'top3_avg_weight': top3,
    }
    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Resumen guardado: {OUT_SUMMARY}")
    print("\n" + "=" * 70)
    print("DIAGNOSTICO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()