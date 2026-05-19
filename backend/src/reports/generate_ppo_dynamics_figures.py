"""Genera tres figuras de dinamica de cartera para el TFM:

A. Stacked area de pesos PPO Optuna seed 1 sobre el periodo de test (n=17).
B. Bar chart de Sharpe multi-seed N=5 por algoritmo+perfil.
C. Cripto exposure timeline PPO Optuna seed 1 (IBIT + ETHA + total).

Output: backend/memoria/{ppo_weights_stacked, multiseed_sharpe, ppo_cripto_exposure}.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta colores activos (17 activos honest) por sector aproximado.
ASSET_COLORS = {
    "APP": "#1f77b4",
    "BND": "#aec7e8",
    "CEG": "#ff7f0e",
    "COR": "#ffbb78",
    "COST": "#2ca02c",
    "GE": "#98df8a",
    "IVV": "#d62728",
    "LLY": "#ff9896",
    "META": "#9467bd",
    "NVDA": "#c5b0d5",
    "PHM": "#8c564b",
    "RSG": "#c49c94",
    "SMCI": "#e377c2",
    "VRT": "#f7b6d2",
    "VST": "#7f7f7f",
    "IBIT": "#bcbd22",
    "ETHA": "#17becf",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_CSV = PROJECT_ROOT / "src" / "reports" / "diagnose_ppo_weights.csv"
MEMORIA_DIR = PROJECT_ROOT / "memoria"


def fig_a_stacked_weights(out_path: Path) -> None:
    """Stacked area pesos PPO Optuna seed 1 sobre test."""
    df = pd.read_csv(WEIGHTS_CSV, parse_dates=["date"])
    df = df.set_index("date")
    # Reordenar columnas: defensivos arriba (peso medio mayor), volatiles abajo.
    avg_weight = df.mean().sort_values(ascending=False)
    ordered_cols = list(avg_weight.index)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [ASSET_COLORS.get(c, "#999999") for c in ordered_cols]
    ax.stackplot(
        df.index,
        df[ordered_cols].T.values,
        labels=ordered_cols,
        colors=colors,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.15,
    )

    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Peso de la cartera", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)

    # Leyenda fuera del area, 2 columnas para 17 items.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, ncol=1, frameon=False)

    fig.suptitle(
        "Evolucion diaria de los pesos de cartera ---PPO Optuna seed 1",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated A: {out_path.name}")


def fig_b_multiseed_sharpe(out_path: Path) -> None:
    """Bar chart Sharpe por semilla, agrupado por algoritmo+perfil."""
    # Datos desde JSONs canonicos.
    opt = json.loads((PROJECT_ROOT / "src" / "reports" / "optuna_retrain_results.json").read_text())
    a2c = json.loads((PROJECT_ROOT / "src" / "reports" / "a2c_multiseed_results.json").read_text())
    sac = json.loads((PROJECT_ROOT / "src" / "reports" / "sac_multiseed_results.json").read_text())

    def runs_to_series(runs, profile_filter=None):
        return [r["sharpe"] for r in runs if profile_filter is None or r.get("profile") == profile_filter]

    configs = [
        ("PPO Optuna\nLT", runs_to_series(opt["runs"]), "#4477aa"),
        ("A2C\nLT", runs_to_series(a2c["runs"], "low_turnover"), "#66bb88"),
        ("A2C\naggressive", runs_to_series(a2c["runs"], "aggressive"), "#aaddaa"),
        ("SAC\nLT", runs_to_series(sac["runs"], "low_turnover"), "#cc4444"),
        ("SAC\naggressive", runs_to_series(sac["runs"], "aggressive"), "#ee9966"),
    ]

    n_groups = len(configs)
    n_seeds = 5
    width = 0.14
    x_centers = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6))
    for seed_idx in range(n_seeds):
        seed_values = [cfg[1][seed_idx] for cfg in configs]
        positions = x_centers + (seed_idx - 2) * width
        # Color del grupo, alpha decreciente con la semilla para visual hint.
        for x_pos, val, cfg in zip(positions, seed_values, configs):
            ax.bar(
                x_pos,
                val,
                width=width * 0.95,
                color=cfg[2],
                edgecolor="black",
                linewidth=0.3,
                alpha=0.55 + 0.10 * (seed_idx == np.argmax([cfg[1][s] for s in range(n_seeds)])),
            )

    # Resaltar la mejor semilla de cada config con borde grueso y etiqueta valor.
    for x_center, (label, values, color) in zip(x_centers, configs):
        best_seed = int(np.argmax(values))
        best_val = values[best_seed]
        best_x = x_center + (best_seed - 2) * width
        ax.bar(
            best_x,
            best_val,
            width=width * 0.95,
            color=color,
            edgecolor="black",
            linewidth=1.8,
        )
        ax.text(
            best_x,
            best_val + 0.04,
            f"{best_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Linea Random Uniform.
    ax.axhline(1.157, color="#666666", linestyle="--", linewidth=0.9, label="Umbral Random Uniform (1.157)")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([cfg[0] for cfg in configs], fontsize=10)
    ax.set_ylabel("Sharpe ratio (test)", fontsize=11)
    ax.set_ylim(0, 1.55)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Distribucion Sharpe por semilla ($N=5$): SAC LT seed 4 es el unico DRL que supera Random Uniform",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated B: {out_path.name}")


def fig_c_cripto_exposure(out_path: Path) -> None:
    """Cripto exposure timeline PPO Optuna seed 1."""
    df = pd.read_csv(WEIGHTS_CSV, parse_dates=["date"])
    df = df.set_index("date")
    ibit = df["IBIT"] * 100
    etha = df["ETHA"] * 100
    total = ibit + etha

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df.index, ibit, label="IBIT (BTC)", color="#bcbd22", linewidth=1.0, alpha=0.85)
    ax.plot(df.index, etha, label="ETHA (ETH)", color="#17becf", linewidth=1.0, alpha=0.85)
    ax.plot(df.index, total, label="Total cripto", color="#222222", linewidth=1.6)

    # Linea media total.
    mean_total = total.mean()
    ax.axhline(
        mean_total,
        color="#cc3333",
        linestyle="--",
        linewidth=0.8,
        label=f"Media total cripto: {mean_total:.2f}\\%",
    )

    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Exposicion (\\% cartera)", fontsize=11)
    ax.set_xlim(df.index.min(), df.index.max())
    ax.set_ylim(0, max(total.max() * 1.1, 30))
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.suptitle(
        "Exposicion diaria a criptoactivos ---PPO Optuna seed 1",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated C: {out_path.name}")


if __name__ == "__main__":
    MEMORIA_DIR.mkdir(parents=True, exist_ok=True)
    fig_a_stacked_weights(MEMORIA_DIR / "ppo_weights_stacked.png")
    fig_b_multiseed_sharpe(MEMORIA_DIR / "multiseed_sharpe.png")
    fig_c_cripto_exposure(MEMORIA_DIR / "ppo_cripto_exposure.png")
