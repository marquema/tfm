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
    """Stacked area pesos PPO Optuna seed 1 sobre test, agrupado en 4 categorias."""
    df = pd.read_csv(WEIGHTS_CSV, parse_dates=["date"])
    df = df.set_index("date")

    # Agrupar 17 activos en 4 categorias estructurales.
    # Orden visual: defensivos abajo (estable), cripto arriba (volatil).
    categories = {
        "Defensivos / Calidad (VST, COR, RSG, LLY, COST)": ["VST", "COR", "RSG", "LLY", "COST"],
        "Equity broad + Renta fija (IVV, BND)": ["IVV", "BND"],
        "Tech / Growth volatil (NVDA, SMCI, APP, VRT, META, CEG, GE, PHM)": [
            "NVDA", "SMCI", "APP", "VRT", "META", "CEG", "GE", "PHM",
        ],
        "Cripto (IBIT, ETHA)": ["IBIT", "ETHA"],
    }
    cat_colors = {
        "Defensivos / Calidad (VST, COR, RSG, LLY, COST)": "#2c7a2c",
        "Equity broad + Renta fija (IVV, BND)": "#4477aa",
        "Tech / Growth volatil (NVDA, SMCI, APP, VRT, META, CEG, GE, PHM)": "#cc5511",
        "Cripto (IBIT, ETHA)": "#aa5599",
    }

    grouped = pd.DataFrame({label: df[tickers].sum(axis=1) for label, tickers in categories.items()})

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(
        grouped.index,
        grouped.T.values,
        labels=list(categories.keys()),
        colors=[cat_colors[k] for k in categories.keys()],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.4,
    )

    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Peso de la cartera", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(grouped.index.min(), grouped.index.max())
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=9, ncol=2, frameon=False)

    fig.suptitle(
        "Evolucion diaria de los pesos de cartera por categoria ---PPO Optuna seed 1",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated A: {out_path.name}")


def fig_a_heatmap_weights(out_path: Path) -> None:
    """Heatmap pesos activo x tiempo, agrupado por categoria estructural."""
    df = pd.read_csv(WEIGHTS_CSV, parse_dates=["date"])
    df = df.set_index("date")

    # Resample a frecuencia mensual (mean) para legibilidad. 579 dias -> ~25-27 meses.
    monthly = df.resample("ME").mean()

    # Orden filas por categoria + peso medio dentro de cada categoria.
    categories = [
        ("Defensivos /\nCalidad", ["VST", "COR", "RSG", "LLY", "COST"]),
        ("Equity broad +\nRenta fija", ["IVV", "BND"]),
        ("Tech / Growth\nvolatil", ["NVDA", "SMCI", "APP", "VRT", "META", "CEG", "GE", "PHM"]),
        ("Cripto", ["IBIT", "ETHA"]),
    ]
    ordered_assets = []
    category_borders = []  # indices donde cambia categoria, para lineas separadoras
    for label, tickers in categories:
        category_borders.append(len(ordered_assets))
        # Dentro de cada categoria, ordenar por peso medio descendente.
        sorted_tickers = sorted(tickers, key=lambda t: -df[t].mean())
        ordered_assets.extend(sorted_tickers)
    category_borders.append(len(ordered_assets))  # final

    matrix = monthly[ordered_assets].T.values * 100  # a porcentajes

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=20,
        interpolation="nearest",
    )

    # Eje X: meses como labels.
    n_cols = matrix.shape[1]
    tick_step = max(1, n_cols // 12)
    tick_positions = np.arange(0, n_cols, tick_step)
    tick_labels = [monthly.index[i].strftime("%Y-%m") for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Eje Y: tickers.
    ax.set_yticks(np.arange(len(ordered_assets)))
    ax.set_yticklabels(ordered_assets, fontsize=9)

    # Lineas horizontales separando categorias.
    for border in category_borders[1:-1]:
        ax.axhline(border - 0.5, color="black", linewidth=1.4)

    # Etiquetas categoria a la derecha del heatmap.
    for i, (label, _) in enumerate(categories):
        mid = (category_borders[i] + category_borders[i + 1] - 1) / 2
        ax.text(
            n_cols + 0.5,
            mid,
            label,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # Colorbar.
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.12)
    cbar.set_label("Peso medio mensual (\\%)", fontsize=10)

    ax.set_xlabel("Mes", fontsize=11)
    fig.suptitle(
        "Heatmap mensual de pesos por activo ---PPO Optuna seed 1",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated A1 heatmap: {out_path.name}")


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
    fig_a_heatmap_weights(MEMORIA_DIR / "ppo_weights_heatmap.png")
    fig_b_multiseed_sharpe(MEMORIA_DIR / "multiseed_sharpe.png")
    fig_c_cripto_exposure(MEMORIA_DIR / "ppo_cripto_exposure.png")
