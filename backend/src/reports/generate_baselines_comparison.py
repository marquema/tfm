"""Genera figura comparativa baselines vs DRL para tfm.tex §4.5.

Bar chart horizontal con Sharpe por estrategia, ordenado descendente. Universo
honest n=17, periodo test 2024-03-09 a 2026-04-30. Datos hard-codeados desde
JSONs canonicos del proyecto para evitar dependencias en runtime.

Output: backend/memoria/baselines_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Datos canonicos (de baselines_honest_results.json + multiseed JSONs).
STRATEGIES = [
    # (nombre, Sharpe, color_grupo)
    ("Cartera 60/40", 0.785, "baseline"),
    ("Markowitz MV", 0.723, "baseline"),
    ("Equal-Weight", 0.926, "baseline"),
    ("Buy & Hold", 0.994, "baseline"),
    ("Especulativo GMM+KMeans", 0.868, "speculative"),
    ("Momentum Top-3", 0.942, "baseline"),
    ("Random Uniform", 1.157, "baseline"),
    ("PPO Optuna LT (media N=5)", 0.474, "drl_mean"),
    ("SAC LT (media N=5)", 0.883, "drl_mean"),
    ("SAC LT (mejor semilla)", 1.280, "drl_top"),
]

COLORS = {
    "baseline": "#6699cc",
    "speculative": "#bba3cc",
    "drl_mean": "#cc8866",
    "drl_top": "#cc3333",
}


def build_figure(out_path: Path) -> None:
    """Genera la figura barra horizontal Sharpe vs estrategia."""
    # Ordenar por Sharpe descendente para lectura inmediata.
    sorted_data = sorted(STRATEGIES, key=lambda item: item[1])
    names = [row[0] for row in sorted_data]
    sharpes = [row[1] for row in sorted_data]
    colors = [COLORS[row[2]] for row in sorted_data]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(names, sharpes, color=colors, edgecolor="black", linewidth=0.4)

    # Valor numerico al final de cada barra.
    for bar, sharpe in zip(bars, sharpes):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{sharpe:.3f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    # Linea de referencia Random Uniform (1.157) como umbral mental.
    ax.axvline(
        1.157,
        color="#666666",
        linestyle="--",
        linewidth=0.8,
        label="Umbral Random Uniform (1.157)",
    )

    ax.set_xlabel("Sharpe Ratio (test, $n=17$, 579 dias)", fontsize=11)
    ax.set_xlim(0, max(sharpes) * 1.15)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)

    # Leyenda manual por grupo.
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["baseline"], label="Baseline clasica"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["speculative"], label="Agente especulativo"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["drl_mean"], label="DRL (media $N=5$)"),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["drl_top"], label="DRL (mejor semilla)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Comparativa Sharpe: agentes DRL vs estrategias de referencia",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    output = project_root / "memoria" / "baselines_comparison.png"
    build_figure(output)
