"""Regenera figuras walk-forward y expanding window con 3 paneles barras
(Sharpe / Retorno / MDD) por ventana, SIN tabla embebida (la tabla LaTeX
tab:wf_por_ventana ya cubre los detalles numericos).

Lee:
    - backend/src/reports/walk_forward_results.csv
    - backend/src/reports/expanding_window_results.csv

Escribe:
    - backend/memoria/walk_forward_analysis.png
    - backend/memoria/expanding_window_analysis.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "src" / "reports"
MEMORIA_DIR = PROJECT_ROOT / "memoria"


def _plot_three_panels(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """3 paneles barras Sharpe / Retorno / MDD por ventana + linea media."""
    n_windows = len(df)
    x_pos = np.arange(n_windows)
    x_labels = [f"V{i + 1}" for i in range(n_windows)]

    panels = [
        ("Sharpe Ratio", "Sharpe Ratio", "#4477aa", 0.0),
        ("Retorno Total (%)", "Retorno Total (\\%)", "#66bb88", 0.0),
        ("Max Drawdown (%)", "Max Drawdown (\\%)", "#cc4444", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n_windows * 0.9), 5))

    for ax, (col, ylabel, color, threshold) in zip(axes, panels):
        values = df[col].values

        # Mark windows con vol<1% en gris (segun filtro robusto)
        vol = df["Volatilidad Anualizada (%)"].values
        low_vol_mask = vol < 1.0

        for i, val in enumerate(values):
            bar_color = "#bbbbbb" if low_vol_mask[i] else color
            edge = "#666666" if low_vol_mask[i] else "white"
            ax.bar(i, val, color=bar_color, alpha=0.85, edgecolor=edge, linewidth=0.5)

        # Etiqueta numerica encima/debajo barra solo si fontsize legible
        label_fontsize = 7 if n_windows > 12 else 8
        if n_windows <= 20:
            for i, val in enumerate(values):
                va = "bottom" if val >= 0 else "top"
                offset = 0.5 if val >= 0 else -0.5
                ax.text(
                    i,
                    val,
                    f"{val:.2f}",
                    ha="center",
                    va=va,
                    fontsize=label_fontsize,
                    color="black",
                )

        # Lineas referencia
        if threshold is not None:
            ax.axhline(threshold, color="black", linestyle="--", linewidth=0.7, alpha=0.6)

        # Media SOLO sobre ventanas con vol >= 1% (filtro robusto)
        valid_mask = ~low_vol_mask
        if valid_mask.any():
            mean_robust = values[valid_mask].mean()
            ax.axhline(
                mean_robust,
                color=color,
                linestyle="-",
                linewidth=1.8,
                alpha=0.9,
                label=f"Media filtrada: {mean_robust:.2f}",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8 if n_windows <= 15 else 6)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, framealpha=0.92, loc="best")

    # Leyenda comun: barras grises = vol<1% (excluidas del filtro robusto)
    fig.text(
        0.5,
        0.005,
        r"\textbf{Barras grises}: ventanas con volatilidad anualizada $<1\,\%$, excluidas del Sharpe robusto.",
        ha="center",
        fontsize=9,
        style="italic",
    )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path.name}")


def main() -> None:
    """Genera ambos PNGs."""
    MEMORIA_DIR.mkdir(parents=True, exist_ok=True)

    wf = pd.read_csv(REPORTS_DIR / "walk_forward_results.csv")
    _plot_three_panels(
        wf,
        title="Walk-forward (rolling): metricas por ventana",
        out_path=MEMORIA_DIR / "walk_forward_analysis.png",
    )

    ew = pd.read_csv(REPORTS_DIR / "expanding_window_results.csv")
    _plot_three_panels(
        ew,
        title="Expanding window: metricas por ventana",
        out_path=MEMORIA_DIR / "expanding_window_analysis.png",
    )


if __name__ == "__main__":
    main()
