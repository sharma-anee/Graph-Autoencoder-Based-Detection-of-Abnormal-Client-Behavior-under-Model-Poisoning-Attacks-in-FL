#!/usr/bin/env python3
"""
Plot test-accuracy line graphs and DR/FDR bar graphs for the fair fixed-schedule FL comparison.

Expected result folder structure:
BASE_DIR/
  signflip_atk10p0_alpha0p5_seed42_rounds100/
    fedavg_attack_nodefense/round_metrics.csv
    krum/round_metrics.csv
    ...
    proposed_thresholding/detection_client_log.csv
    proposed_thresholding/overall_detection_metrics.csv
    proposed_credit_scoring/overall_detection_metrics.csv

Important:
- FedAvg no-attack baseline is read only from BASELINE_ROUND_METRICS and reused in every accuracy subplot.
- DR/FDR bar plots use only proposed_thresholding and proposed_credit_scoring because other robust aggregators do not output detector metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# =========================
# USER SETTINGS
# =========================
BASE_DIR = Path(r"E:\CODES\GAE\Paper_Implementation_2\Logs_Final\fair_comparison_fixed_schedule")

# Your one common no-attack baseline.
BASELINE_ROUND_METRICS = BASE_DIR / "signflip_atk10p0_alpha0p5_seed42_rounds100" / "fedavg_no_attack" / "round_metrics.csv"

OUTPUT_DIR = BASE_DIR / "plots_accuracy_and_detection"
SEED = 42
LIVE_ROUNDS = 100

ATTACK_TYPES = ["signflip", "freeride"]
ATTACK_PCTS = [10.0, 30.0]
ALPHAS = [0.5, 0.7, 1.0]

# Accuracy comparison methods. FedAvg no-attack is handled separately from BASELINE_ROUND_METRICS.
ACCURACY_METHODS = [
    "fedavg_attack_nodefense",
    "krum",
    "multikrum",
    "coord_median",
    "trimmed_mean",
    "proposed_thresholding",
    "proposed_credit_scoring",
]

# Detection bar plots are only meaningful for methods that explicitly log DR/FDR.
DETECTION_METHODS = [
    "proposed_thresholding",
    "proposed_credit_scoring",
]

METHOD_LABELS: Dict[str, str] = {
    "fedavg_no_attack": "FedAvg (No Attack)",
    "fedavg_attack_nodefense": "FedAvg (Attack)",
    "krum": "Krum",
    "multikrum": "Multi-Krum",
    "coord_median": "CWM",
    "trimmed_mean": "TMean",
    "proposed_thresholding": "GAE-Th",
    "proposed_credit_scoring": "GAE-CS",
}

# Same colors and markers are reused across all plots for consistency.
METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "fedavg_no_attack": {"color": "black", "marker": "o", "linestyle": "--"},
    "fedavg_attack_nodefense": {"color": "tab:red", "marker": "s", "linestyle": "-"},
    "krum": {"color": "tab:blue", "marker": "^", "linestyle": "-"},
    "multikrum": {"color": "tab:cyan", "marker": "v", "linestyle": "-"},
    "coord_median": {"color": "tab:purple", "marker": "D", "linestyle": "-"},
    "trimmed_mean": {"color": "tab:brown", "marker": "P", "linestyle": "-"},
    "proposed_thresholding": {"color": "tab:green", "marker": "X", "linestyle": "-"},
    "proposed_credit_scoring": {"color": "tab:orange", "marker": "*", "linestyle": "-"},
}

# Plot appearance for paper figures.
FONT_SIZE = 24
TICK_SIZE = 22
LEGEND_SIZE = 20
LINE_WIDTH = 2.0
MARKER_SIZE = 4
MARK_EVERY = 1  # marker every 10 rounds so the line does not become overcrowded

# Accuracy y-axis. Keep 0-100 for full percentage scale.
TEST_ACC_YLIM = (0, 100)

# Save both PDF and PNG.
SAVE_PDF = True
SAVE_PNG = True
DPI = 400


# =========================
# HELPER FUNCTIONS
# =========================
def safe_float_token(value: float) -> str:
    """Match the folder naming produced by the run script, e.g., 10.0 -> 10p0, 0.5 -> 0p5."""
    return str(float(value)).replace(".", "p")


def condition_folder(attack_type: str, attacker_pct: float, alpha: float) -> Path:
    safe_pct = safe_float_token(attacker_pct)
    safe_alpha = safe_float_token(alpha)
    return BASE_DIR / f"{attack_type}_atk{safe_pct}_alpha{safe_alpha}_seed{SEED}_rounds{LIVE_ROUNDS}"


def existing_csv(path: Path) -> Optional[Path]:
    """Return path if it exists; also tolerate a path accidentally given without .csv."""
    if path.exists():
        return path
    if path.suffix == "" and path.with_suffix(".csv").exists():
        return path.with_suffix(".csv")
    return None


def read_round_metrics_csv(path: Path, method_name: str) -> Optional[pd.DataFrame]:
    csv_path = existing_csv(path)
    if csv_path is None:
        print(f"[MISSING] {method_name}: {path}")
        return None

    df = pd.read_csv(csv_path)
    required = {"Round", "Test Accuracy"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[BAD CSV] {csv_path} missing columns: {sorted(missing_cols)}")
        return None

    df = df.copy()
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["Test Accuracy"] = pd.to_numeric(df["Test Accuracy"], errors="coerce")
    df = df.dropna(subset=["Round", "Test Accuracy"]).sort_values("Round")
    return df


def read_method_round_metrics(attack_type: str, attacker_pct: float, alpha: float, method: str) -> Optional[pd.DataFrame]:
    path = condition_folder(attack_type, attacker_pct, alpha) / method / "round_metrics.csv"
    return read_round_metrics_csv(path, f"{attack_type}, atk={attacker_pct}, alpha={alpha}, {method}")


def read_overall_detection(attack_type: str, attacker_pct: float, alpha: float, method: str) -> Optional[Dict[str, float]]:
    path = condition_folder(attack_type, attacker_pct, alpha) / method / "overall_detection_metrics.csv"
    csv_path = existing_csv(path)
    if csv_path is None:
        print(f"[MISSING] detection summary: {path}")
        return None

    df = pd.read_csv(csv_path)
    required = {"DR", "FDR", "Detection Precision", "Detection Recall", "Detection F1"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[BAD CSV] {csv_path} missing columns: {sorted(missing_cols)}")
        return None

    if df.empty:
        print(f"[EMPTY CSV] {csv_path}")
        return None

    row = df.iloc[-1]
    return {
        "DR": float(row["DR"]),
        "FDR": float(row["FDR"]),
        "Detection Precision": float(row["Detection Precision"]),
        "Detection Recall": float(row["Detection Recall"]),
        "Detection F1": float(row["Detection F1"]),
    }


def apply_common_axis_style(ax: Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def collect_unique_legend(fig: Figure, axes: Iterable[plt.Axes], ncol: int = 4) -> None:
    handles = []
    labels = []
    seen = set()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                seen.add(label)
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=ncol,
            fontsize=LEGEND_SIZE,
            frameon=True,
            bbox_to_anchor=(0.5, -0.005),
        )


def save_figure(fig: Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_PDF:
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(str(pdf_path), bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if SAVE_PNG:
        png_path = OUTPUT_DIR / f"{stem}.png"
        fig.savefig(str(png_path), dpi=DPI, bbox_inches="tight")
        print(f"Saved: {png_path}")


# =========================
# PLOTS
# =========================
def plot_test_accuracy_for_attack(attack_type: str) -> None:
    baseline_df = read_round_metrics_csv(BASELINE_ROUND_METRICS, "fedavg_no_attack baseline")
    if baseline_df is None:
        raise FileNotFoundError(f"Could not read baseline CSV: {BASELINE_ROUND_METRICS}")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
    )

    for row_idx, attacker_pct in enumerate(ATTACK_PCTS):
        for col_idx, alpha in enumerate(ALPHAS):
            ax = axes[row_idx, col_idx]

            # Common no-attack baseline reused in every subplot.
            base_style = METHOD_STYLES["fedavg_no_attack"]
            ax.plot(
                baseline_df["Round"],
                baseline_df["Test Accuracy"],
                label=METHOD_LABELS["fedavg_no_attack"],
                color=base_style["color"],
                linestyle=base_style["linestyle"],
                marker=base_style["marker"],
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                markevery=MARK_EVERY,
            )

            for method in ACCURACY_METHODS:
                df = read_method_round_metrics(attack_type, attacker_pct, alpha, method)
                if df is None:
                    continue
                style = METHOD_STYLES[method]
                ax.plot(
                    df["Round"],
                    df["Test Accuracy"],
                    label=METHOD_LABELS[method],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    markevery=MARK_EVERY,
                )

            ax.set_xlim(1, LIVE_ROUNDS)
            ax.set_ylim(*TEST_ACC_YLIM)
            ax.set_xticks([1, 20, 40, 60, 80, 100])
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.set_title(f"{int(attacker_pct)}% attackers, α={alpha}", fontsize=FONT_SIZE, fontweight="bold")
            apply_common_axis_style(ax)

            if row_idx == len(ATTACK_PCTS) - 1:
                ax.set_xlabel("Round", fontsize=FONT_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Test Accuracy (%)", fontsize=FONT_SIZE, fontweight="bold")

    collect_unique_legend(fig, axes.ravel(), ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, f"{attack_type}_test_accuracy_line")
    plt.close(fig)


def annotate_bars(ax: Axes, bars, decimals: int = 3) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            rotation=0,
        )


def plot_detection_dr_fdr_bars_for_attack(attack_type: str) -> None:
    conditions: List[Tuple[float, float]] = [(pct, alpha) for pct in ATTACK_PCTS for alpha in ALPHAS]
    x = np.arange(len(conditions))
    width = 0.36

    values: Dict[str, Dict[str, List[float]]] = {
        method: {"DR": [], "FDR": []} for method in DETECTION_METHODS
    }
    summary_rows: List[Dict[str, object]] = []

    for pct, alpha in conditions:
        for method in DETECTION_METHODS:
            metrics = read_overall_detection(attack_type, pct, alpha, method)
            if metrics is None:
                dr = np.nan
                fdr = np.nan
            else:
                dr = metrics["DR"]
                fdr = metrics["FDR"]
            values[method]["DR"].append(dr)
            values[method]["FDR"].append(fdr)
            summary_rows.append({
                "Attack Type": attack_type,
                "Attacker Percentage": pct,
                "Alpha": alpha,
                "Method": method,
                "DR": dr,
                "FDR": fdr,
            })

    # Save the bar values as a small CSV for checking/table use.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_DIR / f"{attack_type}_dr_fdr_bar_values.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 9), sharex=True)
    ax_dr, ax_fdr = axes

    offsets = [-width / 2, width / 2]
    for idx, method in enumerate(DETECTION_METHODS):
        style = METHOD_STYLES[method]
        dr_arr = np.array(values[method]["DR"], dtype=float)
        fdr_arr = np.array(values[method]["FDR"], dtype=float)

        bars_dr = ax_dr.bar(
            x + offsets[idx],
            dr_arr,
            width,
            label=METHOD_LABELS[method],
            color=style["color"],
            edgecolor="black",
            linewidth=1.0,
        )
        bars_fdr = ax_fdr.bar(
            x + offsets[idx],
            fdr_arr,
            width,
            label=METHOD_LABELS[method],
            color=style["color"],
            edgecolor="black",
            linewidth=1.0,
        )
        annotate_bars(ax_dr, bars_dr, decimals=3)
        annotate_bars(ax_fdr, bars_fdr, decimals=3)

    condition_labels = [f"{int(pct)}%, α={alpha}" for pct, alpha in conditions]
    ax_fdr.set_xticks(x)
    ax_fdr.set_xticklabels(condition_labels, rotation=25, ha="right", fontsize=TICK_SIZE, fontweight="bold")

    ax_dr.set_ylabel("DR", fontsize=FONT_SIZE, fontweight="bold")
    ax_fdr.set_ylabel("FDR", fontsize=FONT_SIZE, fontweight="bold")
    ax_fdr.set_xlabel("Attack condition", fontsize=FONT_SIZE, fontweight="bold")

    ax_dr.set_ylim(0, 1.08)

    # Auto-scale FDR to make small false detection rates visible, while never exceeding 1.0.
    all_fdr_values = np.concatenate([np.array(values[m]["FDR"], dtype=float) for m in DETECTION_METHODS])
    finite_fdr = all_fdr_values[np.isfinite(all_fdr_values)]
    max_fdr = float(np.max(finite_fdr)) if finite_fdr.size else 0.0
    fdr_ylim_top = min(1.0, max(0.05, max_fdr * 1.35))
    ax_fdr.set_ylim(0, fdr_ylim_top)

    for ax in axes:
        apply_common_axis_style(ax)

    ax_fdr.legend(fontsize=LEGEND_SIZE, prop={"weight": "bold"}, frameon=True, loc="upper right")
    fig.tight_layout()
    save_figure(fig, f"{attack_type}_dr_fdr_bar")
    plt.close(fig)


# =========================
# MAIN
# =========================
def main() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.3,
        "legend.fontsize": LEGEND_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Baseline CSV: {BASELINE_ROUND_METRICS}")

    for attack_type in ATTACK_TYPES:
        print(f"\n=== Plotting test accuracy for {attack_type} ===")
        plot_test_accuracy_for_attack(attack_type)

        print(f"\n=== Plotting DR/FDR bars for {attack_type} ===")
        plot_detection_dr_fdr_bars_for_attack(attack_type)

    print("\nDone.")


if __name__ == "__main__":
    main()
