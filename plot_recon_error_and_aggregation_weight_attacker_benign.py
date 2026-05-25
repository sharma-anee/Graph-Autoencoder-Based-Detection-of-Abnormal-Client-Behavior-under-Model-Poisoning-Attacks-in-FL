#!/usr/bin/env python3
"""
Plot reconstruction error and aggregation weight comparisons for attackers vs benign/non-attacking clients.

This script reads detection_client_log.csv from the proposed methods only:
    proposed_thresholding
    proposed_credit_scoring

It creates:
1. Overall bar plots comparing benign/non-attacking clients vs actual attackers.
2. Per-round line plots comparing mean values of benign/non-attacking clients vs actual attackers.

Metrics:
    Recon_Error
    Aggregation_Weight

Important:
- "Benign" here means Is_Actual_Attacker == 0 in that round.
  This includes normal benign clients and potential attackers who did not actually attack in that round.
- No experiments are rerun. The script only reads saved CSV logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# =========================
# USER SETTINGS
# =========================
BASE_DIR = Path(r"E:\CODES\GAE\Paper_Implementation_2\Logs_Final\fair_comparison_fixed_schedule")
OUTPUT_DIR = BASE_DIR / "plots_recon_error_and_aggregation_weight"

SEED = 42
LIVE_ROUNDS = 100

ATTACK_TYPES = ["signflip", "freeride"]
ATTACK_PCTS = [10.0, 30.0]
ALPHAS = [0.5, 0.7, 1.0]

PROPOSED_METHODS = [
    "proposed_thresholding",
    "proposed_credit_scoring",
]

METHOD_LABELS: Dict[str, str] = {
    "proposed_thresholding": "GAE-Th",
    "proposed_credit_scoring": "GAE-CS",
}

METHOD_STYLES: Dict[str, Dict[str, object]] = {
    "proposed_thresholding": {"color": "tab:green", "marker": "o", "linestyle": "-"},
    "proposed_credit_scoring": {"color": "tab:orange", "marker": "s", "linestyle": "-"},
}

GROUP_LABELS: Dict[int, str] = {
    0: "Benign / Non-attacking",
    1: "Actual attackers",
}

GROUP_STYLES: Dict[int, Dict[str, object]] = {
    0: {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    1: {"color": "tab:red", "marker": "s", "linestyle": "-"},
}

METRICS: Dict[str, Dict[str, str]] = {
    "Recon_Error": {
        "safe_name": "reconstruction_error",
        "ylabel": "Reconstruction Error",
        "title": "Reconstruction Error",
    },
    "Aggregation_Weight": {
        "safe_name": "aggregation_weight",
        "ylabel": "Aggregation Weight",
        "title": "Aggregation Weight",
    },
}

FONT_SIZE = 28
TICK_SIZE = 26
LEGEND_SIZE = 24
LINE_WIDTH = 2.8
MARKER_SIZE = 6
MARK_EVERY = 1
BAR_VALUE_DECIMALS = 2

SAVE_PDF = True
SAVE_PNG = True
DPI = 400


# =========================
# HELPER FUNCTIONS
# =========================
def safe_float_token(value: float) -> str:
    """Match run-folder naming, e.g., 10.0 -> 10p0, 0.5 -> 0p5."""
    return str(float(value)).replace(".", "p")


def condition_folder(attack_type: str, attacker_pct: float, alpha: float) -> Path:
    safe_pct = safe_float_token(attacker_pct)
    safe_alpha = safe_float_token(alpha)
    return BASE_DIR / f"{attack_type}_atk{safe_pct}_alpha{safe_alpha}_seed{SEED}_rounds{LIVE_ROUNDS}"


def existing_csv(path: Path) -> Optional[Path]:
    if path.exists():
        return path
    if path.suffix == "" and path.with_suffix(".csv").exists():
        return path.with_suffix(".csv")
    return None


def detection_client_log_path(attack_type: str, attacker_pct: float, alpha: float, method: str) -> Path:
    return condition_folder(attack_type, attacker_pct, alpha) / method / "detection_client_log.csv"


def read_detection_client_log(
    attack_type: str,
    attacker_pct: float,
    alpha: float,
    method: str,
) -> Optional[pd.DataFrame]:
    path = detection_client_log_path(attack_type, attacker_pct, alpha, method)
    csv_path = existing_csv(path)

    if csv_path is None:
        print(f"[MISSING] detection client log: {path}")
        return None

    df = pd.read_csv(csv_path)
    required = {"Round", "Is_Actual_Attacker", "Recon_Error", "Aggregation_Weight"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"[BAD CSV] {csv_path} missing columns: {sorted(missing_cols)}")
        return None

    df = df.copy()
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["Is_Actual_Attacker"] = pd.to_numeric(df["Is_Actual_Attacker"], errors="coerce").fillna(0).astype(int)
    df["Recon_Error"] = pd.to_numeric(df["Recon_Error"], errors="coerce")
    df["Aggregation_Weight"] = pd.to_numeric(df["Aggregation_Weight"], errors="coerce")

    df = df.dropna(subset=["Round", "Recon_Error", "Aggregation_Weight"])
    df = df.sort_values(["Round", "ClientID"]) if "ClientID" in df.columns else df.sort_values("Round")
    return df


def apply_common_axis_style(ax: Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


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


def collect_unique_legend(fig: Figure, axes: np.ndarray, ncol: int = 2) -> None:
    handles = []
    labels = []
    seen = set()

    for ax in axes.ravel():
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


def annotate_bars(ax: Axes, bars, decimals: int = BAR_VALUE_DECIMALS) -> None:
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
            fontsize=24,
            fontweight="bold",
            rotation=0,
        )


def summarize_metric_by_group(df: pd.DataFrame, metric_col: str) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}

    for group_value in [0, 1]:
        values = df.loc[df["Is_Actual_Attacker"] == group_value, metric_col].dropna().astype(float)
        if values.empty:
            summary[group_value] = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "count": 0.0,
            }
        else:
            summary[group_value] = {
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": float(values.shape[0]),
            }

    return summary


def per_round_group_means(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["Round", "Is_Actual_Attacker"], as_index=False)[metric_col]
        .mean()
        .sort_values(["Round", "Is_Actual_Attacker"])
    )
    grouped["Round"] = grouped["Round"].astype(int)
    return grouped


# =========================
# BAR PLOTS
# =========================
def plot_metric_bar_for_attack(attack_type: str, metric_col: str) -> None:
    metric_info = METRICS[metric_col]
    conditions: List[Tuple[float, float]] = [(pct, alpha) for pct in ATTACK_PCTS for alpha in ALPHAS]
    condition_labels = [f"{int(pct)}%, α={alpha}" for pct, alpha in conditions]

    summary_rows: List[Dict[str, object]] = []

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(30, 12),
        sharey=False,
    )

    for ax_idx, method in enumerate(PROPOSED_METHODS):
        ax = axes[ax_idx]
        benign_means: List[float] = []
        attacker_means: List[float] = []

        for pct, alpha in conditions:
            df = read_detection_client_log(attack_type, pct, alpha, method)

            if df is None:
                summary = {
                    0: {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0.0},
                    1: {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "count": 0.0},
                }
            else:
                summary = summarize_metric_by_group(df, metric_col)

            benign_means.append(summary[0]["mean"])
            attacker_means.append(summary[1]["mean"])

            for group_value in [0, 1]:
                summary_rows.append({
                    "Attack Type": attack_type,
                    "Attacker Percentage": pct,
                    "Alpha": alpha,
                    "Method": method,
                    "Group": GROUP_LABELS[group_value],
                    "Metric": metric_col,
                    "Mean": summary[group_value]["mean"],
                    "Median": summary[group_value]["median"],
                    "Std": summary[group_value]["std"],
                    "Min": summary[group_value]["min"],
                    "Max": summary[group_value]["max"],
                    "Count": summary[group_value]["count"],
                })

        x = np.arange(len(conditions))
        width = 0.36

        bars_benign = ax.bar(
            x - width / 2,
            np.array(benign_means, dtype=float),
            width,
            label=GROUP_LABELS[0],
            color=GROUP_STYLES[0]["color"],
            edgecolor="black",
            linewidth=1.0,
        )
        bars_attack = ax.bar(
            x + width / 2,
            np.array(attacker_means, dtype=float),
            width,
            label=GROUP_LABELS[1],
            color=GROUP_STYLES[1]["color"],
            edgecolor="black",
            linewidth=1.0,
        )

        annotate_bars(ax, bars_benign)
        annotate_bars(ax, bars_attack)

        ax.set_title(METHOD_LABELS[method], fontsize=FONT_SIZE, fontweight="bold")
        ax.set_ylabel(metric_info["ylabel"], fontsize=FONT_SIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(condition_labels, rotation=25, ha="right", fontsize=TICK_SIZE, fontweight="bold")
        apply_common_axis_style(ax)

    if metric_col == "Recon_Error":
        axes[1].legend(
            prop={"size": LEGEND_SIZE, "weight": "bold"},
            frameon=True,
            loc="center right",
            bbox_to_anchor=(0.98, 0.5),
        )
    else:
        axes[0].legend(
            prop={"size": LEGEND_SIZE, "weight": "bold"},
            frameon=True,
            loc="best",
        )

    summary_csv = OUTPUT_DIR / f"{attack_type}_{metric_info['safe_name']}_attacker_benign_summary.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    fig.tight_layout()
    save_figure(fig, f"{attack_type}_{metric_info['safe_name']}_attacker_benign_bar")
    plt.close(fig)


# =========================
# LINE PLOTS
# =========================
def plot_metric_line_for_attack_method(attack_type: str, method: str, metric_col: str) -> None:
    metric_info = METRICS[metric_col]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(30, 12),
        sharex=True,
        sharey=False,
    )

    for row_idx, attacker_pct in enumerate(ATTACK_PCTS):
        for col_idx, alpha in enumerate(ALPHAS):
            ax = axes[row_idx, col_idx]
            df = read_detection_client_log(attack_type, attacker_pct, alpha, method)

            if df is not None and not df.empty:
                grouped = per_round_group_means(df, metric_col)

                for group_value in [0, 1]:
                    group_df = grouped[grouped["Is_Actual_Attacker"] == group_value]
                    if group_df.empty:
                        continue

                    style = GROUP_STYLES[group_value]
                    ax.plot(
                        group_df["Round"],
                        group_df[metric_col],
                        label=GROUP_LABELS[group_value],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        linewidth=LINE_WIDTH,
                        markersize=MARKER_SIZE,
                        markevery=MARK_EVERY,
                    )

            ax.set_xlim(1, LIVE_ROUNDS)
            ax.set_xticks([1, 20, 40, 60, 80, 100])
            ax.set_title(f"{int(attacker_pct)}% attackers, α={alpha}", fontsize=FONT_SIZE, fontweight="bold")
            apply_common_axis_style(ax)

            if row_idx == len(ATTACK_PCTS) - 1:
                ax.set_xlabel("Round", fontsize=FONT_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_info["ylabel"], fontsize=FONT_SIZE, fontweight="bold")

    collect_unique_legend(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, f"{attack_type}_{method}_{metric_info['safe_name']}_attacker_benign_line")
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

    for attack_type in ATTACK_TYPES:
        for metric_col in METRICS.keys():
            print(f"\n=== Plotting {metric_col} bar plot for {attack_type} ===")
            plot_metric_bar_for_attack(attack_type, metric_col)

            for method in PROPOSED_METHODS:
                print(f"\n=== Plotting {metric_col} line plot for {attack_type}, {method} ===")
                plot_metric_line_for_attack_method(attack_type, method, metric_col)

    print("\nDone.")


if __name__ == "__main__":
    main()
