#!/usr/bin/env python3
"""
Plot anomaly-score separation between actual attackers and benign/non-attacking clients
for the proposed GAE-based methods in the fair fixed-schedule FL comparison.

Inputs expected under BASE_DIR:
BASE_DIR/<condition>/<method>/detection_client_log.csv

Each detection_client_log.csv should contain:
Round, ClientID, Is_Potential_Attacker, Is_Actual_Attacker,
Recon_Error, Anomaly_Score, Predicted_Attacker, Aggregation_Weight

Outputs:
- Per-round mean anomaly-score line plots for benign vs actual attackers.
- Overall mean anomaly-score bar plots for benign vs actual attackers.
- CSV files containing the summary statistics used in the bar plots.
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
OUTPUT_DIR = BASE_DIR / "plots_anomaly_scores"

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

GROUP_LABELS: Dict[int, str] = {
    0: "Benign / Non-attacking",
    1: "Actual attacker",
}

GROUP_STYLES: Dict[int, Dict[str, object]] = {
    0: {"color": "tab:blue", "marker": "o", "linestyle": "-"},
    1: {"color": "tab:red", "marker": "s", "linestyle": "-"},
}

FONT_SIZE = 28
TICK_SIZE = 24
LEGEND_SIZE = 22
LINE_WIDTH = 2.8
MARKER_SIZE = 6
MARK_EVERY = 1
BAR_WIDTH = 0.36
DPI = 400
SAVE_PDF = True
SAVE_PNG = True

# Optional: set to True if you want the y-axis of each 2x3 line figure to share a single scale.
SHARE_Y_FOR_LINE_PLOTS = False


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
        print(f"[MISSING] {path}")
        return None

    df = pd.read_csv(csv_path)
    required = {"Round", "Is_Actual_Attacker", "Anomaly_Score"}
    missing = required - set(df.columns)
    if missing:
        print(f"[BAD CSV] {csv_path} missing columns: {sorted(missing)}")
        return None

    df = df.copy()
    df["Round"] = pd.to_numeric(df["Round"], errors="coerce")
    df["Is_Actual_Attacker"] = pd.to_numeric(df["Is_Actual_Attacker"], errors="coerce")
    df["Anomaly_Score"] = pd.to_numeric(df["Anomaly_Score"], errors="coerce")
    df = df.dropna(subset=["Round", "Is_Actual_Attacker", "Anomaly_Score"])
    df["Round"] = df["Round"].astype(int)
    df["Is_Actual_Attacker"] = df["Is_Actual_Attacker"].astype(int)
    df = df[df["Is_Actual_Attacker"].isin([0, 1])]
    return df


def apply_common_axis_style(ax: Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def collect_unique_legend(fig: Figure, axes: Iterable[Axes], ncol: int = 2) -> None:
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


def condition_label(attacker_pct: float, alpha: float) -> str:
    return f"{int(attacker_pct)}%, α={alpha}"


# =========================
# SUMMARY STATISTICS
# =========================
def compute_anomaly_summary_for_attack(attack_type: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for attacker_pct in ATTACK_PCTS:
        for alpha in ALPHAS:
            for method in PROPOSED_METHODS:
                df = read_detection_client_log(attack_type, attacker_pct, alpha, method)
                if df is None or df.empty:
                    for group_value in [0, 1]:
                        rows.append({
                            "Attack Type": attack_type,
                            "Attacker Percentage": attacker_pct,
                            "Alpha": alpha,
                            "Method": method,
                            "Method Label": METHOD_LABELS[method],
                            "Client Group": GROUP_LABELS[group_value],
                            "Is Actual Attacker": group_value,
                            "Mean Anomaly Score": np.nan,
                            "Median Anomaly Score": np.nan,
                            "Std Anomaly Score": np.nan,
                            "Min Anomaly Score": np.nan,
                            "Max Anomaly Score": np.nan,
                            "Count": 0,
                        })
                    continue

                for group_value in [0, 1]:
                    scores = df.loc[df["Is_Actual_Attacker"] == group_value, "Anomaly_Score"].dropna()
                    rows.append({
                        "Attack Type": attack_type,
                        "Attacker Percentage": attacker_pct,
                        "Alpha": alpha,
                        "Method": method,
                        "Method Label": METHOD_LABELS[method],
                        "Client Group": GROUP_LABELS[group_value],
                        "Is Actual Attacker": group_value,
                        "Mean Anomaly Score": float(scores.mean()) if not scores.empty else np.nan,
                        "Median Anomaly Score": float(scores.median()) if not scores.empty else np.nan,
                        "Std Anomaly Score": float(scores.std(ddof=0)) if not scores.empty else np.nan,
                        "Min Anomaly Score": float(scores.min()) if not scores.empty else np.nan,
                        "Max Anomaly Score": float(scores.max()) if not scores.empty else np.nan,
                        "Count": int(scores.size),
                    })
    return pd.DataFrame(rows)


# =========================
# LINE PLOTS
# =========================
def plot_anomaly_score_lines_for_attack_and_method(attack_type: str, method: str) -> None:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(24, 12),
        sharex=True,
        sharey=SHARE_Y_FOR_LINE_PLOTS,
    )

    for row_idx, attacker_pct in enumerate(ATTACK_PCTS):
        for col_idx, alpha in enumerate(ALPHAS):
            ax = axes[row_idx, col_idx]
            df = read_detection_client_log(attack_type, attacker_pct, alpha, method)

            if df is not None and not df.empty:
                grouped = (
                    df.groupby(["Round", "Is_Actual_Attacker"], as_index=False)["Anomaly_Score"]
                    .mean()
                    .sort_values("Round")
                )

                for group_value in [0, 1]:
                    group_df = grouped[grouped["Is_Actual_Attacker"] == group_value]
                    style = GROUP_STYLES[group_value]
                    ax.plot(
                        group_df["Round"],
                        group_df["Anomaly_Score"],
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
            ax.set_title(condition_label(attacker_pct, alpha), fontsize=FONT_SIZE, fontweight="bold")
            apply_common_axis_style(ax)

            if row_idx == len(ATTACK_PCTS) - 1:
                ax.set_xlabel("Round", fontsize=FONT_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Mean anomaly score", fontsize=FONT_SIZE, fontweight="bold")

    collect_unique_legend(fig, axes.ravel(), ncol=2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, f"{attack_type}_{method}_anomaly_scores_line")
    plt.close(fig)


# =========================
# BAR PLOTS
# =========================
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
        )


def plot_anomaly_score_bars_for_attack(attack_type: str, summary_df: pd.DataFrame) -> None:
    conditions: List[Tuple[float, float]] = [(pct, alpha) for pct in ATTACK_PCTS for alpha in ALPHAS]
    x = np.arange(len(conditions))
    offsets = [-BAR_WIDTH / 2, BAR_WIDTH / 2]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 12), sharey=False)

    condition_labels = [condition_label(pct, alpha) for pct, alpha in conditions]

    for method_idx, method in enumerate(PROPOSED_METHODS):
        ax = axes[method_idx]
        method_df = summary_df[summary_df["Method"] == method]

        for group_idx, group_value in enumerate([0, 1]):
            means = []
            for pct, alpha in conditions:
                row = method_df[
                    (method_df["Attacker Percentage"] == pct)
                    & (method_df["Alpha"] == alpha)
                    & (method_df["Is Actual Attacker"] == group_value)
                ]
                means.append(float(row["Mean Anomaly Score"].iloc[0]) if not row.empty else np.nan)

            style = GROUP_STYLES[group_value]
            bars = ax.bar(
                x + offsets[group_idx],
                np.array(means, dtype=float),
                BAR_WIDTH,
                label=GROUP_LABELS[group_value],
                color=style["color"],
                edgecolor="black",
                linewidth=1.0,
            )
            annotate_bars(ax, bars, decimals=3)

        ax.set_ylabel(
            f"{METHOD_LABELS[method]}\nMean anomaly score",
            fontsize=FONT_SIZE,
            fontweight="bold"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            condition_labels,
            rotation=25,
            ha="right",
            fontsize=TICK_SIZE,
            fontweight="bold"
        )
        ax.set_xlabel("Attack condition", fontsize=FONT_SIZE, fontweight="bold")

        apply_common_axis_style(ax)

        if method_idx == 1:
            ax.legend(
                prop={"size": LEGEND_SIZE, "weight": "bold"},
                frameon=True,
                loc="center right",
                bbox_to_anchor=(0.98, 0.5),
            )

    fig.tight_layout()
    save_figure(fig, f"{attack_type}_anomaly_scores_bar")
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for attack_type in ATTACK_TYPES:
        print(f"\n=== Computing anomaly-score summary for {attack_type} ===")
        summary_df = compute_anomaly_summary_for_attack(attack_type)
        summary_csv = OUTPUT_DIR / f"{attack_type}_anomaly_score_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved: {summary_csv}")

        print(f"\n=== Plotting anomaly-score bar graph for {attack_type} ===")
        plot_anomaly_score_bars_for_attack(attack_type, summary_df)

        for method in PROPOSED_METHODS:
            print(f"\n=== Plotting anomaly-score line graph for {attack_type}, {method} ===")
            plot_anomaly_score_lines_for_attack_and_method(attack_type, method)

    print("\nDone.")


if __name__ == "__main__":
    main()
