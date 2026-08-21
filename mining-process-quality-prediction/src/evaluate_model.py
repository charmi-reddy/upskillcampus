"""
evaluate_model.py
=================
Steps 10-11 of the pipeline:  Evaluation -> Prediction Analysis

Run AFTER train_model.py:

    python src/evaluate_model.py

It reloads the models saved in outputs/models/, re-creates the same chronological
test set, and writes:

    outputs/predictions.csv                 timestamp, actual, predicted, abs error
    outputs/model_comparison.md             the markdown comparison table
    outputs/plots/actual_vs_predicted.png   scatter + time-series of test predictions
    outputs/plots/residual_analysis.png     residual diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_preprocessing import OUTPUT_DIR, PLOTS_DIR, find_target_column, load_processed_data
from train_model import (MODELS_DIR, TEST_SIZE, build_feature_matrix,
                         chronological_split, regression_metrics)


def load_selected_model() -> dict:
    """Load the model chosen by train_model.py (best of the without-iron experiment)."""
    sel_path = OUTPUT_DIR / "selected_model.json"
    if not sel_path.exists():
        raise FileNotFoundError(
            f"{sel_path} not found. Run `python src/train_model.py` first."
        )
    sel = json.loads(sel_path.read_text())
    fname = (f"{sel['selected_experiment']}__"
             f"{sel['selected_model'].lower().replace(' ', '_')}.joblib")
    path = MODELS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Trained model {path} missing. Re-run train_model.py.")
    bundle = joblib.load(path)
    bundle["selection"] = sel
    return bundle


def build_predictions(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict on the held-out (latest 20%) test window and tabulate the errors."""
    target = find_target_column(df)
    include_iron = bundle["experiment"] == "with_iron"
    X, y, _ = build_feature_matrix(df, include_iron=include_iron)
    X = X[bundle["features"]]  # guarantee identical column order to training

    _, X_test, _, y_test, _, d_test = chronological_split(X, y, df["date"], TEST_SIZE)
    y_pred = bundle["model"].predict(X_test)

    preds = pd.DataFrame({
        "Timestamp": d_test.values,
        "Actual_Silica": y_test.values,
        "Predicted_Silica": np.round(y_pred, 4),
    })
    preds["Absolute_Error"] = (preds["Actual_Silica"] - preds["Predicted_Silica"]).abs().round(4)
    preds["Residual"] = (preds["Actual_Silica"] - preds["Predicted_Silica"]).round(4)
    return preds


def plot_actual_vs_predicted(preds: pd.DataFrame, model_name: str) -> None:
    """Scatter (agreement) + time series (behaviour over the test window)."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.scatter(preds["Actual_Silica"], preds["Predicted_Silica"],
               s=12, alpha=0.45, color="#2a6f97", label="test predictions")
    lo = float(min(preds["Actual_Silica"].min(), preds["Predicted_Silica"].min()))
    hi = float(max(preds["Actual_Silica"].max(), preds["Predicted_Silica"].max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect prediction")
    ax.set_xlabel("Actual % Silica Concentrate")
    ax.set_ylabel("Predicted % Silica Concentrate")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.legend()

    ax = axes[1]
    ax.plot(preds["Timestamp"], preds["Actual_Silica"], lw=1.0,
            color="#023047", label="actual")
    ax.plot(preds["Timestamp"], preds["Predicted_Silica"], lw=1.0,
            color="#e85d04", alpha=0.85, label="predicted")
    ax.set_xlabel("Timestamp (held-out test period)")
    ax.set_ylabel("% Silica Concentrate")
    ax.set_title("Test-period time series")
    ax.legend()
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close(fig)
    print(f"[save] {PLOTS_DIR / 'actual_vs_predicted.png'}")


def plot_residuals(preds: pd.DataFrame, model_name: str) -> None:
    """Three standard residual diagnostics."""
    resid = preds["Residual"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].scatter(preds["Predicted_Silica"], resid, s=12, alpha=0.45, color="#2a6f97")
    axes[0].axhline(0, color="crimson", ls="--")
    axes[0].set_xlabel("Predicted % Silica")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")

    axes[1].hist(resid, bins=45, color="#2a6f97", edgecolor="white")
    axes[1].axvline(0, color="crimson", ls="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_title(f"Residual distribution (mean = {resid.mean():+.3f})")

    axes[2].plot(preds["Timestamp"], resid, lw=0.8, color="#2a6f97")
    axes[2].axhline(0, color="crimson", ls="--")
    axes[2].set_xlabel("Timestamp")
    axes[2].set_ylabel("Residual")
    axes[2].set_title("Residuals over time")

    fig.suptitle(f"Residual analysis - {model_name}")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "residual_analysis.png", dpi=120)
    plt.close(fig)
    print(f"[save] {PLOTS_DIR / 'residual_analysis.png'}")


def write_comparison_table() -> str:
    """Render outputs/validation_metrics.csv as a markdown table."""
    metrics = pd.read_csv(OUTPUT_DIR / "validation_metrics.csv")
    lines = ["| Experiment | Model | MAE | RMSE | R² |",
             "|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        lines.append(f"| {r['Experiment']} | {r['Model']} | {r['MAE']:.4f} | "
                     f"{r['RMSE']:.4f} | {r['R2']:.4f} |")
    table = "\n".join(lines)
    (OUTPUT_DIR / "model_comparison.md").write_text(table + "\n")
    return table


def main() -> pd.DataFrame:
    df = load_processed_data()
    bundle = load_selected_model()
    name = bundle["model_name"]
    print("=" * 78)
    print(f" EVALUATION - selected model: {name}  [{bundle['experiment']}]")
    print("=" * 78)

    preds = build_predictions(df, bundle)
    m = regression_metrics(preds["Actual_Silica"], preds["Predicted_Silica"])

    print(f"\n  MAE  = {m['MAE']:.4f}  (average error, in % silica)")
    print(f"  RMSE = {m['RMSE']:.4f}  (penalises large misses more heavily)")
    print(f"  R2   = {m['R2']:+.4f}  (share of variance explained)")
    print(f"  Mean |error| as % of mean silica: "
          f"{100 * m['MAE'] / preds['Actual_Silica'].mean():.1f}%")
    print(f"  Worst absolute error: {preds['Absolute_Error'].max():.3f}")
    print(f"  Errors within 0.5 %-points: "
          f"{100 * (preds['Absolute_Error'] <= 0.5).mean():.1f}% of test rows")

    preds.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    print(f"\n[save] {OUTPUT_DIR / 'predictions.csv'}  ({len(preds):,} rows)")

    plot_actual_vs_predicted(preds, name)
    plot_residuals(preds, name)

    print("\n" + write_comparison_table())
    return preds


if __name__ == "__main__":
    main()
