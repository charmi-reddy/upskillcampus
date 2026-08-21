"""
train_model.py
--------------
Trains and compares two tree-based regression models (Random Forest and
Gradient Boosting) on the processed training data, selects the better one
using an engine-wise validation split, retrains it on the full training
data and saves it as outputs/best_model.joblib.

Run standalone with:  python src/train_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Allow both `python src/train_model.py` and `python -m src.train_model`
try:
    from src import data_preprocessing as dp
except ModuleNotFoundError:
    import data_preprocessing as dp

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
MODEL_PATH = dp.OUTPUT_DIR / "best_model.joblib"
VALIDATION_METRICS_PATH = dp.OUTPUT_DIR / "validation_metrics.csv"
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.2   # 20% of the ENGINES are held out for validation


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def rmse(y_true, y_pred):
    """Root Mean Squared Error (same unit as RUL: cycles)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_model(y_true, y_pred):
    """Compute the three regression metrics used in this project."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------
# Train / validation split
# ----------------------------------------------------------------------
def split_by_unit(train_df, features, test_size=VALIDATION_FRACTION,
                  random_state=RANDOM_STATE):
    """
    Split the training data by ENGINE UNIT (not by row!).

    Splitting randomly by row would put different cycles of the SAME
    engine into both training and validation sets, which leaks information
    and makes validation scores unrealistically optimistic. Splitting by
    unit guarantees the validation engines are completely unseen.
    """
    units = train_df["unit"].unique()
    train_units, val_units = train_test_split(
        units, test_size=test_size, random_state=random_state
    )

    train_mask = train_df["unit"].isin(train_units)
    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, "rul"]
    X_val = train_df.loc[~train_mask, features]
    y_val = train_df.loc[~train_mask, "rul"]
    return X_train, X_val, y_train, y_val, train_units, val_units


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def build_models():
    """
    The two candidate models, each wrapped in a Pipeline.

    Tree-based models do not need scaled features, but StandardScaler is
    included anyway so the pipeline stays consistent if other models
    (e.g. linear regression, SVM) are swapped in later - scaling does
    not hurt the trees.
    """
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,          # number of trees
                random_state=RANDOM_STATE,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100,          # number of boosting stages
                learning_rate=0.1,
                max_depth=3,               # shallow trees -> less overfitting
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_model_comparison(metrics_df, save_path=None, figsize=(7.5, 4.5)):
    """Bar chart comparing validation MAE / RMSE of the two models."""
    fig, ax = plt.subplots(figsize=figsize)
    metrics_df.set_index("model")[["MAE", "RMSE"]].plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Validation performance - model comparison (lower is better)")
    ax.set_ylabel("Error in cycles")
    ax.set_xlabel("")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_actual_vs_predicted(y_true, y_pred, title, save_path=None, figsize=(7, 6)):
    """Scatter of predicted vs actual RUL with the ideal y = x line."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.6, s=28, label="engines")
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], "r--", label="perfect prediction (y = x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def main():
    dp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dp.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and preprocess ------------------------------------------------
    print("Loading and preprocessing the training data ...")
    train = dp.load_train_data()
    train = dp.add_rul_column(train)
    features, dropped = dp.select_features(train)
    print(f"Features used ({len(features)}): {features}")
    print(f"Dropped ({len(dropped)}): {dropped}")

    # 2. Engine-wise train / validation split -------------------------------
    X_train, X_val, y_train, y_val, train_units, val_units = split_by_unit(train, features)
    print(f"\nTraining engines   : {len(train_units)}  ({X_train.shape[0]} rows)")
    print(f"Validation engines : {len(val_units)}  ({X_val.shape[0]} rows)")

    # 3. Train both models and compare on the validation engines ------------
    print("\nTraining both models ...")
    models = build_models()
    fitted = {}
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_val)
        m = evaluate_model(y_val, pred)
        rows.append({"model": name, **m})
        print(f"  {name:<18} MAE={m['MAE']:6.2f}  RMSE={m['RMSE']:6.2f}  R2={m['R2']:.3f}")

    compare = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    compare.to_csv(VALIDATION_METRICS_PATH, index=False)
    best_name = compare.iloc[0]["model"]
    print(f"\nBest model on validation: {best_name} (lowest RMSE)")
    print(f"Comparison table saved to {VALIDATION_METRICS_PATH}")

    plot_model_comparison(compare, dp.PLOTS_DIR / "model_comparison.png")

    # 4. Retrain the best model on ALL training data and save it -------------
    print(f"\nRetraining '{best_name}' on the full training set ...")
    X_all, y_all = train[features], train["rul"]
    best_pipe = build_models()[best_name]
    best_pipe.fit(X_all, y_all)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")

    # 5. Feature importances of the best model --------------------------------
    model = best_pipe.named_steps["model"]
    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)
    print("\nTop-10 feature importances (best model):")
    print(importances.head(10).round(3).to_string())

    # 6. Validation scatter plot for the best model ---------------------------
    val_pred_best = fitted[best_name].predict(X_val)
    plot_actual_vs_predicted(
        y_val, val_pred_best,
        f"{best_name} - validation set (unseen engines)",
        dp.PLOTS_DIR / "validation_actual_vs_predicted.png",
    )
    plt.close("all")

    print("\nTraining finished. Next: python src/evaluate_model.py")


if __name__ == "__main__":
    main()
