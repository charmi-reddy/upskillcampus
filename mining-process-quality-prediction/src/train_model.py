"""
train_model.py
==============
Steps 6-9 of the pipeline:

    Feature Preparation -> Train/Test Split -> Regression Models -> Prediction

Run:

    python src/train_model.py                 # both experiments + optional forecasting
    python src/train_model.py --no-forecast   # skip the optional forecasting experiment

Two experiments are run, because the project brief points out that
'% Iron Concentrate' is measured in the SAME hourly lab assay as the target and is
almost perfectly anti-correlated with it:

    Experiment A  "with_iron"     -> all process variables + % Iron Concentrate
    Experiment B  "without_iron"  -> process variables ONLY  (the honest, useful model)

Outputs:
    outputs/validation_metrics.csv        metrics for every model in both experiments
    outputs/feature_importance.csv        Random-Forest importances (both experiments)
    outputs/plots/feature_importance.png
    outputs/models/*.joblib               trained models + the feature list used
    outputs/forecast_metrics.csv          OPTIONAL forecasting experiment (t+1/2/4 h)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_preprocessing import (LEAKY_COL, OUTPUT_DIR, PLOTS_DIR, PROJECT_ROOT,
                                find_target_column, load_processed_data)

MODELS_DIR = OUTPUT_DIR / "models"
TEST_SIZE = 0.20          # last 20% of the timeline is the test set
RANDOM_STATE = 42

# Time features kept as model inputs.
#   hour / day_of_week  -> genuine cyclical operating patterns (shift changes etc.)
#   day / month         -> DELIBERATELY EXCLUDED. With a chronological split the test
#                          months never occur in training, so a tree would simply
#                          extrapolate on an unseen category. They stay in
#                          processed_data.csv for EDA only.
TIME_FEATURES_USED = ["hour", "day_of_week"]
TIME_FEATURES_DROPPED = ["day", "month"]


# --------------------------------------------------------------------------------------
# 6. FEATURE PREPARATION
# --------------------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame, include_iron: bool) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split the processed table into X (inputs) and y (% Silica Concentrate).

    The target is NEVER an input. '% Iron Concentrate' is included only when
    include_iron=True, so we can measure exactly how much the model leans on it.
    """
    target = find_target_column(df)

    drop_cols = ["date", target] + TIME_FEATURES_DROPPED
    if not include_iron:
        drop_cols.append(LEAKY_COL)

    features = [c for c in df.columns if c not in drop_cols]
    X = df[features].copy()
    y = df[target].copy()
    return X, y, features


# --------------------------------------------------------------------------------------
# 7. CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------------------------------------------
def chronological_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series,
                        test_size: float = TEST_SIZE):
    """
    Time-ordered split: the EARLIEST 80% trains, the LATEST 20% tests.

    We do NOT use train_test_split(shuffle=True). Randomly shuffling a time series
    would put readings from (say) 14:00 in training and 14:20 of the same hour in
    testing. Neighbouring samples are nearly identical, so the model would be scored
    on data it has effectively already seen -> optimistic, meaningless results.
    A forward-in-time split mirrors real deployment: fit on the past, predict the future.
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    d_train, d_test = dates.iloc[:split_idx], dates.iloc[split_idx:]

    print(f"  Train: {len(X_train):>6,} rows  {d_train.min()}  ->  {d_train.max()}")
    print(f"  Test : {len(X_test):>6,} rows  {d_test.min()}  ->  {d_test.max()}")
    return X_train, X_test, y_train, y_test, d_train, d_test


# --------------------------------------------------------------------------------------
# 8. MODELS
# --------------------------------------------------------------------------------------
def get_models() -> dict:
    """
    Three regressors, from simplest to most flexible.

    Linear Regression  - the baseline. Assumes silica is a weighted sum of the
                         inputs. Wrapped in a StandardScaler pipeline so the
                         coefficients of variables with different units are comparable.
    Random Forest      - many decision trees on bootstrapped samples, averaged.
                         Captures non-linear effects and variable interactions
                         (flotation chemistry is not linear) and gives us feature
                         importances for free.
    Gradient Boosting  - trees built sequentially, each correcting the previous
                         errors. Included as the 'basic improvement' comparison.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and R^2 - the three standard regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def run_experiment(df: pd.DataFrame, include_iron: bool, label: str) -> tuple[pd.DataFrame, dict]:
    """Train all three models for one feature set and return their test metrics."""
    print("\n" + "=" * 78)
    print(f" EXPERIMENT: {label}   ({'WITH' if include_iron else 'WITHOUT'} {LEAKY_COL})")
    print("=" * 78)

    X, y, features = build_feature_matrix(df, include_iron)
    print(f"  {len(features)} input features, target = % Silica Concentrate")
    X_train, X_test, y_train, y_test, _, _ = chronological_split(X, y, df["date"])

    rows, fitted = [], {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        m_test = regression_metrics(y_test, model.predict(X_test))
        m_train = regression_metrics(y_train, model.predict(X_train))
        rows.append({
            "Experiment": label,
            "Model": name,
            "MAE": round(m_test["MAE"], 4),
            "RMSE": round(m_test["RMSE"], 4),
            "R2": round(m_test["R2"], 4),
            "Train_R2": round(m_train["R2"], 4),
            "N_Features": len(features),
        })
        fitted[name] = model
        print(f"  {name:<20} MAE={m_test['MAE']:.4f}  RMSE={m_test['RMSE']:.4f}  "
              f"R2={m_test['R2']:+.4f}   (train R2={m_train['R2']:+.4f})")

    results = pd.DataFrame(rows)

    # Persist every model plus the exact feature list it expects.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        fname = f"{label}__{name.lower().replace(' ', '_')}.joblib"
        joblib.dump({"model": model, "features": features,
                     "experiment": label, "model_name": name},
                    MODELS_DIR / fname)

    return results, {"fitted": fitted, "features": features,
                     "X_train": X_train, "y_train": y_train,
                     "X_test": X_test, "y_test": y_test}


# --------------------------------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------------------------------------------
def feature_importance_report(bundles: dict) -> pd.DataFrame:
    """Random-Forest impurity-based importances for both experiments + a bar chart."""
    print("\n" + "=" * 78)
    print(" FEATURE IMPORTANCE (Random Forest)")
    print("=" * 78)

    frames = []
    for label, b in bundles.items():
        rf = b["fitted"]["Random Forest"]
        imp = (pd.DataFrame({"Feature": b["features"], "Importance": rf.feature_importances_})
                 .sort_values("Importance", ascending=False))
        imp.insert(0, "Experiment", label)
        frames.append(imp)
        print(f"\n  Top 10 - {label}:")
        for _, r in imp.head(10).iterrows():
            print(f"    {r['Importance']:.4f}  {r['Feature']}")

    all_imp = pd.concat(frames, ignore_index=True)
    all_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Side-by-side bar chart of the top 12 in each experiment.
    fig, axes = plt.subplots(1, len(bundles), figsize=(7 * len(bundles), 7))
    axes = np.atleast_1d(axes)
    for ax, (label, _) in zip(axes, bundles.items()):
        top = all_imp[all_imp["Experiment"] == label].head(12).iloc[::-1]
        ax.barh(top["Feature"], top["Importance"], color="#2a6f97")
        ax.set_title(f"Random Forest importance\n({label})", fontsize=11)
        ax.set_xlabel("relative importance")
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)
    print(f"\n[save] {PLOTS_DIR / 'feature_importance.png'}")
    return all_imp


# --------------------------------------------------------------------------------------
# OPTIONAL EXTENSION: simple forecasting experiment
# --------------------------------------------------------------------------------------
def forecasting_experiment(df: pd.DataFrame, horizons=(1, 2, 4)) -> pd.DataFrame:
    """
    *** OPTIONAL / EXPLORATORY - not part of the core deliverable. ***

    Question from the brief: "how many hours ahead can % Silica be predicted?"

    Setup: using only information available at time t (process variables + a few
    lags of past silica readings), predict the silica measured at t + h hours.
    % Iron Concentrate is EXCLUDED - at time t the future assay does not exist yet.
    A 'persistence' baseline (assume silica stays at its current value) is included,
    because a forecast is only useful if it beats simply doing nothing.
    """
    print("\n" + "=" * 78)
    print(" OPTIONAL FORECASTING EXPERIMENT (predicting h hours ahead)")
    print("=" * 78)

    target = find_target_column(df)
    step = df["date"].diff().median()
    if step > pd.Timedelta("2h"):
        print(f"  Skipped: median sampling step is {step}, too coarse for hourly horizons.")
        return pd.DataFrame()

    work = df.copy()
    # Lag features: the plant operator genuinely knows these at time t.
    for lag in (1, 2, 3):
        work[f"silica_lag_{lag}"] = work[target].shift(lag)
    work["silica_roll_mean_3"] = work[target].shift(1).rolling(3).mean()

    rows = []
    for h in horizons:
        d = work.copy()
        d["y_future"] = d[target].shift(-h)          # the value we want to forecast
        # Guard against gaps: only keep rows where t+h really is h hours later.
        d["date_future"] = d["date"].shift(-h)
        valid = (d["date_future"] - d["date"]) == pd.Timedelta(hours=h)
        d = d[valid].dropna().reset_index(drop=True)
        if len(d) < 200:
            print(f"  h={h}h: only {len(d)} usable rows - skipped.")
            continue

        drop = ["date", "date_future", "y_future", target, LEAKY_COL] + TIME_FEATURES_DROPPED
        feats = [c for c in d.columns if c not in drop]
        X, y = d[feats], d["y_future"]
        split = int(len(d) * (1 - TEST_SIZE))
        Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(Xtr, ytr)
        m = regression_metrics(yte, rf.predict(Xte))

        # Persistence baseline: predict "silica will be what it is right now".
        base = regression_metrics(yte, d[target].iloc[split:])

        rows.append({"Horizon_hours": h, "Model": "Random Forest (+lags)",
                     "MAE": round(m["MAE"], 4), "RMSE": round(m["RMSE"], 4),
                     "R2": round(m["R2"], 4), "N_test": len(yte)})
        rows.append({"Horizon_hours": h, "Model": "Persistence baseline",
                     "MAE": round(base["MAE"], 4), "RMSE": round(base["RMSE"], 4),
                     "R2": round(base["R2"], 4), "N_test": len(yte)})
        print(f"  h={h}h  RF: MAE={m['MAE']:.4f} R2={m['R2']:+.4f}   |   "
              f"persistence: MAE={base['MAE']:.4f} R2={base['R2']:+.4f}")

    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(OUTPUT_DIR / "forecast_metrics.csv", index=False)
        print(f"[save] {OUTPUT_DIR / 'forecast_metrics.csv'}")
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def main(run_forecast: bool = True) -> pd.DataFrame:
    df = load_processed_data()
    print(f"[data] Loaded processed dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    res_a, bundle_a = run_experiment(df, include_iron=True, label="with_iron")
    res_b, bundle_b = run_experiment(df, include_iron=False, label="without_iron")
    metrics = pd.concat([res_a, res_b], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
    print(f"\n[save] {OUTPUT_DIR / 'validation_metrics.csv'}")

    bundles = {"with_iron": bundle_a, "without_iron": bundle_b}
    feature_importance_report(bundles)

    # Best model of the realistic experiment (process variables only) = deployed model.
    best_b = res_b.sort_values("R2", ascending=False).iloc[0]
    selection = {
        "selected_experiment": "without_iron",
        "selected_model": best_b["Model"],
        "reason": ("Chosen from the without_iron experiment because % Iron Concentrate "
                   "comes from the same hourly lab assay as the target and would not be "
                   "available when a prediction is actually needed."),
        "test_MAE": float(best_b["MAE"]),
        "test_RMSE": float(best_b["RMSE"]),
        "test_R2": float(best_b["R2"]),
    }
    (OUTPUT_DIR / "selected_model.json").write_text(json.dumps(selection, indent=2))
    print(f"\n[select] Best deployable model: {best_b['Model']} "
          f"(R2={best_b['R2']:.4f}, MAE={best_b['MAE']:.4f})")

    if run_forecast:
        forecasting_experiment(df)

    print("\n" + "=" * 78)
    print(" MODEL COMPARISON")
    print("=" * 78)
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the silica-prediction models.")
    ap.add_argument("--no-forecast", action="store_true",
                    help="skip the optional forecasting experiment")
    a = ap.parse_args()
    main(run_forecast=not a.no_forecast)
