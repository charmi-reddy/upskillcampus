"""
data_preprocessing.py
=====================
Step 1-5 of the pipeline:

    Industrial Process Data -> Data Loading -> Data Inspection -> Data Cleaning
    -> Timestamp Processing -> Exploratory Data Analysis

Run this file directly:

    python src/data_preprocessing.py
    python src/data_preprocessing.py --data-path data/my_other_file.csv

It writes:
    outputs/processed_data.csv
    outputs/plots/silica_distribution.png
    outputs/plots/silica_over_time.png
    outputs/plots/correlation_heatmap.png
    outputs/plots/feature_target_relationship.png
    outputs/plots/feature_target_correlation.png

The ORIGINAL csv in data/ is never modified - we only ever read it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: we only save .png files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------------------
# CONFIGURATION - everything a student may want to change lives here
# --------------------------------------------------------------------------------------

# Project root = the folder that contains data/, src/, outputs/ ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# >>> Change this line (or pass --data-path) to point at a different csv <<<
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "mining_process_data.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
PROCESSED_PATH = OUTPUT_DIR / "processed_data.csv"

# The target we want to predict (the lab measurement of impurity in the concentrate).
TARGET_COL = "% Silica Concentrate"

# Measured in the SAME hourly lab assay as the target and almost perfectly
# anti-correlated with it. Kept in the data, but treated as a special case in training.
LEAKY_COL = "% Iron Concentrate"

# The timestamp column in the raw Kaggle file is simply called "date".
TIMESTAMP_CANDIDATES = ["date", "Date", "datetime", "timestamp", "Timestamp", "time"]

# Time features we extract (deliberately few - only what a plant engineer would use).
TIME_FEATURES = ["hour", "day", "day_of_week", "month"]

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


# --------------------------------------------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------------------------------------------
def load_raw_data(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the flotation-plant csv into a DataFrame.

    Handles the two shapes this dataset is distributed in:
      * the raw Kaggle export, which uses a COMMA as the decimal separator
        (e.g. "55,2" instead of "55.2"), and
      * already-normalised copies that use a dot.

    Raises a clear, actionable error if the file is missing - we never invent data.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            "\n"
            "============================================================\n"
            f" DATASET NOT FOUND: {data_path}\n"
            "============================================================\n"
            " This project does NOT generate or fake data. Place the real\n"
            " flotation-plant csv in the data/ folder, e.g.:\n\n"
            f"     {PROJECT_ROOT / 'data' / 'mining_process_data.csv'}\n\n"
            " Source: Kaggle -> 'Quality Prediction in a Mining Process'\n"
            "         (MiningProcess_Flotation_Plant_Database.csv)\n"
            " Then re-run, or pass a custom path:\n"
            "     python src/data_preprocessing.py --data-path <your.csv>\n"
            "============================================================"
        )

    # Read everything as-is first; we fix dtypes ourselves in a documented step below.
    df = pd.read_csv(data_path)

    if df.empty:
        raise ValueError(f"The file {data_path} was read successfully but contains 0 rows.")

    print(f"[load] Read {data_path}  ->  {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def find_timestamp_column(df: pd.DataFrame) -> str | None:
    """Identify the date/time column by name, then by a parse-test fallback."""
    for cand in TIMESTAMP_CANDIDATES:
        if cand in df.columns:
            return cand
    # Fallback: first object column whose first 50 values parse as dates.
    for col in df.columns:
        if df[col].dtype == object:
            probe = pd.to_datetime(df[col].head(50), errors="coerce", format="mixed")
            if probe.notna().mean() > 0.9:
                return col
    return None


def find_target_column(df: pd.DataFrame) -> str:
    """
    Identify the silica-in-concentrate column even if the exact spelling differs
    (e.g. '% Silica Concentrate' vs 'Silica_Concentrate').
    """
    if TARGET_COL in df.columns:
        return TARGET_COL
    norm = {c: c.lower().replace("_", " ").replace("%", "").strip() for c in df.columns}
    for col, n in norm.items():
        if "silica" in n and "concentrate" in n:
            return col
    raise KeyError(
        "Could not find a '% Silica Concentrate' column. Columns present: "
        f"{list(df.columns)}"
    )


# --------------------------------------------------------------------------------------
# 2. DATA INSPECTION
# --------------------------------------------------------------------------------------
def inspect_data(df: pd.DataFrame) -> dict:
    """Print the 8 inspection items required by the brief and return a small summary."""
    ts_col = find_timestamp_column(df)
    target_col = find_target_column(df)

    print("\n" + "=" * 78)
    print(" DATA INSPECTION")
    print("=" * 78)

    # 1. shape
    print(f"\n1. Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. column names
    print("\n2. Columns:")
    for i, c in enumerate(df.columns, 1):
        print(f"   {i:2d}. {c}")

    # 3. dtypes
    print("\n3. Data types:")
    print(df.dtypes.to_string())

    # 4. missing values
    miss = df.isna().sum()
    print("\n4. Missing values per column:")
    if miss.sum() == 0:
        print("   None - no missing cells in the raw file.")
    else:
        print(miss[miss > 0].to_string())
        print(f"   Total missing cells: {miss.sum():,}")

    # 5. duplicates
    n_dup = int(df.duplicated().sum())
    print(f"\n5. Fully duplicated rows: {n_dup:,}")

    # 6. statistical summary
    print("\n6. Statistical summary (numeric columns):")
    print(df.describe().T.to_string())

    # 7. timestamp column
    print(f"\n7. Timestamp column detected: {ts_col!r}")
    if ts_col:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", format="mixed")
        print(f"   Range: {parsed.min()}  ->  {parsed.max()}")
        deltas = parsed.sort_values().diff().dropna()
        if not deltas.empty:
            print(f"   Median sampling interval: {deltas.median()}")
            print(f"   Distinct intervals (top 5):\n{deltas.value_counts().head().to_string()}")

    # 8. target column
    print(f"\n8. Target column detected: {target_col!r}")
    print(f"   min={df[target_col].min():.3f}  mean={df[target_col].mean():.3f}  "
          f"max={df[target_col].max():.3f}")

    print("\n" + "=" * 78 + "\n")
    return {"timestamp_col": ts_col, "target_col": target_col,
            "n_rows": len(df), "n_duplicates": n_dup}


# --------------------------------------------------------------------------------------
# 3 + 4. DATA CLEANING AND TIMESTAMP PROCESSING
# --------------------------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a column to numeric.
    The raw Kaggle export stores numbers as strings with a comma decimal mark
    ("55,2"). We swap ',' for '.' before converting - a common, explainable fix.
    """
    if series.dtype != object:
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def clean_data(df: pd.DataFrame, resample_hourly: str = "auto") -> pd.DataFrame:
    """
    Simple, explainable cleaning. Each step prints what it did and why.

    Steps
    -----
    1. Parse the timestamp and sort chronologically (time order matters for a split).
    2. Drop exactly-duplicated rows (same timestamp AND same values = logging glitch).
    3. Force every non-timestamp column to numeric (fixes comma decimals / stray text).
    4. Drop impossible values: negatives in flows/levels, percentages outside 0-100.
    5. Optionally down-sample 20-second data to hourly means, because the LAB TARGET
       is only measured once per hour - this is the data-alignment step.
    6. Fill remaining gaps with time interpolation (correct for slow-moving sensors),
       then drop any row that still has no target value (never impute the target).
    7. Add hour / day / day_of_week / month features.
    """
    print("=" * 78)
    print(" DATA CLEANING")
    print("=" * 78)

    out = df.copy()  # work on a copy - the raw file/frame stays untouched
    ts_col = find_timestamp_column(out)
    target_col = find_target_column(out)

    # --- 1. timestamp -> datetime, sorted ------------------------------------------
    if ts_col is None:
        raise KeyError("No timestamp column found; cannot preserve chronology.")
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", format="mixed")
    n_bad_ts = int(out[ts_col].isna().sum())
    if n_bad_ts:
        out = out.dropna(subset=[ts_col])
        print(f"[clean] Dropped {n_bad_ts:,} rows with an unparseable timestamp.")
    out = out.sort_values(ts_col).reset_index(drop=True)
    out = out.rename(columns={ts_col: "date"})
    print(f"[clean] Timestamp parsed and sorted: {out['date'].min()} -> {out['date'].max()}")

    # --- 2. duplicates --------------------------------------------------------------
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    print(f"[clean] Removed {before - len(out):,} exactly-duplicated rows "
          f"({len(out):,} remain).")

    # --- 3. numeric coercion --------------------------------------------------------
    value_cols = [c for c in out.columns if c != "date"]
    converted = []
    for c in value_cols:
        was_object = out[c].dtype == object
        out[c] = _to_numeric(out[c])
        if was_object:
            converted.append(c)
    if converted:
        print(f"[clean] Converted {len(converted)} text column(s) to numeric "
              f"(comma decimals): {converted[:4]}{'...' if len(converted) > 4 else ''}")
    else:
        print("[clean] All value columns were already numeric.")

    # --- 4. invalid / impossible values --------------------------------------------
    pct_cols = [c for c in value_cols if c.strip().startswith("%")]
    n_invalid = 0
    for c in pct_cols:  # a percentage below 0 or above 100 is physically impossible
        bad = (out[c] < 0) | (out[c] > 100)
        n_invalid += int(bad.sum())
        out.loc[bad, c] = np.nan
    flow_level_cols = [c for c in value_cols
                       if ("Flow" in c or "Level" in c or "Density" in c)]
    for c in flow_level_cols:  # a negative flow rate or tank level cannot happen
        bad = out[c] < 0
        n_invalid += int(bad.sum())
        out.loc[bad, c] = np.nan
    print(f"[clean] Blanked {n_invalid:,} physically impossible value(s) "
          f"(out-of-range % or negative flow/level) -> handled as missing.")

    # --- 5. frequency alignment -----------------------------------------------------
    # Different variables are sampled at different rates (process = every 20 s,
    # lab quality = every hour). Averaging process signals over the hour puts every
    # variable on ONE common clock, which is the frequency the target actually has.
    deltas = out["date"].diff().dropna()
    median_step = deltas.median() if not deltas.empty else pd.Timedelta("1h")
    do_resample = (resample_hourly == "always") or (
        resample_hourly == "auto" and median_step < pd.Timedelta("30min")
    )
    if do_resample:
        before = len(out)
        out = (out.set_index("date")
                  .resample("1h").mean()      # mean of ~180 process samples per hour
                  .dropna(how="all")
                  .reset_index())
        print(f"[clean] Median raw step was {median_step} (sub-hourly). Resampled to "
              f"hourly means: {before:,} -> {len(out):,} rows (aligns process data "
              f"with the hourly lab target).")
    else:
        print(f"[clean] Median step is {median_step}; data already on one clock - "
              f"no resampling applied.")

    # --- 6. missing values ----------------------------------------------------------
    n_missing = int(out[[c for c in out.columns if c != 'date']].isna().sum().sum())
    if n_missing:
        # Time interpolation is appropriate here: these are continuous physical
        # signals sampled in order, so a gap is best estimated from its neighbours.
        out = out.set_index("date")
        num_cols = out.select_dtypes(include=[np.number]).columns
        out[num_cols] = out[num_cols].interpolate(method="time", limit_direction="both")
        out = out.reset_index()
        print(f"[clean] Interpolated {n_missing:,} missing value(s) over time.")
    else:
        print("[clean] No missing values to fill.")

    # Never impute the thing we are trying to predict.
    before = len(out)
    out = out.dropna(subset=[target_col]).reset_index(drop=True)
    if before - len(out):
        print(f"[clean] Dropped {before - len(out):,} row(s) with no target measurement.")

    # --- 7. time features -----------------------------------------------------------
    out["hour"] = out["date"].dt.hour
    out["day"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek  # Monday = 0
    out["month"] = out["date"].dt.month
    print(f"[clean] Added time features: {TIME_FEATURES}")

    print(f"[clean] FINAL cleaned shape: {out.shape[0]:,} rows x {out.shape[1]} columns")
    print("=" * 78 + "\n")
    return out


# --------------------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, plots_dir: Path = PLOTS_DIR) -> pd.Series:
    """Create the five required EDA plots. Returns feature/target correlations."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    target = find_target_column(df)
    print("=" * 78)
    print(" EXPLORATORY DATA ANALYSIS")
    print("=" * 78)

    # (1) Target distribution -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target], bins=50, kde=True, color="#2a6f97", ax=ax)
    ax.axvline(df[target].mean(), color="crimson", ls="--",
               label=f"mean = {df[target].mean():.2f}%")
    ax.set_title("Distribution of % Silica in Iron Ore Concentrate")
    ax.set_xlabel("% Silica Concentrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "silica_distribution.png", dpi=120)
    plt.close(fig)

    # (2) Target over time ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df["date"], df[target], lw=0.7, color="#2a6f97", label="hourly value")
    ax.plot(df["date"], df[target].rolling(24, min_periods=1).mean(),
            lw=1.8, color="crimson", label="24-point rolling mean")
    ax.set_title("% Silica Concentrate over time (chronological)")
    ax.set_xlabel("Date")
    ax.set_ylabel("% Silica Concentrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "silica_over_time.png", dpi=120)
    plt.close(fig)

    # (3) Correlation heatmap -------------------------------------------------------
    num = df.drop(columns=TIME_FEATURES, errors="ignore").select_dtypes(include=[np.number])
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 6}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation heatmap - flotation process variables")
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # (5) Correlation of every feature with the target ------------------------------
    corr_t = corr[target].drop(target).sort_values()
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#c1121f" if v < 0 else "#2a6f97" for v in corr_t.values]
    ax.barh(corr_t.index, corr_t.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"Pearson correlation of each variable with {target}")
    ax.set_xlabel("correlation")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_target_correlation.png", dpi=120)
    plt.close(fig)

    # (4) The strongest process drivers vs the target -------------------------------
    # Pick the 4 most correlated PROCESS variables (exclude the lab-assay partner).
    ranked = corr_t.drop(labels=[LEAKY_COL], errors="ignore").abs().sort_values(ascending=False)
    top4 = ranked.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sample = df.sample(min(3000, len(df)), random_state=42)  # keep the scatter readable
    for ax, col in zip(axes.ravel(), top4):
        ax.scatter(sample[col], sample[target], s=6, alpha=0.35, color="#2a6f97")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        ax.set_title(f"r = {corr.loc[col, target]:+.2f}", fontsize=10)
    fig.suptitle("Most influential process variables vs % Silica Concentrate")
    fig.tight_layout()
    fig.savefig(plots_dir / "feature_target_relationship.png", dpi=120)
    plt.close(fig)

    print("Top correlations with the target:")
    print(corr_t.reindex(corr_t.abs().sort_values(ascending=False).index).head(8).to_string())
    print(f"\n[eda] 5 plots written to {plots_dir}")
    print("=" * 78 + "\n")
    return corr_t


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def build_processed_dataset(data_path: Path | str = DEFAULT_DATA_PATH,
                            resample_hourly: str = "auto",
                            do_eda: bool = True) -> pd.DataFrame:
    """Load -> inspect -> clean -> EDA -> save outputs/processed_data.csv."""
    raw = load_raw_data(data_path)
    inspect_data(raw)
    clean = clean_data(raw, resample_hourly=resample_hourly)
    if do_eda:
        run_eda(clean)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_PATH, index=False)
    print(f"[save] Processed dataset -> {PROCESSED_PATH}")
    return clean


def load_processed_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Helper used by train_model.py / evaluate_model.py."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/data_preprocessing.py` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean the flotation-plant dataset.")
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH),
                   help="path to the raw csv (default: data/mining_process_data.csv)")
    p.add_argument("--resample-hourly", choices=["auto", "always", "never"], default="auto",
                   help="align sub-hourly process data to the hourly lab target")
    p.add_argument("--no-eda", action="store_true", help="skip the EDA plots")
    return p.parse_args()


if __name__ == "__main__":
    args = _cli()
    try:
        build_processed_dataset(args.data_path, args.resample_hourly, not args.no_eda)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
