"""
data_preprocessing.py
---------------------
Data loading, cleaning and feature preparation for the NASA C-MAPSS
turbofan engine dataset (subset FD001).

Steps implemented here:
  1. Load the raw .txt files with pandas
  2. Assign meaningful column names
  3. Check for missing values
  4. Check sensor variance and drop constant / near-constant columns
  5. Create the RUL target for the training set
  6. Prepare the test set (last cycle of every test engine + true RUL)
  7. Save basic EDA plots into outputs/plots/

Run standalone with:  python src/data_preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# Paths - change these if your dataset lives somewhere else
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"

# Where to get the files if they are missing
DOWNLOAD_URL = (
    "https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6"
)

# ----------------------------------------------------------------------
# Column names (same layout for every C-MAPSS .txt file)
# ----------------------------------------------------------------------
COLUMN_NAMES = (
    ["unit", "cycle", "setting1", "setting2", "setting3"]
    + [f"sensor{i}" for i in range(1, 22)]   # sensor1 ... sensor21
)
SETTING_COLUMNS = ["setting1", "setting2", "setting3"]
SENSOR_COLUMNS = [f"sensor{i}" for i in range(1, 22)]

# Columns with variance below this threshold are considered constant /
# uninformative and are dropped.
VARIANCE_THRESHOLD = 0.01

# Optional: piecewise-linear RUL cap (standard C-MAPSS practice, Heimes 2008).
# RUL is first computed as  max_cycle - cycle  (the formula in the project
# brief), then clipped at RUL_CAP. Early in an engine's life it is healthy
# and the exact failure time is not knowable, so capping the target at a
# constant value (e.g. 125 for FD001) makes the task easier and improves
# accuracy. Set RUL_CAP = None to use the pure linear target.
RUL_CAP = None

# Nice default style for all plots
sns.set_theme(style="whitegrid", context="notebook")


# ----------------------------------------------------------------------
# 1. Data loading
# ----------------------------------------------------------------------
def _load_raw(path: Path, name: str) -> pd.DataFrame:
    """
    Load one raw C-MAPSS .txt file and return a DataFrame with proper
    column names. Raises a helpful error if the file is missing.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"\nDataset file '{name}' not found at:\n    {path}\n\n"
            f"Please download the C-MAPSS data from:\n    {DOWNLOAD_URL}\n"
            f"and place train_FD001.txt, test_FD001.txt and RUL_FD001.txt "
            f"inside the folder:\n    {DATA_DIR}\n"
        )
    # The files are space-separated with a trailing space on every line,
    # so we read them with sep=r"\s+" (one or more whitespace characters).
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df


def load_train_data(path: Path = TRAIN_PATH) -> pd.DataFrame:
    """Load the training set (engines that ran until failure)."""
    return _load_raw(path, "train_FD001.txt")


def load_test_data(path: Path = TEST_PATH) -> pd.DataFrame:
    """Load the test set (engines truncated before failure)."""
    return _load_raw(path, "test_FD001.txt")


def load_test_rul(path: Path = RUL_PATH) -> pd.DataFrame:
    """
    Load the true RUL of the test engines. The file contains one value
    per line, in engine order (line 1 -> engine 1, line 2 -> engine 2, ...).
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"\nDataset file 'RUL_FD001.txt' not found at:\n    {path}\n\n"
            f"Please download the C-MAPSS data from:\n    {DOWNLOAD_URL}\n"
            f"and place it inside the folder:\n    {DATA_DIR}\n"
        )
    rul = pd.read_csv(path, sep=r"\s+", header=None, names=["true_rul"], engine="python")
    rul["unit"] = np.arange(1, len(rul) + 1)
    return rul[["unit", "true_rul"]]


# ----------------------------------------------------------------------
# 2. RUL target generation (training set)
# ----------------------------------------------------------------------
def add_rul_column(train_df: pd.DataFrame, cap_rul=RUL_CAP) -> pd.DataFrame:
    """
    Create the RUL target for every row of the training set.

        RUL = (maximum cycle of the engine) - (current cycle)

    The maximum cycle of a training engine is the cycle at which it
    failed, so a row recorded 10 cycles before failure gets RUL = 10.

    If cap_rul is given, the target is additionally clipped at that value
    (piecewise-linear degradation): RUL = min(RUL, cap_rul).
    """
    df = train_df.copy()
    max_cycle_per_engine = df.groupby("unit")["cycle"].max()
    df["rul"] = df["unit"].map(max_cycle_per_engine) - df["cycle"]
    if cap_rul is not None:
        df["rul"] = df["rul"].clip(upper=cap_rul)
    return df


# ----------------------------------------------------------------------
# 3. Missing values and variance checks
# ----------------------------------------------------------------------
def report_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a Series with the number of missing values per column (only non-zero)."""
    missing = df.isna().sum()
    return missing[missing > 0]


def find_low_variance_columns(df, columns, threshold=VARIANCE_THRESHOLD):
    """
    Return (dropped_columns, variances) where dropped_columns are the
    columns whose variance is below `threshold`. A constant column has
    variance 0 and carries no information for the model.
    """
    variances = df[columns].var()
    dropped = list(variances[variances < threshold].index)
    return dropped, variances


# ----------------------------------------------------------------------
# 4. Feature selection
# ----------------------------------------------------------------------
def select_features(train_df: pd.DataFrame, threshold=VARIANCE_THRESHOLD):
    """
    Choose the feature columns used by the model.

    - 'cycle' (the age of the engine) is always kept: it is a strong and
      legitimate signal.
    - Operational settings and sensors with (near-)zero variance carry no
      information, so they are removed.
    """
    candidates = SETTING_COLUMNS + SENSOR_COLUMNS
    dropped, _ = find_low_variance_columns(train_df, candidates, threshold)
    kept = [c for c in candidates if c not in dropped]
    features = ["cycle"] + kept
    return features, dropped


# ----------------------------------------------------------------------
# 5. Test set preparation
# ----------------------------------------------------------------------
def get_test_final_cycles(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the LAST recorded cycle of every test engine - that is the
    single moment in time at which we must predict the engine's RUL.
    """
    return test_df.groupby("unit").tail(1).reset_index(drop=True)


def prepare_train(train_df: pd.DataFrame, features):
    """Split the training DataFrame into feature matrix X and target y."""
    X = train_df[features].copy()
    y = train_df["rul"]
    return X, y


def prepare_test(test_df: pd.DataFrame, rul_df: pd.DataFrame, features):
    """
    Build the test matrix: last cycle of every test engine + its true RUL.
    Returns (X, y_true, test_final) where test_final also keeps the unit id.
    """
    final = get_test_final_cycles(test_df)
    final = final.merge(rul_df, on="unit", how="left")
    X = final[features].copy()
    y_true = final["true_rul"]
    return X, y_true, final


# ----------------------------------------------------------------------
# 6. EDA plots (simple and readable, saved into outputs/plots/)
# ----------------------------------------------------------------------
def plot_rul_distribution(train_df, save_path=None, figsize=(9, 5)):
    """Histogram of the RUL target in the training set."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(train_df["rul"], bins=40, kde=True, color="#1f77b4", ax=ax)
    ax.set_title("Distribution of Remaining Useful Life (RUL) - training set")
    ax.set_xlabel("RUL (cycles remaining)")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_sensor_trends(train_df, sensors, engine_ids=(1,), save_path=None, figsize=(12, 6)):
    """
    Plot selected sensors against the operational cycle for chosen engines.
    A visible drift (up or down) as the cycle increases shows the sensor
    is capturing the engine's degradation.
    """
    n_sensors = len(sensors)
    n_engines = len(engine_ids)
    fig, axes = plt.subplots(
        nrows=n_engines, ncols=n_sensors,
        figsize=(figsize[0], 3.2 * n_engines),
        squeeze=False,
    )
    for row, engine in enumerate(engine_ids):
        eng = train_df[train_df["unit"] == engine]
        for col, sensor in enumerate(sensors):
            ax = axes[row][col]
            sns.lineplot(data=eng, x="cycle", y=sensor, ax=ax, linewidth=1.6)
            ax.set_title(f"{sensor} - engine {engine}", fontsize=10)
            ax.set_xlabel("Cycle" if row == n_engines - 1 else "")
            ax.set_ylabel("Sensor value", fontsize=9)
    fig.suptitle("Sensor readings vs operational cycle (degradation trend)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(train_df, features, save_path=None, figsize=(10, 8)):
    """Correlation heatmap between the selected features and the RUL target."""
    corr = train_df[features + ["rul"]].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False,
                ax=ax, cbar_kws={"label": "Pearson correlation"})
    ax.set_title("Correlation between features and RUL")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Standalone run: loads everything, prints a summary and saves the plots
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 1/4 - Loading the data")
    print("=" * 70)
    train = load_train_data()
    test = load_test_data()
    rul = load_test_rul()
    print(f"Training set   : {train.shape[0]} rows x {train.shape[1]} columns "
          f"({train['unit'].nunique()} engines)")
    print(f"Test set       : {test.shape[0]} rows x {test.shape[1]} columns "
          f"({test['unit'].nunique()} engines)")
    print(f"True RUL file  : {rul.shape[0]} values (one per test engine)")
    print("\nFirst rows of the training set:")
    print(train.head().to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 2/4 - Missing values and variance check")
    print("=" * 70)
    missing = report_missing_values(train)
    print(f"Columns with missing values: {len(missing)}" +
          (f" -> {missing.to_dict()}" if len(missing) else " (none - dataset is complete)"))
    dropped, variances = find_low_variance_columns(
        train, SETTING_COLUMNS + SENSOR_COLUMNS
    )
    print(f"\nConstant / near-constant columns (variance < {VARIANCE_THRESHOLD}):")
    print(variances.loc[dropped].round(6).to_string())

    print("\n" + "=" * 70)
    print("STEP 3/4 - Generating the RUL target for the training set")
    print("=" * 70)
    train = add_rul_column(train)
    print("RUL = max cycle of the engine - current cycle"
          + (f" (clipped at {RUL_CAP})" if RUL_CAP else " (linear target)"))
    print("\nLast 5 rows of engine 1 (RUL should count down to 0):")
    print(train[train["unit"] == 1][["unit", "cycle", "rul"]].tail(5).to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 4/4 - Feature selection, EDA plots and processed files")
    print("=" * 70)
    features, dropped = select_features(train)
    print(f"Features kept ({len(features)}): {features}")
    print(f"Columns dropped ({len(dropped)}): {dropped}")

    # Save the EDA plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_rul_distribution(train, PLOTS_DIR / "rul_distribution.png")
    plot_sensor_trends(
        train,
        sensors=["sensor2", "sensor3", "sensor4", "sensor7", "sensor11", "sensor12"],
        engine_ids=(1, 50),
        save_path=PLOTS_DIR / "sensor_trends.png",
    )
    plot_correlation_heatmap(train, features, PLOTS_DIR / "correlation_heatmap.png")
    plt.close("all")
    print(f"\nEDA plots saved to {PLOTS_DIR}")

    # Save processed datasets (handy for inspection / reuse)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train = prepare_train(train, features)
    processed_train = X_train.copy()
    processed_train["rul"] = y_train
    processed_train.to_csv(OUTPUT_DIR / "train_processed.csv", index=False)

    X_test, y_test, test_final = prepare_test(test, rul, features)
    processed_test = test_final[["unit"] + features + ["true_rul"]].copy()
    processed_test.to_csv(OUTPUT_DIR / "test_processed.csv", index=False)
    print(f"Processed datasets saved to {OUTPUT_DIR}")
    print("\nPreprocessing finished. Next: python src/train_model.py")


if __name__ == "__main__":
    main()
