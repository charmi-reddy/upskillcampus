# Quality Prediction in an Iron Ore Mining Process

Predicting the **% Silica in iron ore concentrate** from froth-flotation plant process data.

> Internship demonstration project. Plain pandas + scikit-learn, no web UI, no API, no database, no Docker, no deep learning. Every number in this README was produced by actually running the code in `src/`.

---

## 1. Problem Statement

A froth-flotation plant concentrates iron ore by floating silica (sand) impurity away from the iron. The amount of silica left in the final concentrate is the plant's key quality number — but it is measured in a **laboratory, only once per hour**. By the time a bad result arrives, an hour of ore has already been processed at the wrong settings.

**Task:** learn the relationship between the process measurements an operator can see in real time (reagent flows, pulp properties, column air flows and levels) and the `% Silica Concentrate` the lab will report.

## 2. Industrial Context

In reverse cationic flotation of iron ore:

- **Starch** is added as a depressant, keeping iron minerals in the pulp.
- **Amina** (an amine collector) makes silica particles hydrophobic so they attach to air bubbles and float off as froth.
- **Air flow** through each of 7 flotation columns controls how many bubbles are available to carry silica away.
- **Column level** (froth depth) controls how selective that separation is — too shallow and iron is lost with the froth, too deep and silica stays behind.
- **Ore pulp pH, density and flow** set the chemical environment the whole reaction runs in.

Silica is a penalty element in steelmaking: excess silica means more slag, more flux, more energy. Predicting it early lets engineers correct reagent dosing before an off-spec hour is produced, reducing both impurity and iron lost to tailings.

## 3. Objective

1. Build a regression model mapping process variables → `% Silica Concentrate`.
2. Compare a linear baseline against tree-based models.
3. Test the brief's specific question: **can silica be predicted _without_ `% Iron Concentrate`**, which the project description flags as highly correlated with the target?
4. Identify which process measurements matter most.

## 4. Dataset

Real operating data from an iron ore flotation plant, **11 March 2017 → 9 September 2017**.

| Property | Value |
|---|---|
| File used | `data/mining_process_data.csv` |
| Rows | 4,069 (hourly) |
| Columns | 24 (1 timestamp + 22 process/quality + target) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Target mean / std | 2.331 % / 1.126 % |
| Target range | 0.60 % – 5.53 % |

**Columns**

| Group | Variables |
|---|---|
| Timestamp | `date` |
| Feed quality (hourly lab) | `% Iron Feed`, `% Silica Feed` |
| Reagents | `Starch Flow`, `Amina Flow` |
| Pulp | `Ore Pulp Flow`, `Ore Pulp pH`, `Ore Pulp Density` |
| Air flow | `Flotation Column 01–07 Air Flow` |
| Level | `Flotation Column 01–07 Level` |
| Output quality (hourly lab) | `% Iron Concentrate`, **`% Silica Concentrate` ← target** |

### A note on sampling frequency and this copy of the data

The original Kaggle release (`MiningProcess_Flotation_Plant_Database.csv`, ~183 MB, ~737,000 rows) logs **process variables every 20 seconds** while the **lab quality columns update only once per hour** — an hourly value is simply repeated across the ~180 rows inside that hour.

The copy used here is that same plant data already **aggregated to hourly means**, which is exactly the alignment step the pipeline would perform anyway. The code handles **both**: `clean_data()` measures the median sampling interval and, if the data is sub-hourly, resamples process signals to hourly means automatically. Drop the full 20-second Kaggle file into `data/` and everything runs unchanged.

## 5. Data Preprocessing

Implemented in `src/data_preprocessing.py`. The raw CSV is **never modified** — output goes to `outputs/processed_data.csv`.

| # | Step | Why |
|---|---|---|
| 1 | Parse `date` to datetime, sort ascending | Chronological order is required for an honest train/test split |
| 2 | Drop exactly-duplicated rows | Identical timestamp *and* values = logging glitch |
| 3 | Coerce value columns to numeric, converting comma decimals (`55,2` → `55.2`) | The raw Kaggle export stores numbers as strings in this format |
| 4 | Blank impossible values → NaN | A percentage outside 0–100, or a negative flow/level, is physically impossible |
| 5 | Resample to hourly means **if** the median step is sub-hourly | Puts every variable on one clock — the frequency the target actually has |
| 6 | Time-interpolate remaining gaps | These are continuous physical signals sampled in order, so a gap is best estimated from its neighbours |
| 7 | Drop rows with no target value | We never impute the thing we are trying to predict |
| 8 | Extract `hour`, `day`, `day_of_week`, `month` | Only these four; no unnecessary feature manufacturing |

On the current file, steps 2–6 each report **0 changes** — the data is already clean. That is a genuine result, and the checks stay in place because they fire on the raw 20-second file.

## 6. Exploratory Data Analysis

Five plots in `outputs/plots/`:

| File | Shows |
|---|---|
| `silica_distribution.png` | Target distribution — right-skewed, most hours 1–3 % |
| `silica_over_time.png` | Silica over the 6-month campaign with a 24 h rolling mean |
| `correlation_heatmap.png` | Full correlation matrix of process variables |
| `feature_target_relationship.png` | The 4 strongest process drivers vs silica |
| `feature_target_correlation.png` | Every variable ranked by correlation with the target |

**Measured correlations with `% Silica Concentrate`:**

| Variable | r |
|---|---|
| `% Iron Concentrate` | **−0.803** |
| `Flotation Column 01 Air Flow` | −0.225 |
| `Flotation Column 03 Air Flow` | −0.225 |
| `Flotation Column 05 Level` | −0.188 |
| `Flotation Column 04 Level` | −0.178 |
| `Flotation Column 02 Air Flow` | −0.175 |
| `Amina Flow` | +0.172 |

The headline: **one variable is at −0.80 and every genuine process variable is below |0.23|.** That gap drives the whole experimental design below.

## 7. Features Used

- **`X`** = 22 process variables + `hour` + `day_of_week`
- **`y`** = `% Silica Concentrate`

The target is never an input. `day` and `month` are computed for EDA but **excluded from the models**: under a chronological split the test months never appear in training, so a tree could only extrapolate on an unseen category.

### The `% Iron Concentrate` question

The concentrate is essentially iron plus silica, so the two are chemically bound to move in opposite directions (r = −0.803). Critically, **both come from the same hourly lab assay**. Knowing the iron result means the silica result is already in hand — so using it as a predictor is *leakage*: it inflates the score while destroying the model's practical purpose.

So two experiments are run:

- **A — `with_iron`:** all features including `% Iron Concentrate` (demonstrates the effect)
- **B — `without_iron`:** process variables only (the realistic, deployable model)

## 8. Train / Test Split

**Chronological, 80 / 20 — no shuffling.**

| Split | Rows | Period |
|---|---|---|
| Train | 3,255 | 2017-03-11 01:00 → 2017-08-06 21:00 |
| Test | 814 | 2017-08-06 22:00 → 2017-09-09 19:00 |

*Why not `train_test_split(shuffle=True)`?* In a continuous process, consecutive readings are nearly identical. Random shuffling would put 14:00 in training and 14:20 in testing, so the model would be graded on rows it has effectively memorised — a flattering, meaningless score. A forward-in-time split mirrors deployment: **fit on the past, predict the future.**

## 9. Machine Learning Models

| Model | How it works | Why included |
|---|---|---|
| **Linear Regression** | Weighted sum of inputs; wrapped in `StandardScaler` so coefficients are comparable across units | The baseline every other model must beat |
| **Random Forest** | 300 decision trees on bootstrapped samples, predictions averaged | Captures non-linearities and interactions; provides feature importances |
| **Gradient Boosting** | 300 shallow trees built sequentially, each correcting the previous errors (lr = 0.05, depth = 3) | The "basic improvement" over the baseline |

## 10. Evaluation Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Mean absolute error in percentage points of silica. If MAE = 0.89, a typical prediction is off by 0.89 %-points. Most intuitive for a plant engineer. |
| **RMSE** | Errors squared before averaging, so occasional large misses are punished harder than many small ones. Always ≥ MAE. |
| **R²** | Fraction of the variance in silica the model explains. 1.0 = perfect; 0.0 = no better than always guessing the mean; **negative = worse than guessing the mean**. |

## 11. Model Comparison

Held-out test set (latest 814 hours), from `outputs/validation_metrics.csv`:

| Experiment | Model | MAE | RMSE | R² | Train R² |
|---|---|---|---|---|---|
| with_iron | Linear Regression | 0.5577 | 0.7072 | 0.6201 | 0.6879 |
| with_iron | Random Forest | 0.5642 | 0.7028 | 0.6248 | 0.9599 |
| with_iron | **Gradient Boosting** | **0.5578** | **0.6980** | **0.6299** | 0.8429 |
| without_iron | Linear Regression | 0.9737 | 1.2377 | −0.1636 | 0.1781 |
| without_iron | Random Forest | 0.9396 | 1.1296 | 0.0307 | 0.8898 |
| without_iron | **Gradient Boosting** | **0.8937** | **1.0878** | **0.1011** | 0.5149 |

**Selected model: Gradient Boosting from the `without_iron` experiment** (MAE 0.8937, RMSE 1.0878, R² 0.1011). It is not the highest score on the page — the `with_iron` models score far better — but those rely on a lab number that does not exist at prediction time. Among honestly deployable models, Gradient Boosting wins on all three metrics.

## 12. Feature Importance

Random Forest importances (`outputs/plots/feature_importance.png`, `outputs/feature_importance.csv`):

**With `% Iron Concentrate` — leakage made visible:**

| Feature | Importance |
|---|---|
| `% Iron Concentrate` | **0.710** |
| `Amina Flow` | 0.020 |
| `% Silica Feed` | 0.019 |
| `Flotation Column 01 Air Flow` | 0.018 |

One variable absorbs **71 %** of the model. The other 23 share the remaining 29 % — the model is barely learning the process at all, just reading the lab's other number.

**Without it — the actual process drivers:**

| Feature | Importance |
|---|---|
| `Flotation Column 01 Air Flow` | 0.082 |
| `Amina Flow` | 0.079 |
| `Flotation Column 04 Air Flow` | 0.063 |
| `Flotation Column 03 Air Flow` | 0.059 |
| `Flotation Column 06 Level` | 0.051 |
| `Flotation Column 07 Level` | 0.049 |
| `% Silica Feed` | 0.045 |

Importance is now spread sensibly across many variables, and the ranking matches flotation theory: **air flow** (bubbles to carry silica), **Amina** (the collector that makes silica floatable), **column levels** (froth depth / selectivity), and **feed silica** (what came in). This is the plot worth discussing in an evaluation — it is chemically coherent.

## 13. Results

### Prediction analysis — selected model on 814 held-out hours

| Statistic | Value |
|---|---|
| MAE | 0.8937 %-points |
| RMSE | 1.0878 |
| R² | 0.1011 |
| Mean error as % of mean silica | 38.9 % |
| Worst absolute error | 3.497 |
| Predictions within ±0.5 %-points | 31.0 % of test rows |

Outputs: `outputs/predictions.csv` (timestamp, actual, predicted, absolute error, residual), `outputs/plots/actual_vs_predicted.png`, `outputs/plots/residual_analysis.png`.

### What the numbers actually say

**1. Silica is predictable from the lab's iron figure — but that is not a useful model.** R² jumps from 0.10 to 0.63 when `% Iron Concentrate` is added, and it takes 71 % of the feature importance. It is the target's twin from the same assay, not a process input.

**2. Without it, process variables explain only ~10 % of hourly variation.** This is an honest negative result and the most important finding here. For context, always guessing the test-set mean gives MAE ≈ 0.94; the model achieves 0.89 — a real but modest improvement. The plant is run at tight setpoints (several air flows sit near 300 with almost no variance), so the recorded inputs simply do not move enough to explain hour-to-hour silica swings, which are driven by unmeasured ore mineralogy and particle size.

**3. Non-linear models clearly beat the linear baseline where it matters.** In the realistic experiment, Linear Regression posts a **negative R² (−0.16)** — literally worse than predicting the average — while Gradient Boosting reaches +0.10. Flotation is not a weighted sum of its inputs.

**4. Random Forest overfits noticeably.** Train R² 0.89 vs test 0.03. Gradient Boosting's shallow trees (depth 3) generalise better: 0.51 vs 0.10.

### OPTIONAL forecasting experiment

*Exploratory extension, clearly separated from the core deliverable* (`outputs/forecast_metrics.csv`). Using only information available at time *t* (process variables + lags of past silica, no `% Iron Concentrate`), predict silica at *t + h*, benchmarked against **persistence** ("assume silica stays where it is"):

| Horizon | RF (+lags) MAE | RF R² | Persistence MAE | Persistence R² |
|---|---|---|---|---|
| 1 h | 0.6537 | 0.4546 | **0.4447** | **0.6119** |
| 2 h | 0.7553 | 0.3185 | **0.5778** | **0.4164** |
| 4 h | 0.9050 | **0.0991** | 0.7773 | 0.0830 |

**Answering the brief's questions honestly:**

- *Can silica be predicted every minute?* Not from this file — the lab target is hourly, so minute-level values would be interpolation, not measurement. It would need the raw 20-second data plus a defensible interpolation scheme.
- *How many hours ahead?* Skill decays fast. At 1–2 h the naive persistence baseline **beats** the ML model; only by 4 h does the model edge ahead, and by then both are near-useless (R² ≈ 0.09). Practical predictive horizon: **roughly 1–2 hours, and the best cheap predictor is the current reading.**
- *Can silica be predicted without `% Iron Concentrate`?* **Yes, but only weakly** — MAE 0.89 vs 0.94 for guessing the mean. Directionally useful, not accurate enough for closed-loop control.

## 14. Limitations

1. **Hourly copy of the data.** 4,069 rows, not the full ~737,000-row 20-second export. Sub-hourly dynamics (rapid air-flow excursions) are averaged away. The code supports the full file.
2. **Weak process signal.** R² ≈ 0.10 without the iron column. The dataset lacks the variables that likely drive silica: ore mineralogy, particle-size distribution, froth imagery, upstream grinding conditions.
3. **No process lag modelling.** Pulp takes time to travel through 7 columns, so the causes of the silica measured at 14:00 occurred somewhat earlier. No lead/lag alignment was applied.
4. **Single chronological split**, not walk-forward cross-validation, so metrics reflect one specific 34-day test window (August–September) and may shift with plant conditions.
5. **Default hyperparameters**, no tuning — deliberate, to keep the project explainable.
6. **Impurity-based feature importances** are biased toward high-cardinality continuous variables; permutation importance would be more rigorous.
7. **Correlation is not causation.** High Amina importance does not prove that changing Amina will change silica by the amount implied.

## 15. Future Improvements

- Run on the **full 20-second dataset** and model within-hour dynamics.
- Add **lag/lead features** with a proper residence-time analysis to align causes with effects.
- Use **`TimeSeriesSplit`** walk-forward validation for more robust estimates.
- **Tune hyperparameters** with `GridSearchCV` inside a time-series CV loop.
- Add **rolling-window statistics** (rolling means/stds of air flow and level) to capture process stability, not just instantaneous values.
- Try **permutation importance** and **SHAP** for trustworthier attribution.
- Reframe as **classification** ("will the next hour exceed the silica spec limit?") — a yes/no alarm may be more actionable than a weak point estimate.
- Enrich with **ore mineralogy / particle-size data** if the plant can supply it; this is most likely the real accuracy ceiling.

## 16. How to Run the Project

### Folder structure

```
mining-process-quality-prediction/
│
├── data/
│   └── mining_process_data.csv          # input data (not modified)
│
├── notebooks/
│   └── mining_quality_analysis.ipynb    # full walkthrough, 29 cells
│
├── src/
│   ├── data_preprocessing.py            # load, inspect, clean, timestamps, EDA
│   ├── train_model.py                   # features, split, 3 models, importance, forecasting
│   └── evaluate_model.py                # predictions, metrics, diagnostic plots
│
├── outputs/
│   ├── plots/
│   │   ├── silica_distribution.png
│   │   ├── silica_over_time.png
│   │   ├── correlation_heatmap.png
│   │   ├── feature_target_relationship.png
│   │   ├── feature_target_correlation.png
│   │   ├── feature_importance.png
│   │   ├── actual_vs_predicted.png
│   │   └── residual_analysis.png
│   ├── models/                          # trained .joblib models
│   ├── processed_data.csv
│   ├── predictions.csv
│   ├── validation_metrics.csv
│   ├── feature_importance.csv
│   ├── forecast_metrics.csv
│   ├── model_comparison.md
│   └── selected_model.json
│
├── requirements.txt
└── README.md
```

### Placing the dataset

The pipeline expects one CSV at:

```
data/mining_process_data.csv
```

To use the original Kaggle file:

1. Download **"Quality Prediction in a Mining Process"** (`MiningProcess_Flotation_Plant_Database.csv`) from Kaggle.
2. Copy it into `data/`.
3. Either rename it to `mining_process_data.csv`, or point the code at it:
   ```bash
   python src/data_preprocessing.py --data-path data/MiningProcess_Flotation_Plant_Database.csv
   ```
   (Or edit `DEFAULT_DATA_PATH` at the top of `src/data_preprocessing.py`.)

The loader handles comma decimal separators and auto-resamples 20-second data to hourly. Column names are detected flexibly. **If the file is missing, the script stops with a clear message — it never invents data.**

### Commands

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. clean the data + generate EDA plots   -> outputs/processed_data.csv
python src/data_preprocessing.py

# 3. train and compare models              -> outputs/validation_metrics.csv
cd src && python train_model.py && cd ..

# 4. evaluate + prediction analysis        -> outputs/predictions.csv
cd src && python evaluate_model.py && cd ..
```

Useful flags:

```bash
python src/data_preprocessing.py --data-path <path>     # different dataset
python src/data_preprocessing.py --no-eda               # skip plots
python src/data_preprocessing.py --resample-hourly always
cd src && python train_model.py --no-forecast           # core pipeline only
```

Notebook alternative (runs the identical functions):

```bash
jupyter notebook notebooks/mining_quality_analysis.ipynb
```

> Steps 3 and 4 are run from inside `src/` because those modules import each other directly — the simplest arrangement for a small project, avoiding package boilerplate.

---

## Implemented vs. optional

**Fully implemented and executed:**

- ✅ Data loading with a clear missing-file error
- ✅ All 8 inspection items
- ✅ Cleaning: duplicates, numeric coercion, invalid values, interpolation, frequency alignment
- ✅ Timestamp parsing + `hour` / `day` / `day_of_week` / `month`
- ✅ All 5 required EDA plots
- ✅ Feature preparation with two experiments (with / without `% Iron Concentrate`)
- ✅ Chronological 80/20 split
- ✅ Linear Regression, Random Forest, Gradient Boosting
- ✅ MAE / RMSE / R² comparison table
- ✅ `predictions.csv` with timestamp, actual, predicted, absolute error
- ✅ Actual-vs-predicted and residual analysis plots
- ✅ Random Forest feature importance + bar chart
- ✅ Model selection and saved `.joblib` artefacts

**Optional / exploratory (implemented, clearly labelled):**

- 🔸 Forecasting experiment at 1 / 2 / 4 h with a persistence baseline

**Documented as future work, not implemented:**

- ⬜ Minute-level prediction (needs the raw 20-second file + an interpolation scheme)
- ⬜ Hyperparameter tuning, walk-forward CV, SHAP, lag/residence-time modelling

## Attribution

Data: *Quality Prediction in a Mining Process*, Eduardo Magalhães Oliveira (Kaggle) — real froth-flotation plant measurements, March–September 2017.
