# Turbofan Engine Remaining Useful Life (RUL) Prediction

A simple, complete machine-learning project that predicts the **Remaining
Useful Life (RUL)** — the number of operational cycles left before failure —
of turbofan engines using the **NASA C-MAPSS** dataset (subset **FD001**).

> Internship demonstration project: a clean, well-commented ML pipeline
> (no deep learning, no web UI, no database). Every number in this README
> was produced by actually running the code in this repository.

---

## 1. Problem statement

A fleet of turbofan engines is monitored cycle by cycle. Each engine starts
healthy and gradually degrades until it fails. The C-MAPSS simulator records,
for every engine and every operational cycle, 3 operational settings and 21
sensor measurements (temperatures, pressures, speeds, etc.).

**The problem:** given the sensor history of an engine *up to its current
cycle*, predict how many more cycles it can run before failure. This is
called the Remaining Useful Life (RUL).

Accurate RUL prediction enables *predictive maintenance*: engines can be
serviced just before they fail, avoiding both unexpected breakdowns and
unnecessary early maintenance.

## 2. Objective

Train regression models on the historical run-to-failure data
(`train_FD001.txt`) and evaluate how well they predict the true RUL of the
test engines (`RUL_FD001.txt`), using MAE, RMSE and R².

## 3. Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), subset
FD001 — a widely used benchmark for prognostics.

| file | description |
|---|---|
| `data/train_FD001.txt` | 100 engines run **until failure**: 20,631 rows × 26 columns |
| `data/test_FD001.txt` | 100 engines truncated at some point **before failure**: 13,096 rows × 26 columns |
| `data/RUL_FD001.txt` | true remaining life (cycles) of every test engine, one value per line |

Each row contains: `unit` (engine id), `cycle`, 3 operational settings and
21 sensor measurements. FD001 has a single operating condition and a single
fault mode.

**The files are already included in `data/`.** If you need to re-download
them (e.g. to run a different subset such as FD002-FD004), get the official
CMAPSSData.zip from NASA:

- NASA PCoE data repository:
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA Open Data Portal (dataset page):
  https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

then place `train_FD001.txt`, `test_FD001.txt` and `RUL_FD001.txt` into
`data/`. The code raises a clear error message if a file is missing.

## 4. Methodology

```
Dataset → Data loading → Data exploration → Preprocessing → RUL target
→ Feature selection → Train/validation split → ML models → Prediction
→ Evaluation → Sample RUL predictions
```

1. **Loading** — the space-separated `.txt` files are read with pandas and
   given meaningful column names (`unit`, `cycle`, `setting1-3`, `sensor1-21`).
2. **Exploration** — RUL distribution, sensor trends vs cycle, correlation
   heatmap (plots in `outputs/plots/`).
3. **Preprocessing** — no missing values were found. Columns with
   (near-)zero variance are dropped because they carry no information.
4. **RUL target (training set only)** — for each row of an engine:

   ```
   RUL = maximum cycle of that engine − current cycle
   ```

   Example: an engine fails at cycle 192. Its row at cycle 182 gets
   RUL = 10; its last row (cycle 192) gets RUL = 0. The test engines are
   truncated, so their true RUL comes from `RUL_FD001.txt` instead.
5. **Feature selection** — keep `cycle` plus every sensor with enough
   variance (see section 5).
6. **Train/validation split** — split by **engine id** (80 train / 20
   validation engines), so validation engines are completely unseen and the
   evaluation is not optimistically biased by data leakage.
7. **Models** — Random Forest and Gradient Boosting regressors are trained
   and compared on the validation engines; the better one is retrained on
   all training data and saved with `joblib`.
8. **Test evaluation** — for each test engine, the model predicts the RUL at
   its **last recorded cycle**; predictions are compared with the true RUL.

## 5. Features used

13 columns had variance below the threshold and were dropped:
`setting1-3`, `sensor1`, `sensor5`, `sensor6`, `sensor8`, `sensor10`,
`sensor13`, `sensor15`, `sensor16`, `sensor18`, `sensor19`.

**12 features are kept:**

```
cycle, sensor2, sensor3, sensor4, sensor7, sensor9,
sensor11, sensor12, sensor14, sensor17, sensor20, sensor21
```

`cycle` (engine age) is by far the most important feature (feature
importance ≈ 0.68 for the selected model), followed by `sensor11`, `sensor4`
and `sensor12` — sensors that show clear degradation trends.

## 6. Machine learning models

Two tree-based regressors, compared on the same validation engines
(hyperparameters are deliberately simple defaults):

| model | idea |
|---|---|
| **Random Forest** | 100 independent decision trees; prediction = average of the trees. Reduces variance / overfitting. |
| **Gradient Boosting** | 100 shallow trees added one after another, each one fixing the errors of the previous ones. Often the strongest "classical" model for tabular data. |

**Why these models?** Turbofan degradation is non-linear and the sensor
columns interact with each other — tree ensembles capture that without any
feature engineering. They need no feature scaling, handle outliers well,
train in seconds on this dataset, and expose feature importances which make
the model easy to explain. Deep learning was intentionally avoided: it is
unnecessary for this demonstration.

A `StandardScaler` is kept in the pipelines anyway, so a linear/SVM model
can be swapped in with one line without changing anything else.

## 7. Evaluation metrics

| metric | meaning |
|---|---|
| **MAE** | mean of `|actual − predicted|`. Average error in cycles — the most intuitive metric. |
| **RMSE** | square root of the mean squared error. Penalises large errors more than MAE; always ≥ MAE. |
| **R²** | share of the variance in the true RUL explained by the model (1 = perfect, 0 = as good as predicting the mean). |

Lower MAE/RMSE and higher R² are better.

## 8. Results

Reproduced with `python src/train_model.py` and `python src/evaluate_model.py`
(seed 42). Validation metrics on the 20 held-out **engines**:

| model | MAE (cycles) | RMSE (cycles) | R² |
|---|---|---|---|
| Random Forest | 24.14 | 31.82 | 0.765 |
| **Gradient Boosting** ✅ | **22.87** | **29.90** | **0.793** |

**Selected model: Gradient Boosting** (lowest validation RMSE). Retrained on
all 100 training engines and evaluated on the 100 test engines:

| metric | value |
|---|---|
| **MAE** | **18.25 cycles** |
| **RMSE** | **25.23 cycles** |
| **R²** | **0.631** |

On average, the predicted RUL is within ~18 cycles of the true value.
`outputs/predictions.csv` contains the per-engine table (engine id, actual
RUL, predicted RUL, absolute error). A few example rows:

| engine_id | actual_rul | predicted_rul | abs_error |
|---|---|---|---|
| 1 | 112 | 185.6 | 73.6 |
| 2 | 98 | 139.8 | 41.8 |
| 3 | 69 | 44.7 | 24.3 |
| 4 | 82 | 82.0 | 0.0 |
| 5 | 91 | 89.2 | 1.8 |

Plots (in `outputs/plots/`): RUL distribution, sensor degradation trends,
correlation heatmap, model comparison, and actual-vs-predicted RUL for both
validation and test sets.

**Honest context:** this is a simple baseline, exactly as intended. The
model is least reliable for engines early in their life (e.g. engines 1 and
15), where the sensors still look healthy but the linear RUL target forces
an exact long-range forecast.

### Optional experiment: piecewise-linear RUL target

The literature on C-MAPSS usually clips the target at a constant early-life
value (`RUL = min(max_cycle − cycle, 125)`) because a healthy engine's exact
failure time is unknowable far in advance. This is available in the code by
setting `RUL_CAP = 125` in `src/data_preprocessing.py`. Measured with the
same Gradient Boosting setup:

| RUL target | val MAE | val RMSE | val R² | test MAE | test RMSE | test R² |
|---|---|---|---|---|---|---|
| linear (default, as in the brief) | 22.87 | 29.90 | 0.793 | 18.25 | 25.23 | 0.631 |
| piecewise, cap = 125 | 10.90 | 15.71 | 0.858 | **12.65** | **17.67** | **0.819** |

The default follows the project brief (pure linear RUL); the cap is offered
as a documented one-line improvement.

## 9. How to run the project

```bash
# 1. (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. make sure the dataset files are in data/  (they are included;
#    see section 3 if you need to re-download them)

# 4. run the pipeline (from the project root)
python src/data_preprocessing.py   # loads data, EDA plots, processed CSVs
python src/train_model.py          # trains + compares models, saves the best
python src/evaluate_model.py       # test predictions + metrics + table

# 5. (optional) open the walkthrough notebook
jupyter notebook notebooks/turbofan_rul_analysis.ipynb
```

Expected run time for the whole pipeline: well under a minute on a normal
laptop.

## 10. Project structure

```
turbofan-rul-prediction/
│
├── data/                          # NASA C-MAPSS FD001 files (included)
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── notebooks/
│   └── turbofan_rul_analysis.ipynb   # full walkthrough with executed outputs
│
├── src/
│   ├── data_preprocessing.py         # loading, cleaning, RUL target, EDA plots
│   ├── train_model.py                # engine-wise split, RF vs GB, save best model
│   └── evaluate_model.py             # test-set predictions, metrics, table
│
├── outputs/                          # generated by running the pipeline
│   ├── plots/                        # all figures (PNG)
│   ├── best_model.joblib             # saved selected model
│   ├── validation_metrics.csv        # model comparison on validation engines
│   ├── predictions.csv               # per-engine test predictions
│   ├── train_processed.csv           # processed training set (features + RUL)
│   └── test_processed.csv            # processed test set (last cycle + true RUL)
│
├── requirements.txt
└── README.md
```

## 11. Future improvements

1. **Piecewise-linear RUL cap** (`RUL_CAP = 125`) — already implemented as an
   option; improves the test RMSE from 25.2 to ~17.7 (section 8).
2. **Feature engineering** — rolling means/trends of the sensors over recent
   cycles (the sensors' *rate of change* often matters more than their level).
3. **Hyperparameter tuning** — small grid search over `n_estimators`,
   `max_depth`, `learning_rate`.
4. **More models** — e.g. XGBoost / LightGBM, or a simple feed-forward
   neural network if deeper comparison is desired.
5. **Other C-MAPSS subsets** — FD002-FD004 add multiple operating
   conditions and fault modes; the pipeline only needs the file names changed.
6. **Uncertainty estimates** — e.g. prediction intervals from the forest
   variance, useful for real maintenance decisions.
