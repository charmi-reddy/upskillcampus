"""
evaluate_model.py
-----------------
Loads the saved best model and evaluates it on the C-MAPSS test set:

  - predicts the RUL for the last cycle of every test engine
  - compares predictions with the true RUL values
  - prints MAE / RMSE / R2 and a per-engine table
  - saves outputs/predictions.csv and a plot

Run standalone with:  python src/evaluate_model.py
(requires outputs/best_model.joblib - run train_model.py first)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Allow both `python src/evaluate_model.py` and `python -m src.evaluate_model`
try:
    from src import data_preprocessing as dp
    from src.train_model import rmse, evaluate_model, plot_actual_vs_predicted
except ModuleNotFoundError:
    import data_preprocessing as dp
    from train_model import rmse, evaluate_model, plot_actual_vs_predicted

MODEL_PATH = dp.OUTPUT_DIR / "best_model.joblib"
PREDICTIONS_PATH = dp.OUTPUT_DIR / "predictions.csv"


def main():
    # 1. Check that a trained model exists -----------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\nTrained model not found at:\n    {MODEL_PATH}\n\n"
            "Please run the training step first:\n"
            "    python src/train_model.py\n"
        )

    # 2. Load the test data and the true RUL values ---------------------------
    print("Loading the test set and true RUL values ...")
    test = dp.load_test_data()
    rul = dp.load_test_rul()

    # 3. Rebuild the feature list exactly as in training -----------------------
    train = dp.add_rul_column(dp.load_train_data())
    features, _ = dp.select_features(train)
    print(f"Features used ({len(features)}): {features}")

    # 4. Prepare the test matrix: last cycle of every test engine --------------
    X_test, y_true, test_final = dp.prepare_test(test, rul, features)
    print(f"Test engines to predict: {X_test.shape[0]}")

    # 5. Load the model and predict --------------------------------------------
    pipe = joblib.load(MODEL_PATH)
    print(f"Loaded model: {pipe.named_steps['model']}")
    y_pred = pipe.predict(X_test)

    # 6. Overall metrics ---------------------------------------------------------
    m = evaluate_model(y_true, y_pred)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"MAE  (mean absolute error)       : {m['MAE']:8.2f} cycles")
    print(f"RMSE (root mean squared error)   : {m['RMSE']:8.2f} cycles")
    print(f"R2   (coefficient of determination): {m['R2']:8.3f}")

    # 7. Per-engine prediction table ---------------------------------------------
    results = pd.DataFrame({
        "engine_id": test_final["unit"].values,
        "actual_rul": y_true.values,
        "predicted_rul": np.round(y_pred, 1),
        "abs_error": np.round(np.abs(y_true.values - y_pred), 1),
    })
    dp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nFull prediction table saved to {PREDICTIONS_PATH}")

    print("\nFirst 10 predictions:")
    print(results.head(10).to_string(index=False))
    print("\n5 predictions with the largest error:")
    print(results.nlargest(5, "abs_error").to_string(index=False))

    # 8. Actual vs predicted plot -------------------------------------------------
    plot_actual_vs_predicted(
        y_true, y_pred,
        "Test set: actual vs predicted RUL",
        dp.PLOTS_DIR / "test_actual_vs_predicted.png",
    )
    plt.close("all")
    print(f"\nPlot saved to {dp.PLOTS_DIR / 'test_actual_vs_predicted.png'}")
    print("Evaluation finished.")


if __name__ == "__main__":
    main()
