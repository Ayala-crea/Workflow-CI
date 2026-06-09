from pathlib import Path
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "namadataset_preprocessing"
MODEL_DIR = CURRENT_DIR / "model_tuning"
MLRUNS_DIR = CURRENT_DIR / "mlruns"
EXPERIMENT_NAME = "ISPU_DKI_Tuning_Manual_Logging"

FEATURE_NUMERIC = ["pm25", "pm10", "so2", "co", "o3", "no2", "month", "day", "dayofweek"]
FEATURE_CATEGORICAL = ["stasiun"]


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_CATEGORICAL),
        ]
    )

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    pipeline = build_pipeline()

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    with mlflow.start_run(run_name="random_forest_gridsearch_manual_logging"):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "best_cv_f1_weighted": search.best_score_,
        }

        # Manual logging, bukan autolog.
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        report_path = MODEL_DIR / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        metric_path = MODEL_DIR / "metric_info.json"
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, xticks_rotation=45)
        plt.title("Training Confusion Matrix - Tuning")
        plt.tight_layout()
        cm_path = MODEL_DIR / "training_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metric_path))
        mlflow.log_artifact(str(cm_path))

        if MODEL_DIR.exists():
            for item in MODEL_DIR.iterdir():
                if item.name not in ["training_confusion_matrix.png", "classification_report.json", "metric_info.json"]:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)

        mlflow.sklearn.save_model(
            sk_model=best_model,
            path=str(MODEL_DIR),
            input_example=X_train.head(3),
        )

        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("[OK] Training tuning selesai.")
        print("[OK] Best params:", search.best_params_)
        print("[OK] Metrics:", metrics)
        print("[OK] Model tuning disimpan di:", MODEL_DIR)


if __name__ == "__main__":
    main()
