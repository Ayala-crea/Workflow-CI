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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "dataset_preprocessing"
MODEL_DIR = CURRENT_DIR / "model"
MLRUNS_DIR = Path(__file__).resolve().parents[1].parent / "mlruns"
EXPERIMENT_NAME = "ISPU_DKI_Baseline_Autolog"

FEATURE_NUMERIC = ["pm25", "pm10", "so2", "co", "o3", "no2", "month", "day", "dayofweek"]
FEATURE_CATEGORICAL = ["stasiun"]


def load_data():
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [file for file in required if not (DATA_DIR / file).exists()]
    if missing:
        raise FileNotFoundError(
            f"File preprocessing belum tersedia: {missing}. Jalankan preprocessing_eda.py terlebih dahulu."
        )

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
        n_estimators=200,
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


def save_confusion_matrix(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, xticks_rotation=45)
    plt.title("Training Confusion Matrix - Baseline")
    plt.tight_layout()
    cm_path = MODEL_DIR / "training_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    return cm_path


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()
    pipeline = build_pipeline()

    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="baseline_random_forest_autolog"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # Autolog sudah mencatat banyak parameter/model.
        # Metrik tambahan ini dicatat agar konsisten dengan file tuning manual.
        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        report_path = MODEL_DIR / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path))

        metric_path = MODEL_DIR / "metric_info.json"
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metric_path))

        cm_path = save_confusion_matrix(pipeline, X_test, y_test)
        mlflow.log_artifact(str(cm_path))

        # Simpan model MLflow ke subfolder terpisah untuk menghindari konflik
        # dengan artifact lain yang disimpan di `MODEL_DIR`.
        target_model_dir = MODEL_DIR / "mlflow_model"
        if target_model_dir.exists():
            shutil.rmtree(target_model_dir)

        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=str(target_model_dir),
            input_example=X_train.head(3),
        )

        # (opsional) juga catat model sebagai artifact MLflow pada run saat ini
        try:
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
        except Exception:
            # jika logging gagal, lanjutkan — model sudah disimpan lokal
            pass

        # Salin ulang artifact tambahan setelah save_model.
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_confusion_matrix(pipeline, X_test, y_test)

        print("[OK] Training baseline selesai.")
        print("[OK] Metrics:", metrics)
        print("[OK] Model disimpan di:", MODEL_DIR)


if __name__ == "__main__":
    main()
