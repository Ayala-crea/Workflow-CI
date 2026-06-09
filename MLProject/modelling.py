from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CURRENT_DIR = Path(__file__).resolve().parent
DATA_DIR = CURRENT_DIR / "dataset_preprocessing"
MODEL_DIR = CURRENT_DIR / "model"

EXPERIMENT_NAME = "ISPU_DKI_Baseline_Autolog"

FEATURE_NUMERIC = [
    "pm25",
    "pm10",
    "so2",
    "co",
    "o3",
    "no2",
    "month",
    "day",
    "dayofweek",
]

FEATURE_CATEGORICAL = ["stasiun"]


def load_data():
    required_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]

    missing_files = [
        file_name
        for file_name in required_files
        if not (DATA_DIR / file_name).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"File preprocessing belum tersedia: {missing_files}. "
            f"Jalankan preprocessing_eda.py terlebih dahulu."
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


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
    )

    X_train, X_test, y_train, y_test = load_data()
    pipeline = build_pipeline()

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("[OK] Training baseline selesai menggunakan MLflow autolog.")
    print(f"[OK] Accuracy: {accuracy:.4f}")
    print(f"[OK] F1 Macro: {f1_macro:.4f}")


if __name__ == "__main__":
    main()