import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent / "namadataset_preprocessing"
EDA_DIR = Path(__file__).resolve().parent / "eda_outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILES = [
    "ispu_dki_all.csv",
    "ispu_dki1.csv",
    "ispu_dki2.csv",
    "ispu_dki3.csv",
    "ispu_dki4.csv",
    "ispu_dki5.csv",
]

NUMERIC_COLS = ["pm25", "pm10", "so2", "co", "o3", "no2"]
TARGET_COL = "categori"
REQUIRED_COLS = ["tanggal", "stasiun", *NUMERIC_COLS, "max", "critical", TARGET_COL]


def read_datasets() -> pd.DataFrame:
    """Membangun_model.preprocessing_eda

    Template-style preprocessing and EDA pipeline adapted to the project.

    Sections:
    1. Perkenalan Dataset
    2. Import Library
    3. Memuat Dataset
    4. Exploratory Data Analysis (EDA)
    5. Data Preprocessing

    This module keeps the original preprocessing functionality but organizes
    the code and adds explanatory headings so it follows the provided template.
    """

    import os
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split


    # ---------------------------
    # 1. Perkenalan Dataset
    # ---------------------------
    # Dataset source: CSV files under the repository `dataset/` directory. The
    # preprocessing reads multiple CSV variants and concatenates them.

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATASET_DIR = BASE_DIR / "dataset"
    OUTPUT_DIR = Path(__file__).resolve().parent / "namadataset_preprocessing"
    EDA_DIR = Path(__file__).resolve().parent / "eda_outputs"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    CSV_FILES = [
        "ispu_dki_all.csv",
        "ispu_dki1.csv",
        "ispu_dki2.csv",
        "ispu_dki3.csv",
        "ispu_dki4.csv",
        "ispu_dki5.csv",
    ]

    NUMERIC_COLS = ["pm25", "pm10", "so2", "co", "o3", "no2"]
    TARGET_COL = "categori"
    REQUIRED_COLS = ["tanggal", "stasiun", *NUMERIC_COLS, "max", "critical", TARGET_COL]


    # ---------------------------
    # 3. Memuat Dataset
    # ---------------------------
    def read_datasets() -> pd.DataFrame:
        """Load and concatenate CSV files from `dataset/`.

        Returns
        -------
        pd.DataFrame
            Raw concatenated DataFrame with normalized column names and a
            `source_file` column indicating the origin CSV.
        """
        frames = []
        for file_name in CSV_FILES:
            file_path = DATASET_DIR / file_name
            if not file_path.exists():
                print(f"[WARNING] File tidak ditemukan: {file_path}")
                continue

            df = pd.read_csv(file_path)
            df.columns = [col.strip().lower() for col in df.columns]
            df["source_file"] = file_name
            frames.append(df)

        if not frames:
            raise FileNotFoundError(
                "Tidak ada dataset CSV ditemukan. Pastikan file CSV berada di folder dataset/."
            )

        data = pd.concat(frames, ignore_index=True)
        return data


    # ---------------------------
    # 5. Data Preprocessing
    # ---------------------------
    def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataset for modeling.

        Steps:
        - normalise column names
        - check required columns
        - parse dates and numeric columns
        - drop duplicates and invalid labels
        - impute numeric missing values with median
        - create simple temporal features
        """
        df = df.copy()
        df.columns = [col.strip().lower() for col in df.columns]

        missing_required = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing_required:
            raise ValueError(f"Kolom wajib tidak ditemukan: {missing_required}")

        df = df[REQUIRED_COLS + ["source_file"]].copy()
        df = df.drop_duplicates()

        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

        for col in NUMERIC_COLS + ["max"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["stasiun"] = df["stasiun"].astype(str).str.strip().str.upper()
        df["critical"] = df["critical"].astype(str).str.strip().str.upper()
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper()

        df = df.dropna(subset=["tanggal", TARGET_COL])
        df = df[df[TARGET_COL].notna()]
        df = df[df[TARGET_COL] != ""]
        df = df[df[TARGET_COL] != "NAN"]

        for col in NUMERIC_COLS:
            df[col] = df[col].fillna(df[col].median())

        df["year"] = df["tanggal"].dt.year
        df["month"] = df["tanggal"].dt.month
        df["day"] = df["tanggal"].dt.day
        df["dayofweek"] = df["tanggal"].dt.dayofweek

        # Fitur utama. max dan critical tidak digunakan untuk menghindari leakage.
        selected_cols = [
            "tanggal",
            "stasiun",
            "pm25",
            "pm10",
            "so2",
            "co",
            "o3",
            "no2",
            "year",
            "month",
            "day",
            "dayofweek",
            TARGET_COL,
        ]

        return df[selected_cols].reset_index(drop=True)


    # ---------------------------
    # 4. Exploratory Data Analysis (EDA)
    # ---------------------------
    def run_eda(df: pd.DataFrame) -> None:
        """Generate simple EDA outputs into `eda_outputs/`.

        The function creates summary CSVs and PNG visualizations used by the
        project documentation.
        """
        summary = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "target_unique": int(df[TARGET_COL].nunique()),
        }
        pd.Series(summary).to_csv(EDA_DIR / "data_summary.csv")

        missing = df.isna().sum().reset_index()
        missing.columns = ["column", "missing_count"]
        missing.to_csv(EDA_DIR / "missing_values.csv", index=False)

        target_counts = df[TARGET_COL].value_counts()
        target_counts.to_csv(EDA_DIR / "target_distribution.csv")

        plt.figure(figsize=(10, 5))
        target_counts.plot(kind="bar")
        plt.title("Distribusi Kategori ISPU")
        plt.xlabel("Kategori")
        plt.ylabel("Jumlah")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "target_distribution.png")
        plt.close()

        corr = df[NUMERIC_COLS].corr()
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar()
        plt.title("Correlation Matrix Parameter Polutan")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "correlation_matrix.png")
        plt.close()

        category_station = pd.crosstab(df["stasiun"], df[TARGET_COL])
        category_station.to_csv(EDA_DIR / "category_by_station.csv")
        category_station.plot(kind="bar", stacked=True, figsize=(12, 6))
        plt.title("Kategori ISPU per Stasiun")
        plt.xlabel("Stasiun")
        plt.ylabel("Jumlah")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "category_by_station.png")
        plt.close()

        monthly = df.groupby(["year", "month"])[NUMERIC_COLS].mean().reset_index()
        monthly.to_csv(EDA_DIR / "monthly_pollutant_average.csv", index=False)


    def save_train_test(df: pd.DataFrame) -> None:
        """Save preprocessed dataset and train/test splits to `namadataset_preprocessing/`.

        The function stratifies the split when possible.
        """
        feature_cols = ["stasiun", "pm25", "pm10", "so2", "co", "o3", "no2", "month", "day", "dayofweek"]
        X = df[feature_cols]
        y = df[TARGET_COL]

        stratify = y if y.value_counts().min() >= 2 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        df.to_csv(OUTPUT_DIR / "ispu_preprocessed.csv", index=False)
        X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
        X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
        y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
        y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

        print("[OK] Dataset siap pakai disimpan ke:", OUTPUT_DIR)


    def main():
        raw_df = read_datasets()
        clean_df = clean_dataset(raw_df)
        run_eda(clean_df)
        save_train_test(clean_df)
        print("[OK] Preprocessing dan EDA selesai.")


    if __name__ == "__main__":
        main()
