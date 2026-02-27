import argparse
import json
import os
import tempfile
from pathlib import Path
from joblib import dump

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# mlflow ui

# python src/train.py --max-depth 2 --run-name "rf_depth_2"
# python src/train.py --max-depth 4 --run-name "rf_depth_4"
# python src/train.py --max-depth 6 --run-name "rf_depth_6"
# python src/train.py --max-depth 10 --run-name "rf_depth_10"
# python src/train.py --max-depth None --run-name "rf_depth_none"

# dvc add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
# git add data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv.dvc data/raw/.gitignore


def parse_args():
    parser = argparse.ArgumentParser(description="Train Telco Customer Churn model with MLflow logging")

    # DVC pipeline inputs/outputs (позиційні аргументи)
    parser.add_argument("prepared_dir", type=str, help="Directory with prepared train/test CSVs (e.g. data/prepared)")
    parser.add_argument("output_dir", type=str, help="Directory to save local model artifacts (e.g. data/models)")

    # Data config
    parser.add_argument("--target-col", type=str, default="Churn", help="Target column name")
    parser.add_argument("--id-col", type=str, default="customerID", help="ID column to drop")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")

    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators")
    parser.add_argument(
        "--max-depth",
        type=str,
        default="None",
        help='RandomForest max_depth (e.g. "5", "10", "None")',
    )
    parser.add_argument("--min-samples-split", type=int, default=2, help="RF min_samples_split")
    parser.add_argument("--min-samples-leaf", type=int, default=1, help="RF min_samples_leaf")
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        help='Class weight: "balanced", "None", etc.',
    )

    # MLflow
    parser.add_argument("--experiment-name", type=str, default="Telco_Churn_Lab1", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="Optional MLflow run name")
    parser.add_argument("--tracking-uri", type=str, default=None, help='Optional MLflow tracking URI (e.g. "file:./mlruns")')
    parser.add_argument("--author", type=str, default="Joanna", help="Author tag")
    parser.add_argument("--dataset-version", type=str, default="v1", help="Dataset version tag")

    # Artifact settings
    parser.add_argument("--top-k-features", type=int, default=20, help="Top K features for importance plot")

    return parser.parse_args()


def parse_none(value: str):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return int(value)


def parse_class_weight(value: str):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def prepare_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    y_raw = df[target_col].copy()

    # Підтримка стандартного Telco формату Yes/No
    if y_raw.dtype == "object":
        unique_vals = set(y_raw.dropna().astype(str).str.strip().unique())
        if unique_vals.issubset({"Yes", "No"}):
            y = y_raw.map({"No": 0, "Yes": 1})
        else:
            raise ValueError(
                f"Unexpected target values in '{target_col}': {sorted(unique_vals)}. "
                "Очікувались 'Yes'/'No'."
            )
    else:
        # Якщо вже 0/1
        y = y_raw.astype(int)

    if y.isna().any():
        raise ValueError("Target column contains NaN after mapping. Please check target values.")

    return y.astype(int)


def build_pipeline(numeric_cols, categorical_cols, rf_params):
    # заповнення пропусків + one-hot для категоріальних
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(**rf_params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass

    return metrics


def save_confusion_matrix_plot(y_true, y_pred, output_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix (Test)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve_plot(y_true, y_proba, output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve (Test)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def get_feature_names(preprocessor: ColumnTransformer):
    """Повертає назви ознак після ColumnTransformer + OneHotEncoder."""
    feature_names = []

    # Числові
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(list(num_cols))

    # Категоріальні (через OneHot)
    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]
    cat_cols = preprocessor.transformers_[1][2]
    cat_feature_names = onehot.get_feature_names_out(cat_cols)
    feature_names.extend(list(cat_feature_names))

    return feature_names


def save_feature_importance_plot(pipeline: Pipeline, output_path: Path, top_k: int = 20):
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return False

    try:
        feature_names = get_feature_names(preprocessor)
        importances = model.feature_importances_

        if len(feature_names) != len(importances):
            return False

        fi = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_k)
        )

        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
        ax.barh(fi["feature"][::-1], fi["importance"][::-1])
        ax.set_title(f"Top {top_k} Feature Importances")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[WARN] Не вдалося побудувати feature importance plot: {e}")
        return False


def main():
    args = parse_args()

    prepared_dir = Path(args.prepared_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = prepared_dir / "train.csv"
    test_path = prepared_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Prepared train file not found: {train_path.resolve()}")
    if not test_path.exists():
        raise FileNotFoundError(f"Prepared test file not found: {test_path.resolve()}")

    max_depth = parse_none(args.max_depth)
    class_weight = parse_class_weight(args.class_weight)

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    # 1) Load prepared data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Гарантуємо чистоту колонок після CSV
    df_train.columns = [c.strip() for c in df_train.columns]
    df_test.columns = [c.strip() for c in df_test.columns]

    # 2) Target
    y_train = prepare_target(df_train, args.target_col)
    y_test = prepare_target(df_test, args.target_col)

    # 3) Features
    X_train = df_train.drop(columns=[args.target_col], errors="ignore")
    X_test = df_test.drop(columns=[args.target_col], errors="ignore")

    X_train = X_train.drop(columns=[args.id_col], errors="ignore")
    X_test = X_test.drop(columns=[args.id_col], errors="ignore")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # Параметри RandomForest
    rf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
        "n_jobs": -1,
        "class_weight": class_weight,
    }

    pipeline = build_pipeline(numeric_cols, categorical_cols, rf_params)

    with mlflow.start_run(run_name=args.run_name):
        # ---- Tags ----
        mlflow.set_tags(
            {
                "author": args.author,
                "dataset_name": "Telco Customer Churn",
                "dataset_version": args.dataset_version,
                "target_col": args.target_col,
                "model_type": "RandomForestClassifier",
                "lab": "Lab1_MLOps",
            }
        )

        # ---- Params ----
        params_to_log = {
            "prepared_dir": str(prepared_dir),
            "output_dir": str(output_dir),
            "random_state": args.random_state,
            "target_col": args.target_col,
            "id_col": args.id_col,
            "n_estimators": args.n_estimators,
            "max_depth": str(max_depth),
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": str(class_weight),
            "train_rows": int(df_train.shape[0]),
            "test_rows": int(df_test.shape[0]),
            "n_cols_total": int(df_train.shape[1]),
            "n_features_before_encoding": int(X_train.shape[1]),
            "n_numeric_features": int(len(numeric_cols)),
            "n_categorical_features": int(len(categorical_cols)),
        }
        mlflow.log_params(params_to_log)

        # ---- Fit ----
        pipeline.fit(X_train, y_train)

        # ---- Predictions ----
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        y_train_proba = None
        y_test_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_train_proba = pipeline.predict_proba(X_train)[:, 1]
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]

        # ---- Metrics ----
        train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # ---- Local artifacts for DVC ----
        # Модель
        model_path = output_dir / "model.joblib"
        dump(pipeline, model_path)

        # Метрики
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {"train_metrics": train_metrics, "test_metrics": test_metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Параметри
        params_path = output_dir / "train_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(rf_params, f, ensure_ascii=False, indent=2)

        # ---- Plots (saved locally + logged to MLflow) ----
        cm_path = output_dir / "confusion_matrix_test.png"
        save_confusion_matrix_plot(y_test, y_test_pred, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        if y_test_proba is not None:
            roc_path = output_dir / "roc_curve_test.png"
            save_roc_curve_plot(y_test, y_test_proba, roc_path)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        fi_path = output_dir / "feature_importance_top.png"
        fi_logged = save_feature_importance_plot(
            pipeline=pipeline,
            output_path=fi_path,
            top_k=args.top_k_features,
        )
        if fi_logged:
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

        # Summary JSON (локально + MLflow)
        summary = {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "rf_params": rf_params,
            "data_shape": {
                "train_rows": int(df_train.shape[0]),
                "test_rows": int(df_test.shape[0]),
                "cols": int(df_train.shape[1]),
            },
            "feature_counts": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "before_encoding": int(X_train.shape[1]),
            },
        }
        summary_path = output_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(str(summary_path), artifact_path="reports")

        # ---- MLflow model ----
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=None,
        )

        # ---- Print ----
        print("\n✅ Training completed successfully")
        print(f"Saved local model to: {model_path}")
        print(f"Saved metrics to: {metrics_path}")

        print("Train metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("Test metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        run = mlflow.active_run()
        if run:
            print(f"\nMLflow run_id: {run.info.run_id}")
            print(f"Experiment: {args.experiment_name}")


if __name__ == "__main__":
    main()