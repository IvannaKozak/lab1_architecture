from pathlib import Path

import pandas as pd
import yaml

# Run with: "pytest tests/test_pretrain.py -v"


def test_train_data_exists():
    data_path = Path("data/prepared/train.csv")
    assert data_path.exists(), f"Train data not found: {data_path}"


def test_required_columns_exist():
    df = pd.read_csv("data/prepared/train.csv")

    required_cols = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"


def test_target_is_valid():
    df = pd.read_csv("data/prepared/train.csv")

    assert df["Churn"].notna().all(), "Churn contains missing values"

    allowed_values = {"Yes", "No"}
    actual_values = set(df["Churn"].dropna().astype(str).str.strip().unique())

    assert actual_values.issubset(
        allowed_values
    ), f"Unexpected values in Churn: {sorted(actual_values)}"


def test_dataset_not_too_small():
    df = pd.read_csv("data/prepared/train.csv")
    assert df.shape[0] >= 50, f"Too few rows for training: {df.shape[0]}"


def test_config_exists():
    config_path = Path("config/config.yaml")
    assert config_path.exists(), f"Config not found: {config_path}"


def test_config_has_required_sections():
    config_path = Path("config/config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top_keys = {"data", "mlflow", "output", "hydra"}
    missing = required_top_keys - set(cfg.keys())

    assert not missing, f"Missing config sections: {sorted(missing)}"
