import json
import os
from pathlib import Path

F1_THRESHOLD = float(os.getenv("F1_THRESHOLD", "0.50"))


def test_ci_model_exists():
    model_path = Path("models/ci_model.joblib")
    assert model_path.exists(), f"Model file not found: {model_path}"


def test_ci_metrics_exists():
    metrics_path = Path("models/ci_metrics.json")
    assert metrics_path.exists(), f"Metrics file not found: {metrics_path}"


def test_ci_confusion_matrix_exists():
    cm_path = Path("models/ci_confusion_matrix.png")
    assert cm_path.exists(), f"Confusion matrix file not found: {cm_path}"


def test_metrics_have_required_keys():
    metrics_path = Path("models/ci_metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_keys = {
        "train_metrics",
        "test_metrics",
        "quality_gate_metric_name",
        "quality_gate_metric_value",
    }

    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing keys in metrics file: {sorted(missing)}"


def test_test_metrics_have_f1():
    metrics_path = Path("models/ci_metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "f1" in metrics["test_metrics"], "Missing f1 in test_metrics"


def test_quality_gate_passes():
    metrics_path = Path("models/ci_metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metric_name = metrics["quality_gate_metric_name"]
    metric_value = metrics["quality_gate_metric_value"]

    assert (
        metric_name == "f1"
    ), f"Expected quality gate metric to be 'f1', got '{metric_name}'"
    assert (
        metric_value >= F1_THRESHOLD
    ), f"Quality gate failed: f1={metric_value:.4f} < threshold={F1_THRESHOLD:.2f}"
