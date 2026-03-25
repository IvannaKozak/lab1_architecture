from datetime import datetime
from pathlib import Path
import json

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator


PROJECT_DIR = "/opt/airflow/project"
METRICS_PATH = f"{PROJECT_DIR}/models/ci_metrics.json"
QUALITY_THRESHOLD = 0.50


def check_data_exists():
    train_path = Path(f"{PROJECT_DIR}/data/prepared/train.csv")
    test_path = Path(f"{PROJECT_DIR}/data/prepared/test.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")


def choose_branch():
    metrics_file = Path(METRICS_PATH)
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1_value = metrics["quality_gate_metric_value"]

    if f1_value >= QUALITY_THRESHOLD:
        return "register_model"
    return "stop_pipeline"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["lab5", "ml", "airflow"],
) as dag:

    check_data = PythonOperator(
        task_id="check_data",
        python_callable=check_data_exists,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/prepare.py "
            "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv "
            "data/prepared"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/train.py data/prepared models --ci "
            "--model-filename ci_model.joblib "
            "--metrics-filename ci_metrics.json "
            "--confusion-matrix-filename ci_confusion_matrix.png "
            "--max-train-rows 1000 "
            "--max-test-rows 400 "
            "--run-name airflow_train"
        ),
    )

    evaluate_and_branch = BranchPythonOperator(
        task_id="evaluate_and_branch",
        python_callable=choose_branch,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "python src/register_model.py"
        ),
    )

    stop_pipeline = EmptyOperator(
        task_id="stop_pipeline",
    )

    finish = EmptyOperator(
        task_id="finish",
        trigger_rule="none_failed_min_one_success",
    )

    check_data >> prepare_data >> train_model >> evaluate_and_branch
    evaluate_and_branch >> register_model >> finish
    evaluate_and_branch >> stop_pipeline >> finish
    