from pathlib import Path
import time

import mlflow
from mlflow.tracking import MlflowClient


MODEL_PATH = "models/ci_model.joblib"
MODEL_NAME = "telco_churn_model"
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "Telco_Churn_Lab5"


def main():
    model_path = Path(MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="airflow_register_model"):
        mlflow.log_artifact(str(model_path), artifact_path="model_artifact")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model_artifact/{model_path.name}"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_NAME,
        )

        client = MlflowClient(tracking_uri=TRACKING_URI)

        time.sleep(2)

        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=False,
        )

        print(f"✅ Registered model '{MODEL_NAME}' version {registered_model.version}")
        print("✅ Transitioned to stage: Staging")


if __name__ == "__main__":
    main()
