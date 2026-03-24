import json
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from joblib import dump
from omegaconf import DictConfig, OmegaConf
from optuna.samplers import TPESampler, RandomSampler

from train import prepare_target, build_pipeline, compute_metrics

# запуск
# TPE: "python src/optimize.py"
# Random: "python src/optimize.py hpo=random"
# mlflow ui


def load_data(cfg: DictConfig):
    prepared_dir = Path(cfg.data.prepared_dir)

    train_path = prepared_dir / "train.csv"
    test_path = prepared_dir / "test.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    target_col = cfg.data.target_col
    id_col = cfg.data.id_col

    y_train = prepare_target(df_train, target_col)
    y_test = prepare_target(df_test, target_col)

    X_train = df_train.drop(columns=[target_col], errors="ignore")
    X_test = df_test.drop(columns=[target_col], errors="ignore")

    X_train = X_train.drop(columns=[id_col], errors="ignore")
    X_test = X_test.drop(columns=[id_col], errors="ignore")

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def get_sampler(cfg: DictConfig):
    sampler_name = cfg.hpo.sampler.lower()
    seed = cfg.hpo.seed

    if sampler_name == "tpe":
        return TPESampler(seed=seed)
    if sampler_name == "random":
        return RandomSampler(seed=seed)

    raise ValueError(f"Unsupported sampler: {cfg.hpo.sampler}")


def suggest_rf_params(trial: optuna.Trial, cfg: DictConfig):
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            cfg.model.n_estimators_min,
            cfg.model.n_estimators_max,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            cfg.model.max_depth_min,
            cfg.model.max_depth_max,
        ),
        "min_samples_split": trial.suggest_int(
            "min_samples_split",
            cfg.model.min_samples_split_min,
            cfg.model.min_samples_split_max,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            cfg.model.min_samples_leaf_min,
            cfg.model.min_samples_leaf_max,
        ),
        "random_state": cfg.hpo.seed,
        "n_jobs": -1,
        "class_weight": cfg.model.class_weight,
    }


def objective(trial: optuna.Trial, cfg: DictConfig):
    X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = load_data(cfg)

    rf_params = suggest_rf_params(trial, cfg)

    pipeline = build_pipeline(numeric_cols, categorical_cols, rf_params)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_proba)
    target_metric = cfg.hpo.metric

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(rf_params)
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.set_tags(
            {
                "trial_number": trial.number,
                "sampler": cfg.hpo.sampler,
                "model_type": "RandomForestClassifier",
                "seed": cfg.hpo.seed,
                "stage": "hpo_trial",
            }
        )

    return metrics[target_metric]


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    sampler = get_sampler(cfg)

    parent_run_name = f"optuna_hpo_rf_{cfg.hpo.sampler}"

    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_params(
            {
                "sampler": cfg.hpo.sampler,
                "n_trials": cfg.hpo.n_trials,
                "direction": cfg.hpo.direction,
                "metric": cfg.hpo.metric,
                "seed": cfg.hpo.seed,
                "prepared_dir": cfg.data.prepared_dir,
                "target_col": cfg.data.target_col,
                "id_col": cfg.data.id_col,
                "model_type": cfg.model.type,
            }
        )

        config_dump_path = Path("temp_hydra_config.yaml")
        OmegaConf.save(cfg, config_dump_path)

        mlflow.log_artifact(str(config_dump_path), artifact_path="config")

        if config_dump_path.exists():
            config_dump_path.unlink()

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
        )
        study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.hpo.n_trials)

        print("\nBest trial:")
        print(f"Best {cfg.hpo.metric.upper()}: {study.best_trial.value:.4f}")
        print("Best params:")
        for key, value in study.best_trial.params.items():
            print(f"{key}: {value}")

        X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = load_data(
            cfg
        )

        best_rf_params = {
            **study.best_trial.params,
            "random_state": cfg.hpo.seed,
            "n_jobs": -1,
            "class_weight": cfg.model.class_weight,
        }

        best_pipeline = build_pipeline(numeric_cols, categorical_cols, best_rf_params)
        best_pipeline.fit(X_train, y_train)

        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_pred, y_proba)

        output_dir = Path(cfg.output.dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"best_model_{cfg.hpo.sampler}.joblib"
        params_path = output_dir / f"best_params_{cfg.hpo.sampler}.json"
        metrics_path = output_dir / f"best_metrics_{cfg.hpo.sampler}.json"

        dump(best_pipeline, model_path)

        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(best_rf_params, f, ensure_ascii=False, indent=2)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_trial.params.items()})
        mlflow.log_metric(f"best_{cfg.hpo.metric}", study.best_trial.value)

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"final_model_{metric_name}", metric_value)

        mlflow.set_tags(
            {
                "sampler": cfg.hpo.sampler,
                "model_type": "RandomForestClassifier",
                "seed": cfg.hpo.seed,
                "stage": "parent_hpo_run",
            }
        )

        mlflow.log_artifact(str(model_path), artifact_path="reports")
        mlflow.log_artifact(str(params_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

        print("\nSaved files:")
        print(f"Model: {model_path}")
        print(f"Params: {params_path}")
        print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
