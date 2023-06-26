# System package
import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

# Dependent package
import joblib  # type: ignore
import mlflow  # type: ignore
import optuna
import pandas as pd
from numpyencoder import NumpyEncoder  # type: ignore
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger

# Library modules
from mlops import data, predict, train, utils

warnings.filterwarnings("ignore")


def elt_data() -> None:
    """Extract, load and transform our data assets."""
    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)  # type: ignore
    tags = pd.read_csv(config.TAGS_URL)  # type: ignore
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")  # type: ignore
    df = df[df.tag.notnull()]  # drop rows w/ no tag # type: ignore
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info("✅ Saved data!")


def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: bool = False,
) -> None:
    """
    Train a model given arguments.

    Args:
        args_fp (str, optional): location of args. Defaults to "config/args.json".
        experiment_name (str, optional): name of experiment. Defaults to "baselines".
        run_name (str, optional): name of specific run in experiment. Defaults to "sgd".
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))  # type: ignore

    # Train
    args = Namespace(**utils.load_dict(file_path=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id  # type: ignore
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))  # type: ignore

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(
                vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder  # type: ignore
            )
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))  # type: ignore
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))  # type: ignore
            utils.save_dict(performance, Path(dp, "performance.json"))  # type: ignore
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)  # type: ignore
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))  # type: ignore


def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """
    Optimize hyperparameters.

    Args:
        args_fp (str, optional): location of args. Defaults to "config/args.json".
        study_name (str, optional): name of optimization study. Defaults to "optimization".
        num_trials (int, optional): number of trials to run in study. Defaults to 20.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))  # type: ignore

    # Optimize
    args = Namespace(**utils.load_dict(file_path=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()  # type: ignore
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)  # type: ignore
    utils.save_dict({**args.__dict__, **study.best_trial.params}, args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def load_artifacts(run_id: str = "") -> Dict[str, Any]:
    """Load artifacts for a given run_id."""
    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id  # type: ignore
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")  # type: ignore

    # Load objects from run
    args = Namespace(**utils.load_dict(file_path=Path(artifacts_dir, "args.json")))  # type: ignore
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))  # type: ignore
    label_encoder = data.LabelEncoder.load(  # type: ignore
        fp=Path(artifacts_dir, "label_encoder.json")  # type: ignore
    )
    model = joblib.load(Path(artifacts_dir, "model.pkl"))  # type: ignore
    performance = utils.load_dict(file_path=str(Path(artifacts_dir, "performance.json")))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


def predict_tag(text: str = "", run_id: str = "") -> List[Dict[str, Any]]:
    """
    Predict tag for text.

    Args:
        text (str, optional): input text to predict label for. Defaults to "".
        run_id (str, optional): run id to load artifacts for prediction. Defaults to "".

    Returns:
        List[Dict[str, Any]]: Result predict of model
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    print(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    pass
