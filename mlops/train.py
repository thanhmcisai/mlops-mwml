import json
from argparse import Namespace
from typing import Any, Dict

import mlflow  # type: ignore
import numpy as np  # type: ignore
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss  # type: ignore

from mlops import data, evaluate, predict, utils


def train(args: Namespace, df: pd.DataFrame, trial: optuna.trial.Trial = None) -> Dict[str, Any]:  # type: ignore
    """
    Train model on data.

    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial.Trial, optional): optimization trial. Defaults to None.

    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.

    Returns:
        dict: artifacts from the run
    """

    # Setup
    utils.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)  # type: ignore
    df = df[: args.subset]  # None = all samples

    # Preproces
    df = data.preprocess(
        df, lower=args.lower, stem=args.stem, min_freq=args.min_freq  # type: ignore
    )  # type: ignore
    label_encoder = data.LabelEncoder().fit(df.tag)  # type: ignore

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(  # type: ignore
        X=df.text.to_numpy(), y=label_encoder.encode(df.tag)  # type: ignore
    )  # type: ignore
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})  # type: ignore

    # Tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)  # type: ignore
    X_val = vectorizer.transform(X_val)  # type: ignore
    X_test = vectorizer.transform(X_test)  # type: ignore

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)  # type: ignore

    # Model
    model = SGDClassifier(
        loss="log",
        penalty="l2",
        alpha=args.alpha,
        max_iter=1,
        learning_rate="constant",
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True,
    )

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)  # type: ignore
        train_loss = log_loss(y_train, model.predict_proba(X_train))  # type: ignore
        val_loss = log_loss(y_val, model.predict_proba(X_val))  # type: ignore
        if not epoch % 10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )

        # Log
        if not trial:
            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, step=epoch  # type: ignore
            )  # type: ignore

        # Pruning (for optimization in next section)
        if trial:
            trial.report(val_loss, epoch)  # type: ignore
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Threshold
    y_pred = model.predict(X_val)  # type: ignore
    y_prob = model.predict_proba(X_val)  # type: ignore
    args.threshold = np.quantile(  # type: ignore
        [y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25  # type: ignore
    )  # Q1 # type: ignore

    # Evaluation
    other_index = label_encoder.class_to_index["other"]  # type: ignore
    y_prob = model.predict_proba(X_test)  # type: ignore
    y_pred = predict.custom_predict(  # type: ignore
        y_prob=y_prob, threshold=args.threshold, index=other_index
    )  # type: ignore
    performance = evaluate.get_metrics(  # type: ignore
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df  # type: ignore
    )  # type: ignore

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }  # type: ignore


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial.Trial) -> float:
    """
    Objective function for optimization trials.

    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial.Trial): optimization trial.

    Returns:
        float: metric value to be used for optimization.
    """
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    print(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]
