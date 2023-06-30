from typing import Any, Dict, List

import numpy as np  # type: ignore
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support  # type: ignore
from snorkel.slicing import PandasSFApplier  # type: ignore
from snorkel.slicing import slicing_function  # type: ignore


@slicing_function()
def nlp_cnn(x: pd.DataFrame):
    """NLP Projects that use convolution."""
    nlp_projects = "natural-language-processing" in x.tag  # type: ignore
    convolution_projects = "CNN" in x.text or "convolution" in x.text  # type: ignore
    return nlp_projects and convolution_projects


@slicing_function()
def short_text(x: pd.DataFrame):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words # type: ignore


def get_slice_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, slices: np.recarray
) -> Dict[str, Dict[str, Any]]:
    """
    Generate metrics for slices of data.

    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        slices (np.recarray): generated slices.

    Returns:
        Dict[str, Dict[str, Any]]: slice metrics
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List[str], df: pd.DataFrame = pd.DataFrame()
) -> Dict[str, Dict[str, Any]]:
    """
    Performance metrics using ground truths and predictions.

    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        classes (List[str]): list of class labels.
        df (pd.DataFrame, optional): dataframe to generate slice metrics on. Defaults to pd.DataFrame().

    Returns:
        Dict[str, Dict[str, Any]]: performance metrics
    """
    # Performance
    metrics: Dict[str, Dict[str, Any]] = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))  # type: ignore

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],  # type: ignore
            "recall": class_metrics[1][i],  # type: ignore
            "f1": class_metrics[2][i],  # type: ignore
            "num_samples": np.float64(class_metrics[3][i]),  # type: ignore
        }

    # Slice metrics
    if df is not None and not df.empty: # type: ignore
        slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
        metrics["slices"] = get_slice_metrics(
            y_true=y_true, y_pred=y_pred, slices=slices
        )  # type: ignore

    return metrics
