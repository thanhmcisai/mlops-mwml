from typing import Dict, List

import numpy as np  # type: ignore


def custom_predict(y_prob: np.ndarray, threshold: float, index: int) -> np.ndarray:
    """
    Custom predict function that defaults
    to an index if conditions are not met.

    Args:
        y_prob (np.ndarray): predicted probabilities
        threshold (float): minimum softmax score to predict majority class
        index (int): label index to use if custom conditions is not met.

    Returns:
        np.ndarray: predicted label indices.
    """
    y_pred = [  # type: ignore
        np.argmax(p) if max(p) > threshold else index for p in y_prob  # type: ignore
    ]  # type: ignore
    return np.array(y_pred)  # type: ignore


def predict(texts: List[str], artifacts: Dict[str, object]) -> List[Dict[str, object]]:
    """Predict tags for given texts."""
    x = artifacts["vectorizer"].transform(texts)
    y_pred = custom_predict(
        y_prob=artifacts["model"].predict_proba(x),
        threshold=artifacts["args"].threshold,
        index=artifacts["label_encoder"].class_to_index["other"],
    )
    tags = artifacts["label_encoder"].decode(y_pred)
    predictions = [
        {
            "input_text": texts[i],
            "predicted_tag": tags[i],
        }
        for i in range(len(tags))
    ]
    return predictions
