import json
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd
from nltk.stem import PorterStemmer  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from config import config


def replace_oos_labels(
    df: pd.DataFrame, labels: List[str], label_col: str, oos_label: str = "other"
) -> pd.DataFrame:
    """
    Replace out of scope (OOS) labels.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        labels (List[str]): list of accepted labels.
        label_col (str): name of the dataframe column that has the labels.
        oos_label (str, optional): name of the new label for OOS labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced OOS labels.
    """
    oos_tags = [  # type: ignore
        item for item in df[label_col].unique() if item not in labels  # type: ignore
    ]
    df[label_col] = df[label_col].apply(lambda x: oos_label if x in oos_tags else x)  # type: ignore
    return df


def replace_minority_labels(
    df: pd.DataFrame, label_col: str, min_freq: int, new_label: str = "other"
) -> pd.DataFrame:
    """
    Replace minority labels with another label.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        label_col (str): name of the dataframe column that has the labels.
        min_freq (int): minimum # of data points a label must have.
        new_label (str, optional): name of the new label to replace minority labels. Defaults to "other".

    Returns:
        pd.DataFrame: Dataframe with replaced minority labels.
    """
    labels = Counter(df[label_col].values)  # type: ignore
    labels_above_freq = Counter(  # type: ignore
        label for label in labels.elements() if (labels[label] >= min_freq)  # type: ignore
    )
    df[label_col] = df[label_col].apply(  # type: ignore
        lambda label: label if label in labels_above_freq else None  # type: ignore
    )
    df[label_col] = df[label_col].fillna(new_label)  # type: ignore
    return df


def clean_text(
    text: str, lower: bool = True, stem: bool = False, stopwords: List[str] = config.STOPWORDS
) -> str:
    """
    clean_text _summary_

    Args:
        text (str): raw text to be cleaned.
        lower (bool, optional): whether to lowercase the text.. Defaults to True.
        stem (bool, optional): whether to stem the text. Defaults to False.
        stopwords (List[str], optional): List stop words. Defaults to config.STOPWORDS.

    Returns:
        str: cleaned text.
    """
    # Lower
    if lower:
        text = text.lower()

    # Remove stopwords
    if len(stopwords):
        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Remove links
    text = re.sub(r"http\S+", "", text)

    # Stemming
    if stem:
        stemmer = PorterStemmer()
        text = " ".join(
            [stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")]  # type: ignore
        )  # type: ignore

    return text


def preprocess(df: pd.DataFrame, lower: bool, stem: bool, min_freq: int) -> pd.DataFrame:
    """
    Preprocess the data.

    Args:
        df (pd.DataFrame): Pandas DataFrame with data.
        lower (bool): whether to lowercase the text.
        stem (bool): whether to stem the text.
        min_freq (int): minimum # of data points a label must have.

    Returns:
        pd.DataFrame: Dataframe with preprocessed data.
    """
    # feature engineering
    df["text"] = df.title + " " + df.description  # type: ignore

    # clean text
    df.text = df.text.apply(clean_text, lower=lower, stem=stem)  # type: ignore
    df = replace_oos_labels(
        df=df, labels=config.ACCEPTED_TAGS, label_col="tag", oos_label="other"
    )  # replace OOS labels
    df = replace_minority_labels(
        df=df, label_col="tag", min_freq=min_freq, new_label="other"
    )  # replace labels below min freq

    return df


class LabelEncoder(object):
    """
    Encode labels into unique indices.

    ```python
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y = label_encoder.encode(labels)
    ```
    """

    def __init__(self, class_to_index: Dict[str, int] = {}) -> None:
        """
        Initialize the label encoder.

        Args:
            class_to_index (Dict[str, int], optional): mapping between classes and unique indices. Defaults to {}.
        """
        self.class_to_index = class_to_index or {}  # mutable defaults ;)
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y: List[str]):
        classes: Tuple[str] = np.unique(y)  # type: ignore
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y: List[str]):
        encoded = np.zeros((len(y)), dtype=int)  # type: ignore
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded

    def decode(self, y: List[int]) -> List[str]:
        classes: List[str] = []
        for _, item in enumerate(y):
            classes.append(self.index_to_class[item])
        return classes

    def save(self, file_path: str):
        with open(file_path, "w") as fp:
            contents = {"class_to_index": self.class_to_index}
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def get_data_splits(X: pd.Series, y: np.ndarray, train_size: float = 0.7) -> Tuple[Any]:  # type: ignore
    """
    Generate balanced data splits.

    Args:
        X (pd.Series[float]): input features.
        y (np.ndarray): encoded labels.
        train_size (float, optional): proportion of data to use for training. Defaults to 0.7.

    Returns:
        Tuple[Any]: data splits as Numpy arrays.
    """
    X_train, X_, y_train, y_ = train_test_split(  # type: ignore
        X, y, train_size=train_size, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(  # type: ignore
        X_, y_, train_size=0.5, stratify=y_  # type: ignore
    )
    return X_train, X_val, X_test, y_train, y_val, y_test  # type: ignore
