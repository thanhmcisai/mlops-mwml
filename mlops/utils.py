import json
import numpy as np
import random
from typing import Any

def load_dict(file_path: str) -> dict:
    """
    Load a dictionary from a JSON's filepath

    Args:
        file_path - str: path to json file

    Returns:
        dict: Dictionary loaded from json file
    """
    with open(file_path, "r") as fp:
        d = json.load(fp)
    return d

def save_dict(dict_data: dict, file_path: str, cls: Any = None, sort_keys: bool = False) -> None:
    """
    Save a dictionary to a specific location.

    Args:
        dict_data (dict): _description_
        file_path (str): _description_
        cls (Any, optional): _description_. Defaults to None.
        sort_keys (bool, optional): _description_. Defaults to False.

    Returns:
        None
    """
    with open(file_path, 'w') as fp:
        json.dump(dict_data, indent=2, fp=fp, cls=cls, sort_keys=sort_keys)

def set_seeds(seed: int = 42) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): _description_. Defaults to 42.

    Returns:
        None
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
