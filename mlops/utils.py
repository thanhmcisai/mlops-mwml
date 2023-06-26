import json
import random
from typing import Any, Dict

import numpy as np  # type: ignore


def load_dict(file_path: str) -> Dict[str, Any]:
    """Load a dictionary from a JSON's filepath.

    Args:
        file_path (str): location of file.

    Returns:
        Dict: loaded JSON data.
    """
    with open(file_path, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(
    dict_data: Dict[str, Any], file_path: str, cls: Any = None, sort_keys: bool = False
) -> None:
    """
    Save a dictionary to a specific location.

    Args:
        dict_data (dict): data to save.
        file_path (str): location of where to save the data.
        cls (Any, optional): encoder to use on dict data. Defaults to None.
        sort_keys (bool, optional): whether to sort keys alphabetically. Defaults to False.

    Returns:
        None
    """
    with open(file_path, "w") as fp:
        json.dump(dict_data, indent=2, fp=fp, cls=cls, sort_keys=sort_keys)


def set_seeds(seed: int = 42) -> None:
    """
    Set seeds for reproducibility.

    Args:
        seed (int): number to be used as the seed. Defaults to 42.

    Returns:
        None
    """
    # Set seeds
    np.random.seed(seed)  # type: ignore
    random.seed(seed)
