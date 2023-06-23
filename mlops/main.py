# System package
import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

# Dependent package
import pandas as pd

# Library modules
from mlops import utils, data, train
from config import config
from config.config import logger

warnings.filterwarnings("ignore")

def elt_data() -> None:
    """Extract, load and transform our data assets."""
    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, 'projects.csv'), index=False)
    tags.to_csv(Path(config.DATA_DIR, 'tags.csv'), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()] # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info("✅ Saved data!")

def train_model(args_fp):
    """Train a model given arguments."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(file_path=args_fp))
    artifacts = train.train(df=df, args=args)
    performance = artifacts["performance"]
    print(json.dumps(performance, indent=2))

if __name__ == "__main__":
    pass