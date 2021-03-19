import os
import numpy as np
import pandas as pd
from typing import Tuple


def create_dir_if_needed(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load(rootdir: str) -> Tuple[pd.DataFrame]:
    X    = pd.read_csv(os.path.join(rootdir, 'train_values.csv'), index_col='building_id')
    y    = pd.read_csv(os.path.join(rootdir, 'train_labels.csv'), index_col='building_id')
    test = pd.read_csv(os.path.join(rootdir, 'test_values.csv'),  index_col='building_id')
    return X, y, test


def save_submission(results: pd.DataFrame, name: str, sample_submission_path: str) -> None:
    submission_format = pd.read_csv(sample_submission_path, index_col='building_id')
    submission = pd.DataFrame(results,
                              columns=submission_format.columns, index=submission_format.index)
    submission.to_csv(name)
