import os
import pandas as pd
from typing import Tuple


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
