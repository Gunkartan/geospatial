import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 1000 px = 10 km at Sentinel-2 10 m resolution. Pixels from the same parcel
# must never land on both sides of a split, or validation measures memorization.
BLOCK_SIZE = 1000

def spatial_blocks(df: pd.DataFrame) -> np.ndarray:
    return (df['row'].to_numpy() // BLOCK_SIZE) * 100000 + df['col'].to_numpy() // BLOCK_SIZE

def spatial_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int
):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    return splitter.split(
        np.zeros(len(y)),
        y,
        groups
    )

def spatial_split(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    train_idx, holdout_idx = next(spatial_folds(
        y,
        groups,
        n_splits
    ))

    return train_idx, holdout_idx
