import pandas as pd
import numpy as np


def split_interactions(df, train_ratio=0.7, val_ratio=0.1, seed=42):
    """
    Random split with fixed seed for reproducibility
    """

    np.random.seed(seed)

    indices = np.random.permutation(len(df))

    train_end = int(train_ratio * len(df))
    val_end = train_end + int(val_ratio * len(df))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df