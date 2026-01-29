import numpy as np


def random_split(indices, seed=42, ratios=(0.7, 0.1, 0.2)):
    """
    Randomly split indices into train, validation, and test sets.

    Args:
        indices (array-like): sample indices
        seed (int): random seed
        ratios (tuple): train/val/test ratios (must sum to 1)

    Returns:
        train_idx, val_idx, test_idx
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

    rng = np.random.default_rng(seed)
    indices = np.array(indices)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx
