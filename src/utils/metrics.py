import numpy as np
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def pearson(y_true, y_pred):
    """
    Pearson correlation coefficient.

    Returns 0.0 if predictions are constant (undefined correlation),
    which is the expected behavior for dummy baselines.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle constant prediction case explicitly
    if np.all(y_pred == y_pred[0]):
        return 0.0

    return pearsonr(y_true, y_pred)[0]


def concordance_index(y_true, y_pred):
    """
    Concordance Index (CI)
    Measures ranking quality for continuous outcomes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                n += 1
                diff_true = y_true[i] - y_true[j]
                diff_pred = y_pred[i] - y_pred[j]

                if diff_pred * diff_true > 0:
                    h_sum += 1
                elif diff_pred == 0:
                    h_sum += 0.5

    return h_sum / n if n > 0 else 0.0
