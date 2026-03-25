import numpy as np
import matplotlib.pyplot as plt
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



#import numpy as np

def expected_calibration_error(y_true, y_pred, y_var, n_bins=10):
    """
    ECE for regression using uncertainty as inverse confidence
    """

    # Convert variance → confidence (higher var = lower confidence)
    confidence = 1 / (1 + y_var)

    errors = np.abs(y_true - y_pred)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        avg_conf = np.mean(confidence[mask])
        avg_err = np.mean(errors[mask])

        ece += np.abs(avg_conf - (1 - avg_err)) * np.sum(mask) / len(y_true)

    return ece
    
    
#import numpy as np
#import matplotlib.pyplot as plt


def plot_reliability_diagram(y_true, y_pred, y_var, n_bins=10, save_path="reliability_diagram.png"):
    """
    Reliability diagram for regression using uncertainty-derived confidence
    """

    # Convert variance → confidence
    confidence = 1 / (1 + y_var)

    errors = np.abs(y_true - y_pred)

    bins = np.linspace(0, 1, n_bins + 1)

    bin_conf = []
    bin_acc = []

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        avg_conf = np.mean(confidence[mask])
        avg_err = np.mean(errors[mask])

        # Convert error → accuracy proxy
        acc = 1 - avg_err

        bin_conf.append(avg_conf)
        bin_acc.append(acc)

    # Plot
    plt.figure(figsize=(5, 5))

    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.plot(bin_conf, bin_acc, marker='o', label="Model")

    plt.xlabel("Confidence")
    plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram")

    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")    
