import numpy as np


class MeanPKDBaseline:
    """
    Dummy baseline that predicts mean pKd from training data.
    """

    def __init__(self):
        self.mean_pkd = None

    def fit(self, y_train):
        """
        Store mean of training pKd values.
        """
        self.mean_pkd = float(np.mean(y_train))

    def predict(self, n_samples):
        """
        Predict mean pKd for all samples.
        """
        assert self.mean_pkd is not None, "Model must be fitted first"
        return np.full(shape=(n_samples,), fill_value=self.mean_pkd)
