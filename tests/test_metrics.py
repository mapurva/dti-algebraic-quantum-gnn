import numpy as np
from src.utils.metrics import rmse, mae, pearson, concordance_index


def main():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    print("RMSE:", rmse(y_true, y_pred))
    print("MAE:", mae(y_true, y_pred))
    print("Pearson:", pearson(y_true, y_pred))
    print("CI:", concordance_index(y_true, y_pred))


if __name__ == "__main__":
    main()
