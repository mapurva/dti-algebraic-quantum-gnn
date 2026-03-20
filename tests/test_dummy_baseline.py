from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.split_interactions import split_interactions
from training.dummy_baseline import MeanPKDBaseline
from utils.metrics import rmse, mae, pearson, concordance_index


def main():
    data_dir = "data/raw/davis"

    # Load and prepare data
    drugs, proteins, affinities = load_davis(data_dir)
    interactions = build_interaction_table(drugs, proteins, affinities)

    train, val, test = split_interactions(interactions, seed=42)

    # Extract targets
    y_train = train["pkd"].values
    y_val = val["pkd"].values
    y_test = test["pkd"].values

    # Train dummy baseline
    model = MeanPKDBaseline()
    model.fit(y_train)

    # Predict
    y_val_pred = model.predict(len(y_val))
    y_test_pred = model.predict(len(y_test))

    # Evaluate
    print("\n=== Dummy Baseline Results ===\n")

    print("Validation Metrics:")
    print("RMSE:", rmse(y_val, y_val_pred))
    print("MAE:", mae(y_val, y_val_pred))
    print("Pearson:", pearson(y_val, y_val_pred))
    print("CI:", concordance_index(y_val, y_val_pred))

    print("\nTest Metrics:")
    print("RMSE:", rmse(y_test, y_test_pred))
    print("MAE:", mae(y_test, y_test_pred))
    print("Pearson:", pearson(y_test, y_test_pred))
    print("CI:", concordance_index(y_test, y_test_pred))


if __name__ == "__main__":
    main()
