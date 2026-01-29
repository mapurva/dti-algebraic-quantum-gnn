from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.split_interactions import split_interactions


def main():
    # Path to raw Davis data
    data_dir = "data/raw/davis"

    # Load Davis dataset
    drugs, proteins, affinities = load_davis(data_dir)

    # Build flat interaction table
    interactions = build_interaction_table(drugs, proteins, affinities)

    # Split interactions
    train, val, test = split_interactions(
        interactions,
        seed=42,
        ratios=(0.7, 0.1, 0.2)
    )

    # Basic statistics
    print("Total interactions:", len(interactions))
    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))

    print("\nSample train rows:")
    print(train.head())

    # -------------------------------
    # Proper leakage checks
    # -------------------------------
    # Use interaction identity, NOT DataFrame index
    train_pairs = set(zip(train["drug_id"], train["protein_id"]))
    val_pairs = set(zip(val["drug_id"], val["protein_id"]))
    test_pairs = set(zip(test["drug_id"], test["protein_id"]))

    assert len(train_pairs & val_pairs) == 0, "Train–Val overlap detected"
    assert len(train_pairs & test_pairs) == 0, "Train–Test overlap detected"
    assert len(val_pairs & test_pairs) == 0, "Val–Test overlap detected"

    print("\nNo overlap between splits (interaction-level): OK")


if __name__ == "__main__":
    main()
