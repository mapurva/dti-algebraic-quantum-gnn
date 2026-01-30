from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_target_split


def main():
    data_dir = "data/raw/davis"

    drugs, proteins, affinities = load_davis(data_dir)
    interactions = build_interaction_table(drugs, proteins, affinities)

    train, val, test = cold_target_split(interactions, seed=42)

    train_targets = set(train["protein_id"])
    val_targets = set(val["protein_id"])
    test_targets = set(test["protein_id"])

    print("Train targets:", len(train_targets))
    print("Val targets:", len(val_targets))
    print("Test targets:", len(test_targets))

    assert train_targets.isdisjoint(val_targets)
    assert train_targets.isdisjoint(test_targets)
    assert val_targets.isdisjoint(test_targets)

    print("\nCold-target split: NO target overlap ✔")


if __name__ == "__main__":
    main()
