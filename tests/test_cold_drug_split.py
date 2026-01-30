from data.load_davis import load_davis
from data.build_interactions import build_interaction_table
from data.cold_splits import cold_drug_split


def main():
    data_dir = "data/raw/davis"

    drugs, proteins, affinities = load_davis(data_dir)
    interactions = build_interaction_table(drugs, proteins, affinities)

    train, val, test = cold_drug_split(interactions, seed=42)

    train_drugs = set(train["drug_id"])
    val_drugs = set(val["drug_id"])
    test_drugs = set(test["drug_id"])

    print("Train drugs:", len(train_drugs))
    print("Val drugs:", len(val_drugs))
    print("Test drugs:", len(test_drugs))

    assert train_drugs.isdisjoint(val_drugs)
    assert train_drugs.isdisjoint(test_drugs)
    assert val_drugs.isdisjoint(test_drugs)

    print("\nCold-drug split: NO drug overlap ✔")


if __name__ == "__main__":
    main()
