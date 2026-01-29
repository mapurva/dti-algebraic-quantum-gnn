from data.load_davis import load_davis
from data.build_interactions import build_interaction_table


def main():
    data_dir = "data/raw/davis"
    drugs, proteins, affinities = load_davis(data_dir)

    interactions = build_interaction_table(drugs, proteins, affinities)

    print("Number of interactions:", len(interactions))
    print("\nSample interactions:")
    print(interactions.head())

    print("\nStatistics:")
    print(interactions[["kd", "pkd"]].describe())


if __name__ == "__main__":
    main()
