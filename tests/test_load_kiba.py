from data.load_kiba import load_kiba


def main():
    drugs, proteins, affinities = load_kiba()

    print("Number of drugs:", len(drugs))
    print("Number of proteins:", len(proteins))
    print("Affinity matrix shape:", affinities.shape)

    print("\nAffinity range:")
    print("Min:", affinities.min())
    print("Max:", affinities.max())


if __name__ == "__main__":
    main()
