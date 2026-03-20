from data.load_davis import load_davis, inspect_davis


def main():
    data_dir = "data/raw/davis"
    drugs, proteins, affinities = load_davis(data_dir)
    inspect_davis(drugs, proteins, affinities)


if __name__ == "__main__":
    main()
