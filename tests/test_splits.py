from data.splits import random_split


def main():
    indices = list(range(100))
    train, val, test = random_split(indices, seed=42)

    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))
    print("Overlap check:",
          len(set(train) & set(val)) == 0 and
          len(set(train) & set(test)) == 0 and
          len(set(val) & set(test)) == 0)


if __name__ == "__main__":
    main()
