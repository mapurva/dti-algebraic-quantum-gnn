# Minimal BLOSUM62 scores (subset, symmetric)
# Values are standard and safe to include

BLOSUM62 = {
    ('A','A'): 4, ('A','V'): 0, ('A','L'): -1, ('A','I'): -1,
    ('V','V'): 4, ('V','L'): 1, ('V','I'): 3,
    ('L','L'): 4, ('L','I'): 2,
    ('I','I'): 4,

    ('D','D'): 6, ('E','E'): 5, ('D','E'): 2,
    ('K','K'): 5, ('R','R'): 5, ('K','R'): 2,
    ('S','S'): 4, ('T','T'): 5, ('S','T'): 1,
    ('F','F'): 6, ('Y','Y'): 7, ('F','Y'): 3,
}

def blosum_score(a, b):
    if (a, b) in BLOSUM62:
        return BLOSUM62[(a, b)]
    if (b, a) in BLOSUM62:
        return BLOSUM62[(b, a)]
    return -4  # default mismatch
