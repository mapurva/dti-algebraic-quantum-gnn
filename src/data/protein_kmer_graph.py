import networkx as nx
from collections import Counter


def build_kmer_graph(sequence, k=3):
    """
    Build a k-mer interaction graph from a protein sequence.
    Nodes: k-mers
    Edges: consecutive k-mer transitions (weighted)
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    transitions = [(kmers[i], kmers[i+1]) for i in range(len(kmers) - 1)]

    G = nx.DiGraph()
    counts = Counter(transitions)

    for (u, v), w in counts.items():
        G.add_edge(u, v, weight=w)

    return G
