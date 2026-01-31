import torch
from training.learnable_spectral_operator import LearnableSpectralOperator


def build_protein_operator_features(proteins, operator):
    """
    Build learned spectral operator features for proteins.
    """
    features = {}
    operator.eval()

    with torch.no_grad():
        for p_id, seq in proteins.items():
            val = operator(seq)
            features[p_id] = val.unsqueeze(0).numpy()

    return features
