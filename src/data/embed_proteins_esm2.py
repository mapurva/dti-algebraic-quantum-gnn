import torch
import esm
import numpy as np
from pathlib import Path


def embed_proteins_esm2(proteins, model_name="esm2_t12_35M_UR50D"):
    """
    proteins: dict {protein_id: sequence}
    returns: dict {protein_id: embedding}
    """
    cache_dir = Path("data/processed/protein_embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    embeddings = {}

    for pid, seq in proteins.items():
        cache_path = cache_dir / f"{pid}.npy"
        if cache_path.exists():
            embeddings[pid] = np.load(cache_path)
            continue

        data = [(pid, seq)]
        _, _, tokens = batch_converter(data)

        with torch.no_grad():
            out = model(tokens, repr_layers=[12], return_contacts=False)
            reps = out["representations"][12]

        # Mean pool over residues (excluding special tokens)
        emb = reps[0, 1:-1].mean(dim=0).cpu().numpy()
        np.save(cache_path, emb)
        embeddings[pid] = emb

    return embeddings
