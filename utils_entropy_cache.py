# utils_entropy_cache.py
import os, numpy as np, torch
from utilities import evaluate          # the same evaluate() you already use

def get_source_entropies(model, loader, device, cache_path="data/source_ents.npy"):
    """
    Returns a 1-D numpy array of clean-set entropies.
    If `cache_path` exists it is loaded, otherwise it is computed and saved.
    """
    if os.path.isfile(cache_path):
        print(f"[cache] loading clean entropies from “{cache_path}”")
        return np.load(cache_path)

    print("[cache] computing clean entropies …")
    ents, *_ = evaluate(model, loader, device)
    ents = np.asarray(ents, dtype=np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, ents)
    print(f"[cache] saved → {cache_path}")
    return ents