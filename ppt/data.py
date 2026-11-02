"""Load and merge the development and sealed feature caches into one cache dict."""
import json
import numpy as np


def load_cache(paths=("data/dev_cache.npz", "data/sealed_cache.npz")):
    meta, order, hz = [], [], []
    for p in paths:
        z = np.load(p); meta += json.loads(str(z["meta"])); order.append(z["order"]); hz.append(z["hz"])
    return dict(meta=meta, order=np.concatenate(order), hz=np.concatenate(hz))
