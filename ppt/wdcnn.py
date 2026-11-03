"""Exploratory arm added after the confirmatory analysis (not registered): WDCNN (Zhang et al., 2017, Sensors 17(2):425),
the most widely used raw-signal baseline for bearing diagnosis. Input: 2,048 raw samples at 12 kHz taken at the start of
each cached window (same windows, units and splits as every other arm), z-scored per segment. Original architecture,
with dropout 0.3 before the classifier to match the shared recipe."""
import json
import numpy as np
import torch.nn as nn

from . import signal as S
from .loaders import dev_manifest, sealed_manifest, load_dev, load_sealed

SEG = 2048


class WDCNN(nn.Module):
    def __init__(self, n_cls=3, dropout=0.3):
        super().__init__()
        def blk(i, o, k, s, p):
            return [nn.Conv1d(i, o, k, s, p), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
        self.body = nn.Sequential(*blk(1, 16, 64, 16, 24), *blk(16, 32, 3, 1, 1), *blk(32, 64, 3, 1, 1),
                                  *blk(64, 64, 3, 1, 1), *blk(64, 64, 3, 1, 0))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 3, 100), nn.ReLU(), nn.Dropout(dropout), nn.Linear(100, n_cls))

    def forward(self, x):
        return self.head(self.body(x))


def build_raw(meta, dev_npz="data/dev_raw.npz", sealed_npz="data/sealed_export.npz"):
    entries = {e["path"]: e for e in dev_manifest(dev_npz) + sealed_manifest(sealed_npz)}
    cache, out = {}, np.zeros((len(meta), SEG), np.float32)
    for i, m in enumerate(meta):
        p = m["path"]
        if p not in cache:
            e = entries[p]
            x, fs, _ = load_sealed(e) if e["dataset"] in ("UORED", "KAIST", "HUST") else load_dev(e)
            cache = {p: S.resample_to_common(x, fs)}          # keep one recording in memory
        seg = cache[p][m["start"]:m["start"] + SEG]
        out[i] = (seg - seg.mean()) / (seg.std() + 1e-12)
    return out
