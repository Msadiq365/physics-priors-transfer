"""Build the feature cache: windows of 40 revolutions, at most CAP per bearing state, evenly spread."""
import hashlib
import json

import numpy as np

from . import signal as S
from .loaders import LOADERS, load_dev, load_sealed

CAP = 40


def build_cache(manifest, out_path, cap=CAP):
    """Compute envelope spectra for every recording, then keep at most `cap` windows per
    (bearing_id, state), chosen evenly through that unit's windows in recording order."""
    rows, ords, hzs, skipped = [], [], [], []
    for e in manifest:
        try:
            x, fs, shaft_hz = ((load_sealed(e) if e["dataset"] in ("UORED", "KAIST", "HUST") else load_dev(e))
                               if "::" in e["path"] else LOADERS[e["dataset"]](e))
        except ValueError as err:
            skipped.append({"path": e["path"], "reason": str(err)}); continue
        x = S.resample_to_common(x, fs)
        starts, n = S.window_starts(len(x), shaft_hz)
        for k, st in enumerate(starts):
            so, sh = S.envelope_spectra(x[st:st + n], shaft_hz)
            ords.append(so); hzs.append(sh)
            rows.append(dict(dataset=e["dataset"], geometry=e.get("geometry", e["dataset"]), bearing_id=e["bearing_id"], cls=e["cls"], state=e["state"],
                             condition=e["condition"], path=e["path"], start=int(st), shaft_hz=shaft_hz))
    keep = []
    units = {}
    for i, r in enumerate(rows):
        units.setdefault((r["bearing_id"], r["state"]), []).append(i)
    for idx in units.values():
        sel = idx if len(idx) <= cap else [idx[int(j)] for j in np.linspace(0, len(idx) - 1, cap).round()]
        keep.extend(sel)
    keep.sort()
    meta = [rows[i] for i in keep]
    np.savez_compressed(out_path, order=np.stack([ords[i] for i in keep]), hz=np.stack([hzs[i] for i in keep]),
                        meta=json.dumps(meta), skipped=json.dumps(skipped))
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    return {"windows": len(keep), "units": len(units), "skipped": skipped, "sha256": digest}
