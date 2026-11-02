"""Run in Google Colab (Drive mounted). Exports the three sealed datasets into ONE compact file.

What it does, mechanically and blind to labels:
  - keeps only the recordings the methodology uses (healthy, inner race, outer race; UORED also keeps
    the 'developing' state for the appendix),
  - keeps only the vibration channels (UORED also the speed channel, which carries no fault signal),
  - resamples to 12 kHz with exactly the pipeline's polyphase filter (scipy resample_poly, same ratios),
  - stores float32 plus each file's SHA-256 so the originals can be traced.
It never plots, summarises or analyses a signal. Do not inspect the output either.

Usage in a Colab cell:
  !python export_sealed_colab.py --base /content/drive/MyDrive/bearing_datasets --out /content/drive/MyDrive/sealed_export.npz
"""
import argparse, glob, hashlib, json, os, re
from math import gcd
import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

FS = 12000


def to12k(x, fs):
    fs = int(round(fs))
    if fs == FS:
        return np.asarray(x, np.float64)
    g = gcd(FS, fs)
    return resample_poly(np.asarray(x, np.float64), FS // g, fs // g)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kaist-seconds", type=float, default=32.0)
    a = ap.parse_args()
    store, manifest = {}, []

    # ---- UORED: H_<b>_0 healthy (20 bearings); I_<b>_<1|2>, O_<b>_<1|2> (1 developing, 2 faulty) ----
    for path in sorted(glob.glob(os.path.join(a.base, "UORED", "*", "*.mat"))):
        name = os.path.basename(path)[:-4]
        m = re.match(r"^([HIO])_(\d+)_(\d)$", name)
        if not m:
            continue                       # ball and cage fault states are not used
        arr = sio.loadmat(path)[name]      # columns: Accelerometer, Acoustic, Speed, Load, Temperature Difference
        store[f"UORED__{name}__acc"] = to12k(arr[:, 0], 42000).astype(np.float32)
        store[f"UORED__{name}__speed"] = to12k(arr[:, 2], 42000).astype(np.float32)
        manifest.append(dict(dataset="UORED", file=name, letter=m.group(1), bearing=int(m.group(2)),
                             state={"0": "healthy", "1": "developing", "2": "faulty"}[m.group(3)],
                             fs_original=42000, sha256=sha(path)))

    # ---- KAIST: Normal, BPFI_03/10/30, BPFO_03/10/30 at 0, 2, 4 Nm; all four channels, first N seconds ----
    for path in sorted(glob.glob(os.path.join(a.base, "KAIST", "*.mat"))):
        name = os.path.basename(path)[:-4]
        m = re.match(r"^(\d)Nm_(Normal|BPFI_\d\d|BPFO_\d\d)$", name)
        if not m:
            continue                       # misalignment and unbalance are not used
        s = sio.loadmat(path)["Signal"][0, 0]
        inc = float(s["x_values"][0, 0]["increment"].ravel()[0])
        y = s["y_values"][0, 0]["values"]  # columns: x_A, y_A, x_B, y_B (g)
        fs = round(1.0 / inc)
        y = y[: int(a.kaist_seconds * fs)]
        for c, cname in enumerate(["xA", "yA", "xB", "yB"]):
            store[f"KAIST__{name}__{cname}"] = to12k(y[:, c], fs).astype(np.float32)
        manifest.append(dict(dataset="KAIST", file=name, load_nm=int(m.group(1)), condition=m.group(2),
                             fs_original=fs, seconds_kept=a.kaist_seconds, sha256=sha(path)))

    # ---- HUST: N, I, O for bearing types 6204-6208 at loads 0, 200, 400 W ----
    for path in sorted(glob.glob(os.path.join(a.base, "HUST", "*", "*.mat"))):
        name = os.path.basename(path)[:-4]
        m = re.match(r"^([NIO])([4-8])0([024])$", name)
        if not m:
            continue                       # ball and combined faults are not used
        d = sio.loadmat(path)
        store[f"HUST__{name}__acc"] = to12k(d["data"].ravel(), 51200).astype(np.float32)
        manifest.append(dict(dataset="HUST", file=name, letter=m.group(1), bearing_type=f"620{m.group(2)}",
                             load_code=int(m.group(3)), shaft_hz=float(np.ravel(d["fs"])[0]),
                             fs_original=51200, sha256=sha(path)))

    store["__manifest__"] = np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8)
    np.savez_compressed(a.out, **store)
    counts = {ds: sum(e["dataset"] == ds for e in manifest) for ds in ("UORED", "KAIST", "HUST")}
    print("wrote", a.out, os.path.getsize(a.out) // 1_000_000, "MB; recordings:", counts)
    print("expected: UORED 40 (20 healthy + 10 inner + 10 outer), KAIST 21, HUST 45")


if __name__ == "__main__":
    main()
