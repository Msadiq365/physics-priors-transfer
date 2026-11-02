"""Run this on YOUR computer, where the raw CWRU and Paderborn files are stored.

It copies only what the development stage needs into one compact file (about 0.5 GB):
  CWRU: drive-end signal and speed of the 28 records used (normal, IR and OR, 0.007/0.014/0.021 in, 0-3 hp)
  Paderborn: vibration_1 and mean speed for 17 bearings (6 healthy, 6 real inner race, 5 real outer race),
             all four operating conditions, repetitions 1 to 6
No signal is changed; float32 storage only.

Usage (Windows example):
  python export_raw_subset.py --cwru-dir "C:\\data\\CWRU_12k" --pad-dir "C:\\data\\paderborn" --out dev_raw.npz
"""
import argparse, re
from pathlib import Path
import numpy as np
import scipy.io as sio

CWRU_RECORDS = [97, 98, 99, 100, 105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212,
                130, 131, 132, 133, 197, 198, 199, 200, 234, 235, 236, 237]
PAD_BEARINGS = ["K001", "K002", "K003", "K004", "K005", "K006", "KI04", "KI14", "KI16", "KI17", "KI18", "KI21",
                "KA04", "KA15", "KA16", "KA22", "KA30"]
PAD_CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
REPS = range(1, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwru-dir", required=True); ap.add_argument("--pad-dir", required=True)
    ap.add_argument("--out", default="dev_raw.npz")
    a = ap.parse_args()
    store, missing = {}, []
    cw = list(Path(a.cwru_dir).rglob("*.mat"))
    for rec in CWRU_RECORDS:
        hits = sorted(f for f in cw if str(rec) in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", f.stem))
        if not hits:
            missing.append(f"CWRU {rec}"); continue
        m = sio.loadmat(str(hits[0]))
        de = [k for k in m if k.endswith("DE_time")][0]
        rp = [k for k in m if k.endswith("RPM")]
        store[f"CWRU__{rec}__sig"] = m[de].ravel().astype(np.float32)
        store[f"CWRU__{rec}__rpm"] = np.float32(m[rp[0]].ravel()[0]) if rp else np.float32(np.nan)
    pad = {f.stem: f for f in Path(a.pad_dir).rglob("*.mat")}
    for b in PAD_BEARINGS:
        for c in PAD_CONDITIONS:
            for r in REPS:
                name = f"{c}_{b}_{r}"
                if name not in pad:
                    missing.append(name); continue
                m = sio.loadmat(str(pad[name]), struct_as_record=False, squeeze_me=True)
                s = m[[k for k in m if not k.startswith("__")][0]]
                d = {ch.Name: np.asarray(ch.Data) for ch in s.Y}
                store[f"PAD__{name}__sig"] = d["vibration_1"].astype(np.float32)
                store[f"PAD__{name}__rpm"] = np.float32(np.mean(d["speed"]))
    np.savez_compressed(a.out, **store)
    print(f"wrote {a.out}: {len(store) // 2} recordings; missing {len(missing)}")
    if missing:
        print("missing:", ", ".join(missing[:30]), "..." if len(missing) > 30 else "")


if __name__ == "__main__":
    main()
