"""Dataset loaders. Each returns (signal, fs, shaft_hz) for one recording, plus a manifest entry.

Manifest fields: dataset, bearing_id (physical specimen), cls (N/IR/OR), state, condition, path.
Loaders for the sealed datasets (UORED, KAIST, HUST) are written only after their documentation has
been read, and must not be run on signals before the protocol is registered.
"""
import re
from pathlib import Path

import numpy as np
import scipy.io as sio

# ---------------- CWRU (12 kHz drive end) ----------------
# record numbers per unit at 0, 1, 2, 3 hp (same table as the frozen study's extraction script)
CWRU_UNITS = {
    "N":     ("N",  [97, 98, 99, 100]),
    "IR007": ("IR", [105, 106, 107, 108]), "IR014": ("IR", [169, 170, 171, 172]),
    "IR021": ("IR", [209, 210, 211, 212]),
    "OR007": ("OR", [130, 131, 132, 133]), "OR014": ("OR", [197, 198, 199, 200]),
    "OR021": ("OR", [234, 235, 236, 237]),
}
CWRU_RPM_FALLBACK = [1797, 1772, 1750, 1730]


def cwru_manifest(root):
    root = Path(root); out = []
    files = list(root.rglob("*.mat"))
    for unit, (cls, recs) in CWRU_UNITS.items():
        for hp, rec in enumerate(recs):
            hits = [f for f in files if str(rec) in re.findall(r"(?<!\d)(\d{2,3})(?!\d)", f.stem)]
            if hits:
                out.append(dict(dataset="CWRU", bearing_id=f"CWRU_{unit}", cls=cls, state="faulty" if cls != "N" else "healthy",
                                condition=f"{hp}hp", path=str(sorted(hits)[0]), rpm_fallback=CWRU_RPM_FALLBACK[hp]))
    return out


def load_cwru(entry):
    m = sio.loadmat(entry["path"])
    de = [k for k in m if k.endswith("DE_time")][0]
    rp = [k for k in m if k.endswith("RPM")]
    rpm = float(m[rp[0]].ravel()[0]) if rp else float(entry["rpm_fallback"])
    return m[de].ravel().astype(float), 12000.0, rpm / 60.0


# ---------------- Paderborn (natural damage subset plus healthy) ----------------
PAD_CLASSES = {
    **{b: ("N", "healthy") for b in ["K001", "K002", "K003", "K004", "K005", "K006"]},
    **{b: ("OR", "real") for b in ["KA04", "KA15", "KA16", "KA22", "KA30"]},
    **{b: ("IR", "real") for b in ["KI04", "KI14", "KI16", "KI17", "KI18", "KI21"]},
    **{b: ("OR", "artificial") for b in ["KA01", "KA03", "KA05", "KA06", "KA07", "KA09"]},
    **{b: ("IR", "artificial") for b in ["KI01", "KI03", "KI05", "KI07", "KI08"]},
}
PAD_NOMINAL_RPM = {"N15": 1500.0, "N09": 900.0}
PAD_RE = re.compile(r"^(N\d\d)_(M\d\d)_(F\d\d)_(K[AIB]?\d\d\d?|K\d\d\d)_(\d+)$")


def pad_manifest(root, origins=("healthy", "real")):
    out = []
    for f in sorted(Path(root).rglob("*.mat")):
        m = PAD_RE.match(f.stem)
        if not m or m.group(4) not in PAD_CLASSES:
            continue
        cls, origin = PAD_CLASSES[m.group(4)]
        if origin not in origins:
            continue
        out.append(dict(dataset="PAD", bearing_id=f"PAD_{m.group(4)}", cls=cls,
                        state="healthy" if cls == "N" else "faulty", origin=origin,
                        condition="_".join(m.group(1, 2, 3)), rep=int(m.group(5)), path=str(f),
                        nominal_rpm=PAD_NOMINAL_RPM[m.group(1)]))
    return out


def load_pad(entry):
    m = sio.loadmat(entry["path"], struct_as_record=False, squeeze_me=True)
    s = m[[k for k in m if not k.startswith("__")][0]]
    d = {c.Name: np.asarray(c.Data, float) for c in s.Y}
    rpm = float(np.mean(d["speed"]))
    if abs(rpm - entry["nominal_rpm"]) / entry["nominal_rpm"] > 0.01:   # label-blind rule from v2.1
        raise ValueError(f"speed {rpm:.1f} rpm more than 1% from nominal")
    return d["vibration_1"], 64000.0, rpm / 60.0


LOADERS = {"CWRU": load_cwru, "PAD": load_pad}


# ---------------- development raw export (tools/export_raw_subset.py output) ----------------
_NPZ = {}


def _npz(path):
    if path not in _NPZ:
        _NPZ[path] = np.load(path)
    return _NPZ[path]


def dev_manifest(npz_path):
    """Manifest for the exported development file: CWRU 28 records, Paderborn 17 bearings."""
    z = _npz(npz_path); out = []
    rec_unit = {rec: (unit, cls, hp) for unit, (cls, recs) in CWRU_UNITS.items() for hp, rec in enumerate(recs)}
    for k in z.files:
        if not k.endswith("__sig"):
            continue
        ds, name, _ = k.split("__")
        if ds == "CWRU":
            unit, cls, hp = rec_unit[int(name)]
            out.append(dict(dataset="CWRU", bearing_id=f"CWRU_{unit}", cls=cls, state="faulty" if cls != "N" else "healthy",
                            condition=f"{hp}hp", path=f"{npz_path}::{k}", rpm_fallback=CWRU_RPM_FALLBACK[hp], fs=12000.0))
        else:
            m = PAD_RE.match(name)
            cls, origin = PAD_CLASSES[m.group(4)]
            out.append(dict(dataset="PAD", bearing_id=f"PAD_{m.group(4)}", cls=cls, state="healthy" if cls == "N" else "faulty",
                            origin=origin, condition="_".join(m.group(1, 2, 3)), rep=int(m.group(5)), path=f"{npz_path}::{k}",
                            nominal_rpm=PAD_NOMINAL_RPM[m.group(1)], fs=64000.0))
    return sorted(out, key=lambda e: e["path"])


def load_dev(entry):
    npz_path, key = entry["path"].split("::")
    z = _npz(npz_path)
    x = z[key].astype(float)
    rpm = float(z[key.replace("__sig", "__rpm")])
    if entry["dataset"] == "CWRU":
        if not np.isfinite(rpm):
            rpm = float(entry["rpm_fallback"])
    elif abs(rpm - entry["nominal_rpm"]) / entry["nominal_rpm"] > 0.01:
        raise ValueError(f"speed {rpm:.1f} rpm more than 1% from nominal")
    return x, entry["fs"], rpm / 60.0


# ---------------- sealed export (tools/export_sealed_colab.py output; 12 kHz already) ----------------
KAIST_NOMINAL_HZ = 3010 / 60.0          # Jung et al. 2023: 3010 rpm, shaft 50.17 Hz
KAIST_CHANNEL = "yA"                     # registered: vertical at housing A (fault-seeded housing); xA as sensitivity


def sealed_manifest(npz_path, kaist_channel=KAIST_CHANNEL):
    z = _npz(npz_path)
    man = __import__("json").loads(bytes(z["__manifest__"]).decode())
    out = []
    for e in man:
        if e["dataset"] == "UORED":
            cls = {"H": "N", "I": "IR", "O": "OR"}[e["letter"]]
            out.append(dict(dataset="UORED", bearing_id=f"UORED_{e['bearing']}", cls=cls, state=e["state"],
                            condition="1", path=f"{npz_path}::UORED__{e['file']}__acc", fs=12000.0,
                            speed_key=f"UORED__{e['file']}__speed", geometry="UORED"))
        elif e["dataset"] == "KAIST":
            cond = e["condition"]
            cls = "N" if cond == "Normal" else ("IR" if cond.startswith("BPFI") else "OR")
            bid = "KAIST_N" if cls == "N" else f"KAIST_{cond}"
            out.append(dict(dataset="KAIST", bearing_id=bid, cls=cls, state="healthy" if cls == "N" else "faulty",
                            condition=f"{e['load_nm']}Nm", path=f"{npz_path}::KAIST__{e['file']}__{kaist_channel}",
                            fs=12000.0, geometry="KAIST"))
        elif e["dataset"] == "HUST":
            cls = {"N": "N", "I": "IR", "O": "OR"}[e["letter"]]
            out.append(dict(dataset="HUST", bearing_id=f"HUST_{e['bearing_type']}_{cls}", cls=cls,
                            state="healthy" if cls == "N" else "faulty", condition=f"load{e['load_code']}",
                            path=f"{npz_path}::HUST__{e['file']}__acc", fs=12000.0, shaft_hz=e["shaft_hz"],
                            geometry=f"HUST_{e['bearing_type']}", bearing_type=e["bearing_type"]))
    return sorted(out, key=lambda e: e["path"])


def refine_shaft_hz(x, fs, nominal_hz, tol=0.03):
    """Label-blind: the largest spectral peak of the raw signal within +-tol of the nominal shaft frequency."""
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f >= nominal_hz * (1 - tol)) & (f <= nominal_hz * (1 + tol))
    return float(f[band][np.argmax(X[band])])


def load_sealed(entry):
    npz_path, key = entry["path"].split("::")
    z = _npz(npz_path)
    x = z[key].astype(float)
    if entry["dataset"] == "UORED":
        rpm = 3.5 * float(z[entry["speed_key"]].astype(float).sum())   # stored first-sample speed, see DEV_LOG
        return x, entry["fs"], rpm / 60.0
    if entry["dataset"] == "KAIST":
        return x, entry["fs"], refine_shaft_hz(x, entry["fs"], KAIST_NOMINAL_HZ)
    return x, entry["fs"], float(entry["shaft_hz"])                     # HUST: file's shaft-frequency field
