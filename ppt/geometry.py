"""Geometry registry: fault orders from primary-source bearing geometry."""
import csv
import math
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "registry" / "geometry_registry.csv"


def load_registry(path=REGISTRY):
    rows = {}
    for r in csv.DictReader(open(path)):
        rows[r["dataset"]] = r
    return rows


def fault_orders(n, d, D, contact_deg=0.0):
    """BPFO and BPFI as multiples of shaft frequency (outer ring fixed)."""
    c = math.cos(math.radians(contact_deg))
    return {"BPFO": n / 2 * (1 - d / D * c), "BPFI": n / 2 * (1 + d / D * c)}


def orders_for(dataset, registry=None, allow_unverified=False):
    reg = registry or load_registry()
    r = reg[dataset]
    if r["status"] == "to_verify" and not allow_unverified:
        raise ValueError(f"geometry for {dataset} is not verified yet; confirm it before use")
    return fault_orders(int(r["n_elements"]), float(r["ball_d_mm"]), float(r["pitch_D_mm"]),
                        float(r["contact_deg"]))
