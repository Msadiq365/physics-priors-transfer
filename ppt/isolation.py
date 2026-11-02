"""Gate G3: whole-bearing isolation is asserted in code, not assumed."""


def assert_isolated(train_meta, test_meta):
    tr = {m["bearing_id"] for m in train_meta}
    te = {m["bearing_id"] for m in test_meta}
    shared = tr & te
    if shared:
        raise AssertionError(f"bearing isolation violated: {sorted(shared)} in both train and test")
    return True


def grouped_folds(meta, n_folds=3, seed=0):
    """Folds over physical bearings, stratified by class: each fold holds out whole bearings of every class."""
    import numpy as np
    rng = np.random.default_rng(seed)
    by_cls = {}
    for b, c in {(m["bearing_id"], m["cls"]) for m in meta}:
        by_cls.setdefault(c, []).append(b)
    folds = [set() for _ in range(n_folds)]
    for c, bs in sorted(by_cls.items()):
        if len(bs) < n_folds:      # too few bearings (e.g. one CWRU healthy bearing): always trains
            continue
        bs = sorted(bs); rng.shuffle(bs)
        for k, b in enumerate(bs):
            folds[k % n_folds].add(b)
    return folds
