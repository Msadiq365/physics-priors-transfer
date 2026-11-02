# Physics priors under bearing-isolated cross-rig transfer (development code)

Implements the methodology "Physics priors under bearing-isolated cross-rig transfer" (Claude Docs, 2 Nov 2026).
Status: development stage on CWRU and Paderborn only. UORED-VAFCLS, KAIST and HUST stay sealed until the
protocol is registered on OSF. Separate from the frozen v2.1 benchmark repository.

- `registry/geometry_registry.csv` geometry with sources; rows marked `to_verify` are refused by code until confirmed
- `ppt/signal.py` 12 kHz resampling, 40-revolution windows, squared-envelope order spectra, representations M1, M2, M3, M7
- `ppt/baseline.py` train-free baseline C0
- `ppt/features.py` feature cache, at most 40 windows per bearing state
- `ppt/isolation.py` bearing-isolation assertion and grouped folds
- `tools/export_raw_subset.py` run on the data owner's computer to export the development raw data
- `tests/test_subset.py` physics and isolation checks on a small raw sample
