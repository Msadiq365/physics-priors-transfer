# Physics priors and cross-rig transfer in bearing fault diagnosis

Code, registry, logs and results for the study "Physics priors and cross-rig transfer in bearing fault diagnosis:
a bearing-isolated benchmark against envelope analysis" (M. Sadiq, Federal University of Technology Babura).

Preregistration: https://osf.io/qsh2a (frozen at branch `prereg-freeze`, commit 913154a, registered as 331fdff before a metadata-only history rewrite, see `HISTORY_NOTE.md`; deviations in
`registration/OSF_update_deviations.md` and `DEV_LOG.md`).

## Data

No data are redistributed here. Download the public datasets from their sources:
CWRU Bearing Data Center (12 kHz drive end), Paderborn KAt-DataCenter, UORED-VAFCLS (Mendeley Data y2px5tg92h),
KAIST (Mendeley Data ztmf3m7h5x) and HUST (Mendeley Data cbv7jyx4p9). `tools/export_raw_subset.py` and
`tools/export_sealed_colab.py` build `data/dev_raw.npz` and `data/sealed_export.npz`; their SHA-256 values are in
`registration/FREEZE.txt`.

## Reproducing the results

```
python3 build_dev_cache.py          # development feature cache (CWRU, Paderborn)
python3 build_sealed_cache.py       # target feature cache (UORED-VAFCLS, KAIST, HUST)
python3 gates.py                    # gates G2 and G3
python3 run_all.py 0 1              # all registered runs (122 jobs)
python3 run_sens_stored.py          # sensitivity: stored UORED speed
python3 run_wdcnn.py 0 1            # exploratory WDCNN arm
python3 tune_run.py 0 1             # exploratory per-arm tuning
python3 analysis.py                 # confirmatory tests and tables (results/analysis_main.json)
python3 make_figures.py             # Figures 1 to 7
```

Python 3.13, torch 2.14, numpy 2.5, scipy 1.18; one thread per process (`TRAIN_THREADS=1`).

## Layout

- `ppt/` pipeline: signal processing, arms M1 to M7, train-free baseline C0, WDCNN, isolation checks, training, statistics
- `registry/geometry_registry.csv` bearing geometry with sources
- `registration/` frozen registration, freeze record, deviation update
- `results/`, `runs/tuned/` result files; `figures/`, `submission/` figures and supplementary table
- `DEV_LOG.md` dated log of every decision, before and after registration

## Licence

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
