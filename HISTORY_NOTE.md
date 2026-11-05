# Commit history note

On 5 November 2026 the commit metadata of this repository was rewritten so that every commit carries the author's name and e-mail and no tool-generated message trailers. File contents were not touched: every rewritten commit has exactly the same tree hash as before, so the frozen code is byte-for-byte identical.

The OSF registration (osf.io/qsh2a) records the freeze by its old commit ID. Use the table below to match it.

| Role | Old commit (as registered) | New commit | Tree hash (unchanged) |
| --- | --- | --- | --- |
| Registered freeze | 331fdff | 913154a | c26089060fefbb8afe7588904e09f7c38bdf25bc |
| Branch prereg-freeze | 6890a07 | 9f52f5c | 7998f698dbe195256c6df153bb3ec53df416d0c5 |

To verify: `git rev-parse 913154a^{tree}` prints the tree hash above.

## Full mapping

| Old | New | Subject |
| --- | --- | --- |
| 78172d6 | 0113770 | Title wording: physics-guided deep networks |
| 612f67d | 9734942 | Single-line title in graphical abstract |
| 2db6509 | f54921f | Figure 1 workflow as SVG |
| 337a5a6 | 1b39ce7 | Annotated, colour-blind-safe figure versions |
| fc17814 | 77efa23 | Annotated, colour-blind-safe figure versions |
| 4bbc27f | 8d1eb1e | Ignore derived data arrays |
| 5bb0c64 | 78a9ae0 | README and licence for release; manuscript drafts no longer tracked |
| b427c8a | f91a9b1 | Submission package: Word manuscript, highlights, graphical abstract, figures |
| 832b25e | 97f331a | Exploratory per-arm tuning, graphical abstract, highlights |
| 6c0a58d | 48e7067 | Publication figures, cross-rig analysis, supplementary table |
| 19abeb3 | 23b88fb | Exploratory WDCNN arm; OSF deviations update |
| 0da70a3 | e6e6317 | Speed adjudication logged |
| 5b2dcf0 | 66cbd7a | Stored-speed sensitivity |
| ddb1ba1 | c913134 | All 12 pairs and HUST case study: results and analysis |
| e677f82 | 388b139 | Gates and confirmatory results logged |
| 86907c2 | f1ab528 | Fix fold assignment for two-state bearings; empty inner folds |
| 883d9c1 | bcdfd59 | Deviation 1: UORED speed from label-blind spectral rule after G3 failure; gates and runner scripts |
| 687dc9e | 263b012 | OSF registration qsh2a recorded; opening sealed data |
| 7986a19 | 9e4c5a9 | Log OSF upload; sealed data still closed |
| 6890a07 | 9f52f5c | Freeze record |
| 331fdff | 913154a | Registered endpoint (per-bearing Brier, exact sign-flip, Holm), probabilistic C0, H3 arm M4, Brier MDD; registration finalised |
| 38dfb3e | 275039b | Sealed loaders, per-window geometry, M6, reference pools, M4 controls, label-shuffle; tests on synthetic sealed-format file |
| 9684e8e | 5916dee | Registration draft for supervisor review |
| a5dbd1c | 3098a8d | Minimum detectable difference simulation and log |
| 18a5a20 | 919a18f | Arms M1-M7 with parameter matching, transfer runner, development dry run |
| 8eb4e01 | 0ec8285 | Record sealed export checksum and manifest check |
| 4b390c7 | b5646a6 | KAIST geometry and channel decision from the primary article; log |
| 5c70328 | c511afa | Geometry registry: HUST types, UORED and KAIST checked; Colab export script for sealed datasets |
| 04245c8 | fe16e08 | Shared recipe: dropout 0.3, CWRU fixed epochs; G1 version 3 results |
| ec9ba9b | 0d4c74e | Log G1 version 2 results |
| 5a1124d | 6fd5602 | Dev data loader, feature cache, CNN, gate G1 runs and diagnosis log |
| 7a21408 | 5ac88b3 | Development pipeline: registry, signal processing, representations, C0 baseline, isolation checks |
