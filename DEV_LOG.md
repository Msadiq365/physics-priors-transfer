# Development log (CWRU and Paderborn only; nothing here touches sealed data)

## 2 Nov 2026
- Raw development data received: data/dev_raw.npz, SHA-256 55e57df6...8f142 (CWRU 28 records; Paderborn 17 bearings x 4 conditions x 6 repetitions = 408). CWRU records 98 and 99 carry no RPM; nominal 1772 and 1750 rpm used.
- Feature cache data/dev_cache.npz: 888 windows, 24 bearing states, no exclusions (all Paderborn speeds within 1% of nominal).
- G1 version 1 (fixed 50 epochs, a shortcut, not the specified procedure): FAIL. PAD M2 0.633, PAD M3 0.505, CWRU M2 0.667, CWRU M3 0.500.
- Diagnosis: C0 train-free under the same folds PAD 0.689, CWRU 0.833. CNN score by epoch peaks early then falls (PAD M3 0.76 at 20 epochs, 0.50 at 50): overtraining. CWRU limited by 4 training fault bearings per fold and OR014, whose fault line is barely above healthy level (known in the literature).
- G1 version 2 started: epoch count chosen by inner grouped cross-validation on training bearings only, as the methodology specifies.
- G1 version 2 (nested epoch selection on training bearings): PAD M2 0.633 FAIL (epochs 60/25/60); PAD M3 0.812 PASS (40/60/40); CWRU M2 0.167 and CWRU M3 0.167 FAIL (epochs 5/5/5 everywhere).
- Reading: on CWRU each outer fold trains on 4 fault bearings, so the inner folds hold out 1 to 2 fault bearings and cannot rank epochs; selection falls to the earliest checkpoint. Nested selection is not workable with a CWRU source.
- Caution: M3 beating M2 inside Paderborn cannot be a geometry effect (all Paderborn bearings share one nominal geometry); it is a representation effect (range 0 to 6 fault orders, two channels, 600 bins) and must not be read as evidence for H1.
- Recipe change for all arms (development decision, before registration): dropout 0.3 before the final layer; window cap raised to 120 (adds little: the export holds about 42 windows per Paderborn bearing); CWRU source uses a fixed 30 epochs because nested selection cannot work with 4 training fault bearings (30 taken from the CWRU epoch curve of the G1 diagnosis, where scores plateaued from about 20 epochs).
- G1 version 3: PAD M2 0.708 PASS (epochs 50/50/10); PAD M3 0.731 PASS (60/30/60); CWRU M2 0.833 PASS; CWRU M3 0.667 FAIL.

## 2 Nov 2026 (evening): sealed-dataset documentation, read before any signal
- Downloaded to the user's Google Drive via Colab: UORED-VAFCLS v5, KAIST (ztmf3m7h5x) v6, HUST v3. File lists and Mendeley descriptions read; no signal inspected.
- UORED: columns Accelerometer, Acoustic, Speed, Load, Temperature Difference; 42 kHz, 10 s; bearings 1-5 inner race, 6-10 outer race, 11-20 ball/cage (healthy states used as reference pool); states 0 healthy, 1 developing, 2 faulty.
- KAIST (Jung et al. 2023, Data in Brief 48:109049): fault-seeded bearing in housing A; channels x_A, y_A, x_B, y_B at 25.6 kHz; 3010 rpm (50.17 Hz); normal 120 s, faulty 60 s; one seeded bearing per size (0.3, 1.0, 3.0 mm). Decision before opening: primary channel y_A (vertical, load direction), x_A as sensitivity; speed nominal 50.17 Hz refined by shaft-peak search within +-3%; first 32 s kept.
- HUST (arXiv 2302.12533 Table 2): geometry per type added to the registry; reproduces published BPFO/BPFI ratios; 'fs' field in each file is the shaft frequency; load codes 0, 200, 400 W; accelerometer vertical.
- Sealed export received (data/sealed_export.npz, SHA-256 530894752c4d11dc676fd71e05ee7a5dd99cb26071c50b381710ed2bdc58932e): UORED 40 recordings (20 healthy; 5 IR and 5 OR bearings x developing and faulty), KAIST 21 (normal, IR and OR at 3 sizes, 3 loads; 32 s; 4 channels), HUST 45 (N, I, O x 5 types x 3 loads). Only the manifest and array lengths were read. The file stays sealed until registration.
- Parameter matching: all learned arms within 1% of 10,211 parameters (M1, M2, M7 first width 20; M3 20; M4, M5 16). Note: G1 was run with width 16 for M2 and M3.
- Dry run (development pairs, 5 seeds, every arm except M6), bearing-level macro-F1 [classes predicted]:
  CWRU->PAD: C0 0.568 [3]; M1 0.152 [1]; M2 0.250 [2]; M3 0.152 [1]; M4 0.174 [1]; M5 0.174 [1]; M7 0.152 [1].
  PAD->CWRU: C0 0.517 [2]; M1 0.200 [1]; M2 0.190 [2]; M3 0.222 [2]; M4 0.222 [2] (window 0.476); M5 0.452 [2]; M7 0.467 [2].
  Time: about 20 s per arm with CWRU as source (fixed epochs), 2.5 to 5 min per arm with Paderborn as source (nested epoch selection). Whole dry run 27 min.
- Reading (development only, not evidence for any hypothesis): learned arms trained on CWRU collapse onto one class; the train-free C0 transfers in both directions and is the best arm in both. CWRU as target has 7 bearing states, so bearing-level scores move in large steps.
- Minimum detectable difference (mdd_sim.py, 10 physical bearings, bearing-level paired permutation): for bearing-level macro-F1, power stays below 0.72 at alpha 0.05 (below 0.45 at the strict Holm level 0.0167) even for a true difference of 0.4; the registered rule (MDD above 0.15 -> estimates, not tests) is triggered for macro-F1.
- Continuous alternative (mean true-class probability per physical bearing, paired): SD of the paired difference between arms 0.22 (CWRU->PAD, 17 bearings) to 0.42 (PAD->CWRU, 7 bearings); MDD for 10 bearings 0.20 at alpha 0.05, 0.23 at 0.0167 (using the 17-bearing estimate).

## 2 Nov 2026 (night): UORED shaft-speed source
- Declared before reading: only the UORED speed column (Hall-effect gear-tooth sensor, carries no fault information) is read, to find its pulse rate; no accelerometer value is read. Summary statistics are computed over all 40 recordings together and per file without class labels attached.
- Finding: each UORED speed column is zero except its first sample, which stores the shaft speed (rpm). The export's resampling scaled that impulse by 2/7, so speed = 3.5 x (sum of the exported column). Recovered speeds: median 1799 rpm, range 1710 to 2191 rpm over the 40 recordings. Registered speed source for UORED: this stored value. Integrity check after opening (G3): the 1x shaft peak of the accelerometer spectrum must lie within 2% of it.

## 2 Nov 2026 (late): remaining code before freezing, tested without touching sealed signals
- Sealed loaders (ppt/loaders.py: sealed_manifest, load_sealed, refine_shaft_hz): bearing ids UORED_<b>, KAIST_<condition> (one seeded bearing per size, loads pooled into one physical bearing), HUST_<type>_<class>; each window carries its own geometry key. Tested on a synthetic file in the sealed-export format built from a CWRU development signal (tests/test_pipeline.py): UORED speed recovered from a spread impulse to within 0.05 Hz; HUST shaft frequency taken from the file; KAIST refinement returns a peak within +-3% of nominal.
- Geometry is now per window (needed for HUST, five bearing types). M3, M6 and M7 build each window on its own geometry's axis; C0 scores each window at its own fault orders.
- M4 now receives each window's own 8 centres appended to the input (training: source centres; test: target centres), so "recentring" is the input itself and needs no switch on the model. Augmentation (jitter, noise) never touches the centre values.
- Reference pools (all arms): PAD_K005 and PAD_K006, and UORED bearings 11 to 20, are never trained on and never scored. 'Developing' UORED states are excluded from all registered analyses (appendix only).
- M6: order spectrum divided by the median healthy-reference spectrum of the same dataset (pool windows only, no labels of scored bearings), then M3's dual-axis representation.
- Controls implemented: M4 forward passes with target centres scaled by 0.95, 0.98, 1.02, 1.05 and 20 random sorted centre sets in 1 to 16 orders (seed 7); label-shuffle M2 (labels permuted across physical source bearings, seed 11).
- Development check, CWRU->PAD, M4, 1 seed: macro-F1 0.259 with true target centres (reproduced exactly by the control path at scale 1.0); every perturbed or random set gives 0.14 (all windows predicted healthy). Learned bump widths are about 0.05 orders, narrower than a 2% shift at the first BPFO harmonic (0.06 orders), so the trained M4 is sharply tied to exact centres. Reading: the perturbation control is sensitive, as intended; development only.
- Label-shuffle M2 CWRU->PAD (5 seeds): 0.167. Constant all-healthy prediction on that target: 0.14. G4 definition made explicit for registration: label-shuffle score at most 0.05 above the best constant-prediction macro-F1 on the target.

## 2 Nov 2026 (night): supervisor decisions and freeze
- Supervisor: proceed; sole author; inferential metric and H3 arm delegated to the author.
- After a literature review: primary endpoint = per-physical-bearing multiclass Brier score (strictly proper; mean true-class probability is improper), exact two-sided sign-flip test over 1,024 sign patterns, Holm over H1 to H3. C0 made probabilistic by softmax of log(threshold, P_I, P_O) with one temperature fitted leave-one-source-bearing-out on source only; decisions unchanged (asserted in code).
- H3 arm fixed a priori as M4 (same physics prior as C0, plus learning). Not chosen from the dry run (M5 0.313 vs M7 0.310 is noise).
- Dry-run Brier (descriptive, pools excluded): CWRU->PAD C0 0.801 (one-hot 0.96), M4 0.743, M5 0.791, M2 0.796; PAD->CWRU C0 0.298 (one-hot 0.357), M5 0.555, M4 0.657. C0 temperatures 0.317 and 0.134.
- MDD for the registered endpoint (mdd_brier.py): paired-difference SD median 0.73; 80% power needs about 0.75 Brier units at alpha 0.05, 0.95 at the strict Holm level. Brier is noisier than true-class probability on these data; kept for propriety, the cost stated in the registration.

## 2 Nov 2026 (23:24): OSF
- Freeze files uploaded by the author to OSF project n8k6c (view-only link received). Awaiting confirmation that a timestamped OSF registration (not only project files) has been submitted; sealed data stay unopened until then.
- Methodology document updated to match the frozen registration (Brier endpoint, sign-flip test, H3 = M4, MDD 0.75/0.95).

## 2 Nov 2026 (23:42): OSF registration submitted
- The author submitted the OSF registration: https://osf.io/qsh2a (from project n8k6c). OSF pages are not machine-readable from this environment; the registration link and time are recorded as given by the author. Code and data checksums verified unchanged against registration/FREEZE.txt immediately before opening.
- Sealed data opened from this point.

## 2 Nov 2026 (23:50): operational definitions fixed before any sealed result is computed
- G2 AUROC: per target bearing-state unit, C0 inner-race score = mean over windows of log P_I, outer-race score = mean of log P_O; AUROC of IR vs rest (P_I) and OR vs rest (P_O), averaged. Threshold-free, so no source information enters.
- G2 ceiling: within-target leave-bearings-out M2 exactly as gate G1 version 3 (outer grouped 3 folds, nested epoch selection; fixed 30 epochs where only one healthy bearing exists, scored then as mean IR/OR recall since the healthy bearing never leaves training).
- G3 speed check (UORED): the largest peak of the raw accelerometer spectrum within +-10% of the stored speed must lie within 2% of it.
- Pairs: as the methodology table (12 directed pairs). M6 only where both source and target have a healthy-reference pool (Paderborn and UORED). M4 runs with its centre controls on every pair. G4 label-shuffle M2 on the confirmatory pair.

## 3 Nov 2026 (00:05): Gate G3 failure and Deviation 1 (UORED shaft speed)
- G3 speed check failed: only 21 of 40 UORED recordings have their accelerometer 1x peak within 2% of the stored speed. The stored value is identical across all states of a bearing (for example 30.34 Hz for H_1_0, I_1_1, I_1_2) and reaches 32.5 to 36.5 Hz on bearings 7 and 8, above the 30 Hz synchronous speed of a 60 Hz four-pole motor; the accelerometer shows a strong peak at 29.5 to 29.8 Hz in 36 of 40 recordings, with its second harmonic at 1.97 to 2.02 times. Reading: the first-sample value is a per-bearing setting or note, not the measured speed during the recording.
- Correction (registered rule "stop and correct"): UORED speed now uses the same label-blind rule registered for KAIST: largest raw-spectrum peak within +-3% of nominal (1800 rpm, 30 Hz); a peak on the band edge falls back to the median interior estimate (29.70 Hz). Result: 29.5 to 30.7 Hz; scored bearings 1 to 10 all 29.5 to 29.8 Hz. Only spectra below 45 Hz and no labels were used to decide this.
- Sensitivity: the confirmatory pair is also run with the registered stored speed (UORED_SPEED=stored), reported beside the main result.
- This changes ppt/loaders.py after the freeze; the change is a reported deviation, to be stated in the paper and added to the OSF registration as an update.
- 3 Nov 00:15: bug fix in choose_epochs: inner folds with no held-out windows are skipped (they occur with few source bearings, UORED and KAIST); no effect on Paderborn or CWRU as source. Runner restarted.
- 3 Nov 00:25: second fix: grouped_folds assigned a bearing with both a healthy and a faulty state (UORED) to up to two folds; it now assigns each physical bearing once, stratified by its fault class (unchanged for Paderborn, CWRU, KAIST, HUST, where every bearing id has one class). If no inner fold is usable, epochs fall back to the fixed 30 used for a CWRU source. Gates and runner restarted; the PAD->UORED C0 result is unaffected (no training).

## 3 Nov 2026 (00:45): gates G2 and G3 on the opened data (results/gates.json)
- G3: sealed checksum OK; registry recomputation matches all nine geometry rows; UORED speed check failed with the stored value and led to Deviation 1 (above).
- G2 (C0 IR/OR AUROC; within-target M2 ceiling): UORED 0.747 / 0.540 FAIL (ceiling); KAIST 0.792 / 0.833 PASS; HUST 0.840 / 0.519 FAIL (ceiling); Paderborn 0.900 / 0.634 PASS; CWRU 0.750 / 0.833 PASS.
- Registered consequence: UORED and HUST are reported as not reaching the within-target ceiling of 0.60, and leave any pooled analysis. The confirmatory tests on Paderborn -> UORED are still run and reported as registered (the registration does not cancel them), with this gate result stated beside them.
- 3 Nov 01:10: confirmatory analysis (results/analysis_main.json): H1 d=-0.181 [-0.36, 0.014] p=0.057 Holm 0.170; H2 d=-0.047 p=0.295 Holm 0.295; H3 d=+0.270 [-0.046, 0.531] p=0.080 Holm 0.170. None rejected. Sensitivity H3: C0 one-hot p=0.037 (unadjusted); correctness sign test p=1.0 (1 vs 0 discordant). G4 label shuffle 0.178 <= 0.222+0.05 PASS. Descriptive reference: a uniform forecast has Brier 0.667.
- 3 Nov 02:40: all 122 runs complete (runs/main, results/analysis_main.json). Sensitivity with the stored UORED speed: runs/sens_stored, results/analysis_sens_stored.json.
