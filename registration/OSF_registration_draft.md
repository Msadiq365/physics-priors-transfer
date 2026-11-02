# Preregistration (draft for supervisor review): physics priors under bearing-isolated cross-rig transfer

Status: DRAFT, not yet posted. Development work on CWRU and Paderborn is complete. UORED-VAFCLS, KAIST and HUST
have been downloaded, exported and checksummed; no signal from them has been inspected. This document is frozen
(SHA-256 recorded) and posted on OSF before any sealed signal is read.

Authors: Muawiya Sadiq (corresponding), co-authors to be confirmed with the supervisor.
Code: repository physics-priors-transfer, development commit a5dbd1c (final freeze commit recorded at posting).

## 1. Research questions and hypotheses

The study asks whether geometry-normalised envelope-order representations and fault-frequency priors improve
transfer between test rigs under strict whole-bearing isolation, compared with parameter-matched learnable
controls and a train-free envelope baseline.

| ID | Comparison on Paderborn to UORED | Type |
| --- | --- | --- |
| H1 | M3 (dual-axis geometry-normalised spectrum) versus M2 (raw order spectrum) | Confirmatory estimate with test, see section 6 |
| H2 | M4 (physics weighting, centres at target geometry) versus M5 (parameter-matched learnable weighting) | Confirmatory estimate with test |
| H3 | Best learned arm on the development pairs (fixed before opening, see 5.3) versus C0 (train-free baseline), two-sided | Confirmatory estimate with test |
| R1 | M7 (replication of Ma et al. 2024) minus M2 compared with the published gain of 0.11 | Replication criterion, outside the Holm family |
| RQ4 | Why transfer fails where it fails; whether any gain survives a geometry change | Exploratory |

## 2. Data

| Dataset | Role | Units used | Status at registration | SHA-256 of the file used |
| --- | --- | --- | --- | --- |
| CWRU 12 kHz drive end | Development source and target | 7 bearings (1 healthy, 3 IR, 3 OR), 0 to 3 hp | Inspected | dev_raw.npz 55e57df68aa667264c15e5a76f6745ac7e5457614c1ca0d1a2f67c50f95f8142 |
| Paderborn (natural damage plus healthy) | Primary source; secondary target | 17 bearings (6 healthy, 6 IR, 5 OR), 4 conditions, repetitions 1 to 6 | Inspected | same file |
| UORED-VAFCLS v5 | Confirmatory target | Healthy states of bearings 1 to 10 and faulty states of IR bearings 1 to 5 and OR bearings 6 to 10 (20 units, 10 physical bearings); healthy states of bearings 11 to 20 form the healthy-reference pool; developing states only in an appendix | Sealed | sealed_export.npz 530894752c4d11dc676fd71e05ee7a5dd99cb26071c50b381710ed2bdc58932e |
| KAIST (Jung et al. 2023) v6 | Secondary target and source | Normal, IR and OR at 0.3, 1.0, 3.0 mm, 0/2/4 Nm; channel y at housing A (fault-seeded housing), x at housing A as sensitivity | Sealed | same file |
| HUST v3 | Case study | N, I, O for types 6204 to 6208 at 0, 200, 400 W | Sealed | same file |

Original-file SHA-256 values for every sealed recording are stored in the export manifest. Geometry is taken from
registry/geometry_registry.csv (SHA-256 09578f0d58bc24e2c2bfb5f3b8d5ebbb66acb5152f923b210982394b1ac71b2b), with the
primary source of every row.

## 3. Processing (fixed)

Resample to 12 kHz (scipy resample_poly); windows of 40 shaft revolutions; at most 40 windows per bearing state,
spread evenly; band-pass 2.0 to 5.5 kHz (4th-order Butterworth, zero phase); squared Hilbert envelope with mean
removed; Hann window; FFT amplitude; order grid 0 to 40 orders at 0.025; per-window division by the window's own
median. Shaft speed: Paderborn and CWRU from the recorded speed; HUST from the file's shaft-frequency field; KAIST
nominal 50.17 Hz refined by a shaft-peak search within plus or minus 3%; UORED from the speed stored in the first
sample of each recording's speed column (read before registration, see DEV_LOG; median 1799 rpm). Geometry is taken
per recording (HUST has five bearing types).

## 4. Arms and training (fixed)

| Arm | Input | Parameters |
| --- | --- | --- |
| C0 | Train-free: harmonic prominence of BPFO and BPFI (harmonics 1 to 3, plus or minus 2%), threshold from source | none |
| M1 | Envelope spectrum 0 to 1 kHz | 10,131 |
| M2 | Order spectrum 0 to 25 orders | 10,131 |
| M3 | Dual-axis spectrum, 0 to 6 fault orders per channel | 10,311 |
| M4 | M2 with 8 Gaussian bumps at BPFO and BPFI harmonics 1 to 4; source centres in training, target centres at test | 10,211 |
| M5 | M2 with 8 learnable-centre bumps | 10,211 |
| M6 | M3 after division by the median healthy-reference spectrum of the same dataset (Paderborn and UORED only) | 10,311 |
| M7 | Ma et al. (2024) replication: BPFO-normalised axis 0 to 8, Gaussian weights at outer orders 1, 2 and inner orders r, 2r | 10,131 |

Training: Adam, learning rate 1e-3, weight decay 1e-4, batch 64, class-weighted cross-entropy, amplitude jitter 0.9
to 1.1, Gaussian noise 0.05, dropout 0.3, one thread per process, 5 seeds; epochs chosen by inner grouped
cross-validation over source bearings (checkpoints 5 to 60), except CWRU as source (fixed 30). Weights saved.
Controls: 20 random-centre sets and target geometry perturbed by minus 5, minus 2, plus 2, plus 5% on trained M4;
label-shuffle M2. Reference pools (Paderborn K005 and K006; UORED healthy bearings 11 to 20) are used only by M6
and are never trained on or scored by any arm. In M4 each window carries its own centres as input.

## 5. Development results already known (reported in full in the paper)

### 5.1 Gate G1 (within-dataset, leave-bearings-out)

Paderborn: M2 0.708, M3 0.731 (pass). CWRU: M2 0.833 (pass), M3 0.667 (fail). Every CWRU-source result for M3
carries this flag. History of recipe changes made to pass G1 is in DEV_LOG.md.

### 5.2 Dry run (development pairs)

Bearing-level macro-F1, CWRU to Paderborn: C0 0.568; learned arms 0.15 to 0.25. Paderborn to CWRU: C0 0.517;
M5 0.452, M7 0.467, other learned arms 0.19 to 0.22.

### 5.3 Arm for H3

The best learned arm on the development pairs (mean bearing-level macro-F1 over both directions) is fixed before
opening: M5 (mean 0.313), just ahead of M7 (0.310). The margin is negligible, so this choice is weak. To confirm with
the supervisor: whether H3 should instead use a pre-chosen arm for theoretical reasons (for example M3, the
representation the study is about).

## 6. Statistics

Unit of inference: the physical target bearing (both its units resampled or permuted together).

The minimum detectable difference was estimated before registration (mdd_sim.py). With 10 physical bearings,
bearing-level macro-F1 cannot detect differences below about 0.4 with 80% power. The registered rule therefore
applies: macro-F1 differences are reported as estimates with 95% bootstrap intervals, not as tests.

Inferential metric (proposed, to confirm with the supervisor): mean probability assigned to the true class,
averaged per physical bearing. With the development variance, its minimum detectable paired difference for 10
bearings is about 0.20 at alpha 0.05 and 0.23 at the strictest Holm level. H1 to H3 are tested on this metric with a
paired permutation over bearings (10,000 permutations), Holm-corrected across H1 to H3, two-sided.

No equivalence claim is made: with 10 bearings an equivalence margin of 0.05 cannot be supported. A non-significant
result is reported as "no detectable difference of the registered size", with its interval.

## 7. Gates on the sealed data

G2: C0 IR/OR AUROC at least 0.65 and within-target ceiling at least 0.60, else the target is reported as not
diagnosable by envelope methods. G3: isolation assertion, checksums, registry check. G4: label-shuffle M2 at most
0.05 above the best constant-prediction macro-F1 on that target. G3 also checks that the UORED accelerometer 1x
shaft peak lies within 2% of the stored speed.

## 8. Reported regardless of outcome

Every pair, arm and seed; confusion matrices; number of classes predicted; window-level scores; the dry run; G1
history; deviations log.

## 9. To complete before this draft is frozen

1. Done: UORED shaft speed (section 3).
2. Done: M6, the KAIST speed refinement, loaders for the sealed export, per-recording geometry, the reference
   pools, and the perturbation, random-centre and label-shuffle controls; tested on development data and on a
   synthetic file in the sealed format (tests/test_pipeline.py).
3. Supervisor decisions: co-authors; the inferential metric in section 6; the arm for H3.
4. Final freeze: commit hash, file checksums, software versions (Python 3.13.16, torch 2.14.1, numpy 2.5.3, scipy 1.18.1).
