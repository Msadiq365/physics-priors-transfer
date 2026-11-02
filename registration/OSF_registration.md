# Preregistration: physics priors under bearing-isolated cross-rig transfer

Status: FROZEN for posting. Development work on CWRU and Paderborn is complete. UORED-VAFCLS, KAIST and HUST
have been downloaded, exported and checksummed; no signal from them has been inspected. This document is frozen
(SHA-256 recorded) and posted on OSF before any sealed signal is read.

Author: Muawiya Sadiq (sole author; approved by the supervisor).
Code: repository physics-priors-transfer; freeze commit and checksums in registration/FREEZE.txt.

## 1. Research questions and hypotheses

The study asks whether geometry-normalised envelope-order representations and fault-frequency priors improve
transfer between test rigs under strict whole-bearing isolation, compared with parameter-matched learnable
controls and a train-free envelope baseline.

| ID | Comparison on Paderborn to UORED | Type |
| --- | --- | --- |
| H1 | M3 (dual-axis geometry-normalised spectrum) versus M2 (raw order spectrum), two-sided | Confirmatory, see section 6 |
| H2 | M4 (physics weighting, centres at target geometry) versus M5 (parameter-matched learnable weighting), two-sided | Confirmatory |
| H3 | M4 versus C0 (train-free baseline), two-sided; arm fixed on theoretical grounds, see 5.3 | Confirmatory |
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

H3 uses M4, fixed on theoretical grounds and not from development results. M4 and C0 use the same prior knowledge
(fault-order harmonic locations from the target bearing's geometry); H3 therefore asks whether learning on top of
that knowledge changes diagnostic quality compared with its train-free use. The arm was not chosen from the dry run:
the margin between the two best learned arms there (macro-F1 0.313 for M5, 0.310 for M7) is far below sampling
uncertainty, so choosing on it would carry winner's-curse bias, and the development pairs do not include the
confirmatory target rig. M5 is also the non-physics control built for H2 and would not represent learning on the
physics prior. All other learned arms, and the mean of all learned arms, are compared with C0 as exploratory
estimates (section 6.5). In the dry run C0 was the best arm in both directions, so H3 is genuinely two-sided.

## 6. Statistics

### 6.1 Unit of inference

The independent unit is the physical target bearing (n = 10 for UORED: 5 inner-race and 5 outer-race bearings,
each giving a healthy and a faulty bearing-state unit). Windows and seeds are nested within bearings and are never
treated as independent observations.

### 6.2 Primary endpoint

For each arm and bearing-state unit, the forecast is the 3-class probability vector averaged over all windows and
over the 5 seeds. The unit score is the multiclass Brier score, the sum over classes of (p_k minus y_k) squared
(range 0 to 2, lower is better). The bearing score is the mean of that bearing's two unit scores (ppt/stats.py).
The Brier score is used because it is a strictly proper scoring rule (Gneiting and Raftery, 2007), is bounded, and
is defined for one-hot forecasts. Mean probability of the true class, considered in an earlier draft, is not proper:
it rewards overconfident forecasts, which matters under cross-rig shift.

### 6.3 Probabilistic C0

C0's class scores are the logarithms of (threshold, P_I, P_O), whose largest entry reproduces every C0 decision. Its
forecast is the softmax of these scores divided by one temperature T, fitted by minimising the Brier score with
leave-one-source-bearing-out on the source bearings only (the threshold is refitted within each fold). No target
data are used, and the code asserts that every C0 decision is unchanged. Rationale: scoring hard labels against
probabilistic forecasts on a proper score penalises the format, not the diagnosis (Assel et al., 2017).

### 6.4 Confirmatory tests

For H1, H2 and H3 the per-bearing paired difference in Brier score is tested with an exact two-sided sign-flip
permutation test over all 2^10 = 1,024 sign patterns, statistic the mean difference. Familywise error across H1 to
H3 is controlled by Holm at alpha 0.05. For each hypothesis we report the mean paired difference with a 95% interval
from inverting the sign-flip test, and a percentile bootstrap interval over physical bearings (10,000 resamples),
labelled approximate because there are only 10 clusters.

### 6.5 Estimates without tests, exploratory comparisons and sensitivity analyses

Bearing-level macro-F1, accuracy, mean true-class probability and window-level scores are reported as estimates
with bootstrap intervals, without tests (with 10 bearings, macro-F1 has power below 0.72 even for a difference of
0.4; mdd_sim.py). Every other learned arm (M1, M2, M3, M5, M6, M7) and the mean of all learned arms are compared
with C0 by the same procedure and reported with unadjusted exact p-values, labelled exploratory. Preregistered
sensitivity analyses for H3: (i) C0 entered as one-hot forecasts; (ii) per-bearing correctness (both units
correct) with an exact two-sided sign test on discordant bearings, uninformative if not significant; (iii) the
mean of window-level Brier scores instead of the Brier score of the averaged forecast. The 5-seed average makes
each learned arm a small ensemble, which tends to help Brier scores under shift; this is part of each arm's defined
procedure and is stated as an advantage over the single C0.

### 6.6 Minimum detectable difference and interpretation

Simulated from the development dry run (mdd_brier.py): the paired per-bearing Brier differences between arms have
median SD 0.73 (range 0.05 to 1.60). With 10 bearings and the exact sign-flip test, 80% power needs a true
difference of about 0.75 Brier units at alpha 0.05 and 0.95 at the strictest Holm level (normal and empirical
resampling agree). The confirmatory tests therefore detect only large differences; the interval estimates are the
main result. A non-significant result is reported as "no detectable difference of the registered size", with its
interval, never as equivalence. The development SD is probably pessimistic (it includes arms that collapse to one
class and a 7-bearing CWRU target), and the confirmatory SD is reported.

## 7. Gates on the sealed data

G2: C0 IR/OR AUROC at least 0.65 and within-target ceiling at least 0.60, else the target is reported as not
diagnosable by envelope methods. G3: isolation assertion, checksums, registry check. G4: label-shuffle M2 at most
0.05 above the best constant-prediction macro-F1 on that target. G3 also checks that the UORED accelerometer 1x
shaft peak lies within 2% of the stored speed.

## 8. Reported regardless of outcome

Every pair, arm and seed; confusion matrices; number of classes predicted; window-level scores; the dry run; G1
history; deviations log.

## 9. Freeze record

1. UORED shaft speed: resolved (section 3).
2. M6, KAIST speed refinement, sealed loaders, per-recording geometry, reference pools, controls, probabilistic C0
   and the Brier statistics: implemented and tested on development data and on a synthetic sealed-format file.
3. Supervisor decisions: sole author; inferential metric and H3 arm delegated to the author and fixed in sections 5.3
   and 6 after a literature review.
4. Freeze: commit hash, file checksums and software versions are listed in registration/FREEZE.txt.

## References

Assel, M., Sjoberg, D. D., and Vickers, A. J. (2017). The Brier score does not evaluate the clinical utility of
diagnostic tests or prediction models. Diagnostic and Prognostic Research, 1, 19.
Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning
Research, 7, 1-30.
Gneiting, T., and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the
American Statistical Association, 102(477), 359-378.
Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6, 65-70.
