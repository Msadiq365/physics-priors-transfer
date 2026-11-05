# Update to registration osf.io/qsh2a: deviations and additions after opening the sealed data

Date: 3 November 2026. Author: Muawiya Sadiq.

This update lists every departure from the frozen registration (freeze commit 331fdff as registered, now 913154a after a metadata-only history rewrite with an identical tree; see HISTORY_NOTE.md). Each is logged with its time in DEV_LOG.md in the repository. Results under both the registered and the corrected procedure are reported in the paper.

## 1. Deviation 1: UORED shaft speed

Registered: UORED shaft speed is the value stored in the first sample of each recording's speed column. Gate G3 checks that the accelerometer's shaft peak lies within 2% of it.

What happened: G3 failed. Only 21 of 40 recordings had their shaft peak within 2% of the stored value. The stored value is identical across all states of a bearing, and on two bearings it lies above the 30 Hz synchronous speed of the drive. The accelerometer shows a clear shaft peak at 29.5 to 29.8 Hz in 36 of 40 recordings, with its second harmonic at about twice that frequency.

Correction, as the registration requires ("stop and correct"): UORED speed now uses the rule already registered for KAIST. This is the largest raw-spectrum peak within plus or minus 3% of the nominal 1800 rpm. A peak on the band edge falls back to the median estimate of 29.70 Hz. The rule uses no labels.

Independent check, also label-blind: in the 2.0 to 5.5 kHz envelope used by the pipeline, the shaft-rate peak is stronger at the corrected speed than at the stored speed in 16 of the 18 recordings where the two differ by more than 2%.

Sensitivity: the confirmatory pair was also run with the registered stored speed. No confirmatory hypothesis is rejected under either speed. Descriptive scores on UORED are higher with the stored speed. Both sets of results are reported side by side.

## 2. Code fixes found while running

1. When a training set had few bearings, an inner cross-validation fold could be empty. Empty folds are now skipped. If no fold is usable, the epoch count falls back to the fixed 30 epochs used for a CWRU source.
2. The fold assignment could place a bearing with both a healthy and a faulty state (UORED) in two folds. Each physical bearing is now assigned once, stratified by its fault class. Paderborn, CWRU, KAIST and HUST are unaffected, because each of their bearings has one class.

Neither fix touches the confirmatory scoring. Both were made before any confirmatory learned-arm result existed.

## 3. Gate results

G2 (registered: C0 detection AUROC at least 0.65 and within-target ceiling at least 0.60):

| Dataset | C0 AUROC | Ceiling | Gate result |
| --- | --- | --- | --- |
| UORED | 0.747 | 0.540 | Fail (ceiling) |
| KAIST | 0.792 | 0.833 | Pass |
| HUST | 0.840 | 0.519 | Fail (ceiling) |
| Paderborn | 0.900 | 0.634 | Pass |
| CWRU | 0.750 | 0.833 | Pass |

The confirmatory tests on Paderborn to UORED were run and are reported as registered, with the G2 result stated beside them.

G4: the label-shuffle M2 control scored 0.178, against 0.222 for always predicting healthy. It passes.

## 4. Exploratory additions (not registered)

WDCNN (Zhang et al., 2017), a standard raw-signal network, was added after the confirmatory analysis. It runs on all 17 settings with the same windows, splits and training recipe. It is reported as exploratory only and enters no test. Result: WDCNN scores below the train-free C0 in 16 of the 17 settings (the exception is HUST type 6205), so it does not change the conclusion.

Repository commits after the freeze record each change.
