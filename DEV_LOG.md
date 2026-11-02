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
