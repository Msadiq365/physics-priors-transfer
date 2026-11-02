# Development log (CWRU and Paderborn only; nothing here touches sealed data)

## 2 Nov 2026
- Raw development data received: data/dev_raw.npz, SHA-256 55e57df6...8f142 (CWRU 28 records; Paderborn 17 bearings x 4 conditions x 6 repetitions = 408). CWRU records 98 and 99 carry no RPM; nominal 1772 and 1750 rpm used.
- Feature cache data/dev_cache.npz: 888 windows, 24 bearing states, no exclusions (all Paderborn speeds within 1% of nominal).
- G1 version 1 (fixed 50 epochs, a shortcut, not the specified procedure): FAIL. PAD M2 0.633, PAD M3 0.505, CWRU M2 0.667, CWRU M3 0.500.
- Diagnosis: C0 train-free under the same folds PAD 0.689, CWRU 0.833. CNN score by epoch peaks early then falls (PAD M3 0.76 at 20 epochs, 0.50 at 50): overtraining. CWRU limited by 4 training fault bearings per fold and OR014, whose fault line is barely above healthy level (known in the literature).
- G1 version 2 started: epoch count chosen by inner grouped cross-validation on training bearings only, as the methodology specifies.
