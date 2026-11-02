"""Signal processing shared by every arm (methodology section 'Signal processing').

Steps: resample to 12 kHz, cut windows of a fixed number of shaft revolutions, band-pass 2.0-5.5 kHz,
squared Hilbert envelope (mean removed), Hann window, FFT amplitude, map to shaft orders with the
window's own speed, normalise by the window's own median. All functions are deterministic.
"""
from math import gcd

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt

FS = 12000.0               # common sampling rate (Hz)
BAND = (2000.0, 5500.0)    # demodulation band (Hz), carried over from the frozen study
REVS = 40                  # revolutions per window -> order resolution 1/40 = 0.025
ORDER_STEP = 1.0 / REVS
BASE_MAX_ORDER = 40.0      # cached order range; representations take slices of it
BASE_GRID = np.arange(0.0, BASE_MAX_ORDER, ORDER_STEP)      # 1,600 bins
HZ_GRID = np.linspace(0.0, 1000.0, 1000, endpoint=False)    # M1 input, 1,000 bins
_SOS = butter(4, BAND, btype="band", fs=FS, output="sos")


def resample_to_common(x, fs):
    """Polyphase resampling with anti-alias filtering to the common 12 kHz rate."""
    fs_i = int(round(fs))
    if fs_i == int(FS):
        return np.asarray(x, float)
    g = gcd(int(FS), fs_i)
    return resample_poly(np.asarray(x, float), int(FS) // g, fs_i // g)


def window_starts(n_samples, shaft_hz, revs=REVS):
    """Non-overlapping windows of `revs` shaft revolutions; returns (starts, window length)."""
    n_win = int(round(revs * FS / shaft_hz))
    starts = np.arange(0, n_samples - n_win + 1, n_win)
    return starts, n_win


def envelope_spectra(seg, shaft_hz):
    """Squared-envelope spectrum of one segment on the order grid and on the Hz grid.

    Returns (Sn_order, Sn_hz): each divided by its own median, so values are linear ratios to the
    window's typical level. Models apply log1p; the train-free baseline uses the ratios directly.
    """
    seg = seg - seg.mean()
    env = np.abs(hilbert(sosfiltfilt(_SOS, seg))) ** 2
    env = env - env.mean()
    n = len(env)
    amp = np.abs(np.fft.rfft(env * np.hanning(n))) / n * 2
    f = np.fft.rfftfreq(n, 1.0 / FS)
    s_ord = np.interp(BASE_GRID, f / shaft_hz, amp)
    s_hz = np.interp(HZ_GRID, f, amp)
    return (s_ord / (np.median(s_ord) + 1e-30)).astype(np.float32), \
           (s_hz / (np.median(s_hz) + 1e-30)).astype(np.float32)


# ---------------- representations (methodology section 'Models, arms and controls') ----------------
def rep_m2(sn_order):
    """M2: envelope order spectrum, 0 to 25 orders (1,000 bins)."""
    return np.log1p(sn_order[..., : int(round(25.0 / ORDER_STEP))])


def rep_m3(sn_order, bpfo, bpfi, max_fault_order=6.0, bins=600):
    """M3: dual-axis spectrum. Channel 0 rescaled so BPFO harmonics fall on 1, 2, 3...;
    channel 1 so BPFI harmonics do. Shape (..., 2, bins)."""
    u = np.linspace(0.0, max_fault_order, bins, endpoint=False)
    for fo in (bpfo, bpfi):
        if u[-1] * fo >= BASE_MAX_ORDER:
            raise ValueError("fault-order range exceeds cached order range")
    x = np.log1p(sn_order)
    o = np.apply_along_axis(lambda s: np.interp(u * bpfo, BASE_GRID, s), -1, x)
    i = np.apply_along_axis(lambda s: np.interp(u * bpfi, BASE_GRID, s), -1, x)
    return np.stack([o, i], axis=-2).astype(np.float32)


def rep_m1(sn_hz):
    """M1: envelope spectrum in Hz, 0 to 1 kHz."""
    return np.log1p(sn_hz)


def ma_weight_function(u, centres, width=0.15, heights=(2.0, 1.0)):
    """Weighting function of Ma et al. (2024): 1 plus Gaussian bumps at the given fault orders
    (their Fig. 8a shows peaks of about 3 and 2 at orders 1 and 2 on a baseline of 1).
    The width is not reported in the screenshots read; 0.15 order is our assumption, logged as a deviation."""
    w = np.ones_like(u)
    for c, h in zip(centres, heights):
        w += h * np.exp(-0.5 * ((u - c) / width) ** 2)
    return w


def rep_m7(sn_order, bpfo, bpfi, max_order=8.0, bins=800):
    """M7: replication of Ma et al. (2024). Order axis normalised so BPFO sits at 1 (0 to 8);
    outer-race weights at 1 and 2, inner-race weights at the BPFI/BPFO ratio and twice it;
    the two weighted spectra are summed."""
    u = np.linspace(0.0, max_order, bins, endpoint=False)
    if u[-1] * bpfo >= BASE_MAX_ORDER:
        raise ValueError("order range exceeds cached order range")
    x = np.apply_along_axis(lambda s: np.interp(u * bpfo, BASE_GRID, s), -1, np.log1p(sn_order))
    r = bpfi / bpfo
    return (x * ma_weight_function(u, (1.0, 2.0)) + x * ma_weight_function(u, (r, 2 * r))).astype(np.float32)
