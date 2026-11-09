"""Signal-processing building blocks for VVC (numpy/scipy only)."""
from __future__ import annotations

import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly, medfilt


# ---------------------------------------------------------------- audio I/O
def load_wav(path: str, sr: int) -> np.ndarray:
    in_sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / np.iinfo(x.dtype).max
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128) / 128
    else:
        x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if in_sr != sr:
        g = gcd(in_sr, sr)
        x = resample_poly(x, sr // g, in_sr // g)
    return x


def save_wav(path: str, x: np.ndarray, sr: int) -> None:
    peak = np.max(np.abs(x)) or 1.0
    y = x / peak * 0.89
    wavfile.write(path, sr, (y * 32767).astype(np.int16))


# ---------------------------------------------------------------- STFT
def frame_count(n_samples: int, hop: int) -> int:
    return 1 + n_samples // hop


def stft(x: np.ndarray, nfft: int, hop: int, n_frames: int | None = None) -> np.ndarray:
    """Centered STFT: frame i is centred on sample i*hop. Returns (frames, bins)."""
    if n_frames is None:
        n_frames = frame_count(len(x), hop)
    pad = nfft // 2
    need = (n_frames - 1) * hop + nfft
    xp = np.zeros(need)
    seg = x[: max(0, need - pad)]
    xp[pad:pad + len(seg)] = seg
    idx = np.arange(nfft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = np.hanning(nfft + 1)[:-1]
    return np.fft.rfft(xp[idx] * win, axis=1)


def istft(X: np.ndarray, nfft: int, hop: int, length: int) -> np.ndarray:
    win = np.hanning(nfft + 1)[:-1]
    frames = np.fft.irfft(X, n=nfft, axis=1) * win
    n_frames = X.shape[0]
    total = max((n_frames - 1) * hop + nfft, length + nfft)
    y = np.zeros(total)
    wsum = np.zeros(total)
    for i in range(n_frames):
        s = i * hop
        y[s:s + nfft] += frames[i]
        wsum[s:s + nfft] += win ** 2
    y /= np.maximum(wsum, 1e-8)
    pad = nfft // 2
    return y[pad:pad + length]


# ---------------------------------------------------------------- mel bands
def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_bands: int, nfft: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Triangular filters with peak 1 (they sum to ~1 between the first and last centre)."""
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_bands + 2))
    fb = np.zeros((n_bands, len(freqs)))
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[b] = np.clip(np.minimum(up, down), 0, None)
    return fb


def band_interp_weights(fb: np.ndarray) -> np.ndarray:
    """Per-bin weights for spreading per-band values back onto FFT bins.
    Bins below the first centre / above the last centre take that band's value."""
    n_bands, n_bins = fb.shape
    centres = np.argmax(fb, axis=1)
    w = fb.copy()
    w[0, : centres[0] + 1] = 1.0
    w[-1, centres[-1]:] = 1.0
    for b in range(1, n_bands):
        w[b, : centres[0] + 1] = 0.0
    for b in range(n_bands - 1):
        w[b, centres[-1]:] = 0.0
    w /= np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
    return w


# ---------------------------------------------------------------- pitch (YIN)
def yin(x: np.ndarray, sr: int, hop: int, n_frames: int,
        fmin: float = 60.0, fmax: float = 400.0, win_s: float = 0.025,
        threshold: float = 0.15):
    """Vectorised YIN. Returns (f0_hz, aperiodicity) per frame; f0 is the raw
    estimate for every frame — voicing is decided by the caller."""
    w = int(win_s * sr)
    tau_min = int(sr / fmax)
    tau_max = int(np.ceil(sr / fmin))
    L = w + tau_max
    pad = L // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(L + n_frames * hop)])
    idx = np.arange(L)[None, :] + hop * np.arange(n_frames)[:, None]
    fr = xp[idx]

    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    A = np.fft.rfft(fr, nfft, axis=1)
    B = np.fft.rfft(fr[:, :w], nfft, axis=1)
    r = np.fft.irfft(A * np.conj(B), nfft, axis=1)[:, : tau_max + 1]

    sq = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(fr ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e0 = sq[:, w][:, None]
    et = sq[:, taus + w] - sq[:, taus]
    d = np.maximum(e0 + et - 2 * r, 0.0)

    cmnd = np.ones_like(d)
    cs = np.cumsum(d[:, 1:], axis=1)
    cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cs, 1e-12)

    f0 = np.zeros(n_frames)
    ap = np.ones(n_frames)
    seg = cmnd[:, tau_min: tau_max]
    for i in range(n_frames):
        c = seg[i]
        below = np.nonzero(c < threshold)[0]
        if len(below):
            k = below[0]
            while k + 1 < len(c) and c[k + 1] < c[k]:
                k += 1
        else:
            k = int(np.argmin(c))
        t = k + tau_min
        # parabolic interpolation around the dip
        if 1 <= t < tau_max:
            a, b, cc = cmnd[i, t - 1], cmnd[i, t], cmnd[i, t + 1]
            den = a - 2 * b + cc
            shift = 0.5 * (a - cc) / den if abs(den) > 1e-12 else 0.0
        else:
            shift = 0.0
        f0[i] = sr / (t + np.clip(shift, -1, 1))
        ap[i] = float(np.clip(cmnd[i, t], 0.0, 1.0))
    return f0, ap


def clean_voicing(f0: np.ndarray, voiced: np.ndarray, min_run: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Drop voiced islands shorter than min_run frames and median-smooth f0 inside runs."""
    voiced = voiced.copy()
    f0 = f0.copy()
    i, n = 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            if j - i < min_run:
                voiced[i:j] = False
            else:
                run = np.log2(f0[i:j])
                k = 5 if j - i >= 5 else (3 if j - i >= 3 else 1)
                f0[i:j] = 2 ** medfilt(run, k)
            i = j
        else:
            i += 1
    f0[~voiced] = 0.0
    return f0, voiced
