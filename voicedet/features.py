"""Feature extraction.

We extract two views from each clip:
    * frame-level features (per ~25 ms window, 10 ms hop)
    * aggregate features (mean / std / skewness over frames + global stats)

The aggregate vector is what the classifier sees. The frame-level matrix
is exposed so the detector can also produce a per-segment timeline of
suspicion scores.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


N_MFCC = 13
N_FFT = 512
HOP = 160                          # 10 ms at 16 kHz
WIN = 400                          # 25 ms at 16 kHz

FRAME_FEATURE_NAMES: List[str] = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + ["spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
        "zero_crossing_rate", "rms"]
    + ["spectral_flatness", "high_freq_energy_ratio"]
)

AGGREGATE_FEATURE_NAMES: List[str] = []
for s in ("mean", "std"):
    for n in FRAME_FEATURE_NAMES:
        AGGREGATE_FEATURE_NAMES.append(f"{n}_{s}")
AGGREGATE_FEATURE_NAMES.extend([
    "f0_mean", "f0_std", "f0_jitter",
    "spectral_edge_4khz_drop", "spectral_dynamic_range",
])


@dataclass
class FeatureVector:
    aggregate: np.ndarray         # shape (D,)
    frame_features: np.ndarray    # shape (T, F)
    sr: int


# ---------------------------------------------------------------------------

def _safe_load_librosa():
    import librosa
    return librosa


def _frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Stack overlapping frames into shape (T, win)."""
    n_frames = max(1 + (x.size - win) // hop, 1)
    if x.size < win:
        x = np.pad(x, (0, win - x.size), mode="constant")
        n_frames = 1
    out = np.empty((n_frames, win), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        out[i] = x[start:start + win]
    return out


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    sign = np.sign(frames)
    sign[sign == 0] = 1
    zcr = np.abs(np.diff(sign, axis=1)).sum(axis=1) / (2.0 * frames.shape[1])
    return zcr


def _rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def _f0_from_autocorr(frames: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame f0 estimate via autocorrelation peak in [80, 350] Hz."""
    out = np.zeros(frames.shape[0], dtype=np.float32)
    min_lag = int(sr / 350)
    max_lag = int(sr / 80)
    for i in range(frames.shape[0]):
        f = frames[i] - frames[i].mean()
        if np.max(np.abs(f)) < 1e-3:
            out[i] = 0.0
            continue
        ac = np.correlate(f, f, mode="full")[f.size - 1:]
        if ac[0] <= 0:
            out[i] = 0.0
            continue
        ac /= ac[0]
        seg = ac[min_lag:max_lag + 1]
        if seg.size == 0:
            out[i] = 0.0
            continue
        peak = int(np.argmax(seg)) + min_lag
        if ac[peak] > 0.3:
            out[i] = sr / peak
        else:
            out[i] = 0.0
    return out


def extract_features(audio: np.ndarray, sr: int) -> FeatureVector:
    """Extract aggregate + frame-level features from a 1-D waveform."""
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if audio.size == 0:
        raise ValueError("audio must be non-empty")
    audio = audio.astype(np.float32)
    librosa = _safe_load_librosa()

    # Frame-level via librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP).T
    spec_centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP, roll_percent=0.85)[0]
    spec_bw = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP)[0]
    spec_flat = librosa.feature.spectral_flatness(
        y=audio, n_fft=N_FFT, hop_length=HOP)[0]

    # Manual ZCR / RMS / high-freq energy ratio (per-frame)
    frames = _frame_signal(audio, WIN, HOP)
    n_frames = min(mfcc.shape[0], spec_centroid.size, frames.shape[0])
    zcr = _zero_crossing_rate(frames[:n_frames])
    rms = _rms(frames[:n_frames])
    # High-frequency energy ratio: STFT-based
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP)) ** 2
    freqs = np.linspace(0, sr / 2, stft.shape[0])
    hf_mask = freqs >= 4000
    total_e = stft.sum(axis=0) + 1e-12
    hf_ratio = stft[hf_mask].sum(axis=0) / total_e
    hf_ratio = hf_ratio[:n_frames]

    # Truncate to common length
    mfcc = mfcc[:n_frames]
    spec_centroid = spec_centroid[:n_frames]
    spec_rolloff = spec_rolloff[:n_frames]
    spec_bw = spec_bw[:n_frames]
    spec_flat = spec_flat[:n_frames]

    frame_feats = np.concatenate([
        mfcc,
        spec_centroid[:, None], spec_rolloff[:, None], spec_bw[:, None],
        zcr[:, None], rms[:, None],
        spec_flat[:, None], hf_ratio[:, None],
    ], axis=1).astype(np.float32)

    # Aggregate: mean + std of each frame feature
    agg_mean = frame_feats.mean(axis=0)
    agg_std = frame_feats.std(axis=0) + 1e-12

    # Pitch summary
    f0 = _f0_from_autocorr(frames[:n_frames], sr)
    voiced = f0[f0 > 0]
    if voiced.size > 1:
        f0_mean = float(voiced.mean())
        f0_std = float(voiced.std())
        # Jitter: mean abs frame-to-frame f0 difference / mean f0
        diffs = np.abs(np.diff(voiced))
        f0_jitter = float(diffs.mean() / max(f0_mean, 1.0))
    else:
        f0_mean = f0_std = f0_jitter = 0.0

    # Vocoder-style spectral-edge cue: fraction of energy lost just above 4 kHz
    band_lo_mask = (freqs >= 3500) & (freqs < 4000)
    band_hi_mask = (freqs >= 4000) & (freqs < 4500)
    e_lo = stft[band_lo_mask].sum() + 1e-12
    e_hi = stft[band_hi_mask].sum() + 1e-12
    spectral_edge_drop = float(1.0 - (e_hi / e_lo))
    # Dynamic range of total energy across frames (TTS often shows compressed range)
    spectral_dynamic_range = float(np.log10(total_e.max() + 1e-12)
                                       - np.log10(total_e.min() + 1e-12))

    aggregate = np.concatenate([
        agg_mean, agg_std,
        np.array([f0_mean, f0_std, f0_jitter,
                    spectral_edge_drop, spectral_dynamic_range],
                  dtype=np.float32),
    ]).astype(np.float32)

    return FeatureVector(aggregate=aggregate, frame_features=frame_feats, sr=sr)
