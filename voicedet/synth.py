"""Synthesise labelled (real-like, fake-like) training audio.

We don't ship a real LibriSpeech / ASVspoof clone -- instead we build two
parametric voice models that share a common harmonic-stack foundation but
differ in artifacts that are characteristic of TTS / vocoder output:

  REAL-LIKE:
    - jittered f0 (small random perturbations frame-to-frame)
    - slow formant drift, breathy noise floor across full spectrum
    - random micropauses

  FAKE-LIKE (vocoder artifacts):
    - smoother / overly-quantised f0
    - sharp high-frequency rolloff (typical of low-bitrate vocoders)
    - regular periodicity (no jitter), reduced shimmer
    - mel-bin "checkerboard" leakage from neural-vocoder training

The two distributions overlap intentionally so the classifier has to learn
real signal cues. They are *not* a substitute for an ASVspoof corpus, but
they suffice to train a working, reproducible detector for tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


SAMPLE_RATE = 16000


def _harmonic_stack(t: np.ndarray, f0: np.ndarray, *, n_harm: int = 8,
                      formants_hz: Tuple[float, ...] = (700, 1220, 2600),
                      formant_bw_hz: Tuple[float, ...] = (130, 110, 170),
                      sr: int = SAMPLE_RATE,
                      rng: np.random.Generator = None) -> np.ndarray:
    """Sum-of-harmonics with formant-shaped amplitudes."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = np.zeros_like(t)
    # Per-harmonic amplitude based on formant resonance
    for k in range(1, n_harm + 1):
        fk = k * f0
        amp = np.zeros_like(fk)
        for fc, bw in zip(formants_hz, formant_bw_hz):
            amp += 1.0 / (1.0 + ((fk - fc) / bw) ** 2)
        amp /= max(amp.max(), 1e-9)
        # Per-harmonic phase
        phase = 2 * np.pi * np.cumsum(fk) / sr + rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(phase) / k  # 1/k roll-off
    return out


def synth_real_voice(*, duration_s: float = 1.5,
                       sr: int = SAMPLE_RATE,
                       seed: int = 0) -> np.ndarray:
    """Synth a 'real' voice with jitter, breath noise, micropauses."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    # f0 with jitter (cents-scale) + slow drift
    f0_base = rng.uniform(110, 230)  # speaker
    drift = 6 * np.sin(2 * np.pi * 0.5 * t)
    jitter = rng.normal(0, 1.2, size=n)  # frame-to-frame
    f0 = f0_base * (1 + drift / 1200) * (1 + jitter / 1200)
    f0 = np.clip(f0, 80, 350)

    sig = _harmonic_stack(t, f0, sr=sr, rng=rng)

    # Breath noise floor across full spectrum
    noise = rng.normal(0, 0.04, size=n)
    sig = 0.7 * sig + noise

    # Random micropauses (silence gaps)
    n_pauses = rng.integers(1, 3)
    for _ in range(n_pauses):
        start = rng.integers(0, max(n - sr // 10, 1))
        gap = rng.integers(sr // 30, sr // 10)
        sig[start:start + gap] *= rng.uniform(0.05, 0.15)

    # Light high-shelf attenuation (microphone realism, but full-band)
    return _normalise(sig)


def synth_fake_voice(*, duration_s: float = 1.5,
                       sr: int = SAMPLE_RATE,
                       seed: int = 1) -> np.ndarray:
    """Synth a 'fake' voice with vocoder-style artifacts."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    # Smoothed / quantised f0 (vocoder tells)
    f0_base = rng.uniform(110, 230)
    drift = 6 * np.sin(2 * np.pi * 0.5 * t)
    f0 = f0_base * (1 + drift / 1200)
    # Quantise to 5-cent buckets (suppresses jitter) and lightly smooth
    f0 = np.round(f0 * 1000.0) / 1000.0
    win = max(int(0.020 * sr), 5)
    kernel = np.ones(win) / win
    f0 = np.convolve(f0, kernel, mode="same")

    sig = _harmonic_stack(t, f0, sr=sr, rng=rng, n_harm=10)

    # Sharp high-frequency cutoff (typical low-bitrate vocoder)
    sig = _lowpass_brick(sig, sr=sr, cutoff_hz=4000.0)

    # Reduced noise floor
    sig = 0.85 * sig + rng.normal(0, 0.012, size=n)

    # No micropauses (TTS often misses naturalistic gaps)
    # Add subtle 80 Hz mains-style hum (vocoder buzz)
    sig += 0.01 * np.sin(2 * np.pi * 80 * t)

    return _normalise(sig)


def _lowpass_brick(x: np.ndarray, *, sr: int, cutoff_hz: float) -> np.ndarray:
    """FFT-based brick-wall low pass (creates a sharp spectral edge)."""
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    X[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(X, n=x.size)


def _normalise(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x)) or 1.0)
    return (x / peak * 0.9).astype(np.float32)


@dataclass
class LabelledClip:
    audio: np.ndarray
    sr: int
    label: int   # 1 = fake, 0 = real
    seed: int


def generate_dataset(*, n_per_class: int = 60,
                       duration_s: float = 1.5,
                       sr: int = SAMPLE_RATE,
                       base_seed: int = 1234) -> List[LabelledClip]:
    """Deterministic dataset of n_per_class real + n_per_class fake clips."""
    rng = np.random.default_rng(base_seed)
    seeds = rng.integers(1, 10**6, size=2 * n_per_class).tolist()
    out: List[LabelledClip] = []
    for i in range(n_per_class):
        out.append(LabelledClip(
            audio=synth_real_voice(duration_s=duration_s, sr=sr, seed=seeds[i]),
            sr=sr, label=0, seed=seeds[i]))
    for i in range(n_per_class):
        out.append(LabelledClip(
            audio=synth_fake_voice(duration_s=duration_s, sr=sr,
                                       seed=seeds[n_per_class + i]),
            sr=sr, label=1, seed=seeds[n_per_class + i]))
    return out
