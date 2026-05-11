# voicedet

Synthetic-voice / deepfake-audio detector for SOC and fraud teams.

Pipeline:

```
raw audio  ->  handcrafted features (MFCC + spectral + prosodic)
           ->  GradientBoosting classifier
           ->  per-window timeline + aggregate verdict
           ->  LLM forensic advisor  (verdict + recommendations)
```

## Why hand-crafted features

Modern TTS / vocoder pipelines leave a small set of forensic
artifacts that handcrafted features capture cheaply and explainably:

- **High-frequency energy ratio**: most low-bitrate vocoders
  introduce a sharp cutoff around 4 kHz.
- **Pitch jitter** (`f0_jitter`): authentic speech varies frame-to-frame
  by tens of cents; vocoder f0 is smoother / over-quantised.
- **Spectral bandwidth + flatness**: TTS audio shows a narrower
  bandwidth and lower flatness because of harmonic-stack synthesis.
- **Spectral dynamic range**: TTS engines compress the per-frame energy
  range (no breaths, no micro-pauses).

These are encoded in `voicedet/features.py` as a 45-dim aggregate
vector (mean+std of 20 frame features + 5 global summaries).

## Sample data

We don't ship a real ASVspoof / LibriSpeech sample. Instead
`voicedet.synth` builds a **parametric synthetic dataset** with two
labelled distributions (real-like / fake-like) that share a common
formant / harmonic-stack core but differ in the artifacts above. This
is enough to train a working classifier reproducibly, but should be
swapped out for ASVspoof2019 / WaveFake / In-the-Wild Audio Deepfakes
in production.

## Quick start

```bash
pip install -r requirements.txt

# Train on synthetic data and evaluate on a held-out synthetic set
python -m voicedet.cli train --model-out models/voicedet.joblib

# Score a wav file
python -m voicedet.cli predict --model models/voicedet.joblib --audio call.wav

# Full pipeline incl. LLM forensic advisor
python -m voicedet.cli advise --model models/voicedet.joblib \
    --audio call.wav \
    --metadata '{"caller":"cfo@corp.example","scenario":"wire_transfer"}'

# Emit a sanity-check pair of wavs (real-like + fake-like)
python -m voicedet.cli demo --out-dir data/demo
```

## Output

`predict` returns:

```json
{
  "p_fake": 0.97,
  "label": "fake",
  "confidence": 0.94,
  "timeline": [0.12, 0.41, 0.85, 0.91, 0.88, 0.93],
  "window_seconds": 0.75,
  "hop_seconds": 0.25,
  "duration_seconds": 2.5,
  "top_features": [
    {"name": "high_freq_energy_ratio_mean", "value": 0.0001, "importance": 0.21},
    ...
  ],
  "sr": 16000
}
```

`advise` adds a structured LLM verdict:

```json
{
  "verdict": "likely_synthetic",
  "confidence": 0.95,
  "evidence": [
    "p_fake=0.97 with confidence 0.94",
    "high_freq_energy_ratio_mean is near zero, consistent with vocoder cutoff",
    "all 6 windows above 0.5"
  ],
  "recommendations": [
    "Do NOT process the wire transfer based on this call alone",
    "Initiate callback to a known-good number for the claimed CFO",
    "Open ticket and preserve the original audio for chain-of-custody",
    ...
  ],
  "risk_to_organisation": "..."
}
```

## Layout

```
voicedet/
  synth.py        parametric real/fake audio synthesiser
  features.py     45-dim aggregate + frame-level extractor
  detector.py     DeepfakeVoiceDetector (sklearn pipeline + persistence)
  advisor.py     LLMVoiceForensics + ForensicReport
  cli.py          voicedet {train, predict, advise, demo}
tests/test_voicedet.py    33 unit + 1 live LLM smoke
```

## Limitations & honesty

- The synthetic dataset is for reproducibility, not for benchmarking.
  A model trained only on `voicedet.synth` will *not* generalise to
  real-world TTS engines. Re-train on a real corpus before deploying.
- The LLM advisor never overrides the model's quantitative score; it
  contextualises it. Treat its recommendations as a starting point for
  analyst review.
- No model is perfect at this task. Always pair detection with
  out-of-band callback verification for high-value calls.

## Tests

```bash
pytest tests/ -v
LLM_LIVE=1 pytest tests/ -v
```

## License

MIT
