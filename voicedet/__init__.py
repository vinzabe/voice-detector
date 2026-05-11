"""voicedet: deepfake / synthetic-voice detector.

Pipeline:
    raw audio -> features (MFCC stats, spectral, prosodic) -> RandomForest
              -> per-segment probabilities + aggregate decision
              -> LLM forensic advisor

Modules:
    synth      - synthesise labelled (real-like, fake-like) training audio
                 for offline reproducible training without external corpora
    features   - extract a fixed-dim feature vector from a waveform
    detector   - DeepfakeVoiceDetector: train, save, load, predict
    advisor    - LLMVoiceForensics: turn a detection result into a structured
                 forensic interpretation
    cli        - voicedet {train, predict, advise}
"""
from .synth import synth_real_voice, synth_fake_voice, generate_dataset
from .features import (
    extract_features, FeatureVector, FRAME_FEATURE_NAMES,
    AGGREGATE_FEATURE_NAMES,
)
from .detector import DeepfakeVoiceDetector, DetectionResult
from .advisor import LLMVoiceForensics, ForensicReport

__all__ = [
    "synth_real_voice", "synth_fake_voice", "generate_dataset",
    "extract_features", "FeatureVector",
    "FRAME_FEATURE_NAMES", "AGGREGATE_FEATURE_NAMES",
    "DeepfakeVoiceDetector", "DetectionResult",
    "LLMVoiceForensics", "ForensicReport",
]
