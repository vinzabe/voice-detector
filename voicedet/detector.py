"""DeepfakeVoiceDetector: train, save, load, predict.

Sklearn `GradientBoostingClassifier` over the aggregate feature vector,
plus a per-frame-window timeline of suspicion scores so the caller can
visualise where the suspect content lies. The frame-level timeline is
produced by re-extracting aggregate features from sliding windows
(default 750 ms with 250 ms hop) and re-scoring each window.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import extract_features, AGGREGATE_FEATURE_NAMES
from .synth import generate_dataset, LabelledClip, SAMPLE_RATE


@dataclass
class DetectionResult:
    p_fake: float                    # aggregate probability (0..1) of FAKE
    label: str                       # "fake" or "real"
    confidence: float                # |p - 0.5| * 2
    timeline: List[float]            # per-window p_fake
    window_seconds: float
    hop_seconds: float
    duration_seconds: float
    top_features: List[dict] = field(default_factory=list)
    sr: int = SAMPLE_RATE

    def to_dict(self) -> dict:
        return asdict(self)


class DeepfakeVoiceDetector:
    """RandomForest/GBM over a fixed-dim aggregate feature vector."""

    DEFAULT_WINDOW_S = 0.75
    DEFAULT_HOP_S = 0.25

    def __init__(self, *, n_estimators: int = 120, max_depth: int = 4,
                  random_state: int = 0):
        self._pipeline: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )),
        ])
        self._trained = False
        self._train_meta: dict = {}

    # ------------------------------------------------------------------ train

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepfakeVoiceDetector":
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y first dim mismatch")
        if X.shape[1] != len(AGGREGATE_FEATURE_NAMES):
            raise ValueError(
                f"feature width {X.shape[1]} != "
                f"expected {len(AGGREGATE_FEATURE_NAMES)}")
        self._pipeline.fit(X, y)
        self._trained = True
        self._train_meta = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "class_balance": {int(c): int((y == c).sum())
                                for c in np.unique(y)},
        }
        return self

    def fit_synthetic(self, *, n_per_class: int = 80,
                        duration_s: float = 1.5) -> "DeepfakeVoiceDetector":
        """Convenience: generate a synthetic dataset and train on it."""
        clips = generate_dataset(n_per_class=n_per_class,
                                     duration_s=duration_s)
        X, y = self._batch_extract(clips)
        return self.fit(X, y)

    @staticmethod
    def _batch_extract(clips: Sequence[LabelledClip]):
        feats, labels = [], []
        for c in clips:
            f = extract_features(c.audio, c.sr)
            feats.append(f.aggregate)
            labels.append(c.label)
        return np.vstack(feats), np.asarray(labels, dtype=np.int32)

    # ----------------------------------------------------------------- score

    def predict(self, audio: np.ndarray, sr: int = SAMPLE_RATE,
                  *, window_s: Optional[float] = None,
                  hop_s: Optional[float] = None,
                  top_k_features: int = 6) -> DetectionResult:
        if not self._trained:
            raise RuntimeError("detector not trained")
        feats = extract_features(audio, sr)
        proba = float(self._pipeline.predict_proba(feats.aggregate[None])[0, 1])
        label = "fake" if proba >= 0.5 else "real"
        confidence = float(abs(proba - 0.5) * 2)

        win_s = window_s or self.DEFAULT_WINDOW_S
        hop = hop_s or self.DEFAULT_HOP_S
        timeline = self._timeline(audio, sr, win_s, hop)

        top_feats = self._top_feature_contributions(feats.aggregate,
                                                          top_k=top_k_features)
        return DetectionResult(
            p_fake=proba, label=label, confidence=confidence,
            timeline=timeline,
            window_seconds=win_s, hop_seconds=hop,
            duration_seconds=float(audio.size / sr),
            top_features=top_feats, sr=sr,
        )

    def _timeline(self, audio: np.ndarray, sr: int,
                    window_s: float, hop_s: float) -> List[float]:
        win = int(window_s * sr)
        hop = int(hop_s * sr)
        if audio.size <= win:
            f = extract_features(audio, sr)
            return [float(self._pipeline.predict_proba(f.aggregate[None])[0, 1])]
        out: List[float] = []
        for start in range(0, audio.size - win + 1, hop):
            seg = audio[start:start + win]
            try:
                f = extract_features(seg, sr)
                p = float(self._pipeline.predict_proba(f.aggregate[None])[0, 1])
            except Exception:
                p = 0.5
            out.append(p)
        return out

    def _top_feature_contributions(self, feature_vec: np.ndarray,
                                          top_k: int = 6) -> List[dict]:
        """Return top-k features by GBM impurity importance, with values."""
        clf = self._pipeline.named_steps["clf"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return []
        idx = np.argsort(importances)[::-1][:top_k]
        return [
            {"name": AGGREGATE_FEATURE_NAMES[int(i)],
              "value": float(feature_vec[int(i)]),
              "importance": float(importances[int(i)])}
            for i in idx
        ]

    # ------------------------------------------------------------- evaluate

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        if not self._trained:
            raise RuntimeError("detector not trained")
        proba = self._pipeline.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        n = int(y.shape[0])
        acc = (tp + tn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
        }

    # -------------------------------------------------------------- persist

    def save(self, path: str | Path) -> None:
        if not self._trained:
            raise RuntimeError("cannot save untrained detector")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self._pipeline,
            "feature_names": AGGREGATE_FEATURE_NAMES,
            "train_meta": self._train_meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "DeepfakeVoiceDetector":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        blob = joblib.load(path)
        if blob.get("feature_names") != AGGREGATE_FEATURE_NAMES:
            raise ValueError(
                "feature schema mismatch: model was trained against a "
                "different feature set")
        det = cls()
        det._pipeline = blob["pipeline"]
        det._trained = True
        det._train_meta = blob.get("train_meta", {})
        return det
