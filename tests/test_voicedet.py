"""Test suite for voicedet."""
from __future__ import annotations
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from voicedet.synth import (  # noqa: E402
    synth_real_voice, synth_fake_voice, generate_dataset, SAMPLE_RATE,
    LabelledClip, _lowpass_brick,
)
from voicedet.features import (  # noqa: E402
    extract_features, AGGREGATE_FEATURE_NAMES, FRAME_FEATURE_NAMES,
)
from voicedet.detector import DeepfakeVoiceDetector, DetectionResult  # noqa: E402
from voicedet.advisor import LLMVoiceForensics, ForensicReport  # noqa: E402


# =================================================================== synth

def test_synth_real_shape_and_dtype():
    r = synth_real_voice(seed=1, duration_s=1.0)
    assert r.dtype == np.float32
    assert r.shape == (16000,)
    assert np.max(np.abs(r)) <= 0.91  # normalised to 0.9


def test_synth_fake_shape_and_dtype():
    f = synth_fake_voice(seed=1, duration_s=1.0)
    assert f.dtype == np.float32
    assert f.shape == (16000,)
    assert np.max(np.abs(f)) <= 0.91


def test_synth_seeds_are_deterministic():
    a = synth_real_voice(seed=42, duration_s=0.5)
    b = synth_real_voice(seed=42, duration_s=0.5)
    np.testing.assert_array_equal(a, b)
    c = synth_fake_voice(seed=42, duration_s=0.5)
    d = synth_fake_voice(seed=42, duration_s=0.5)
    np.testing.assert_array_equal(c, d)


def test_synth_real_and_fake_are_distinct():
    """Real and fake clips with the same seed should differ noticeably."""
    r = synth_real_voice(seed=7, duration_s=1.0)
    f = synth_fake_voice(seed=7, duration_s=1.0)
    diff_norm = float(np.sqrt(((r - f) ** 2).mean()))
    assert diff_norm > 0.05


def test_lowpass_brick_kills_high_freq():
    rng = np.random.default_rng(0)
    sr = 16000
    # White noise -> after low-pass at 4 kHz, energy above 4 kHz must vanish
    x = rng.normal(0, 1, size=sr).astype(np.float32)
    y = _lowpass_brick(x, sr=sr, cutoff_hz=4000.0)
    Y = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, 1 / sr)
    high_band_energy = float((Y[freqs > 4100] ** 2).sum())
    assert high_band_energy < 1e-6


def test_generate_dataset_balance_and_count():
    ds = generate_dataset(n_per_class=10, duration_s=0.5)
    assert len(ds) == 20
    labels = [c.label for c in ds]
    assert sum(1 for l in labels if l == 0) == 10
    assert sum(1 for l in labels if l == 1) == 10
    for c in ds:
        assert isinstance(c, LabelledClip)
        assert c.sr == SAMPLE_RATE
        assert c.audio.shape[0] == int(0.5 * SAMPLE_RATE)


def test_fake_has_lower_high_freq_energy_than_real():
    """The intentional 4 kHz brick wall should make fake clips' >4 kHz
    energy strictly less than real clips' on the same seed."""
    sr = 16000
    real_hf, fake_hf = [], []
    for seed in range(8):
        r = synth_real_voice(seed=seed)
        f = synth_fake_voice(seed=seed)
        for buf, bucket in [(r, real_hf), (f, fake_hf)]:
            X = np.abs(np.fft.rfft(buf))
            freqs = np.fft.rfftfreq(buf.size, 1 / sr)
            bucket.append(float((X[freqs > 4100] ** 2).sum()))
    # Brick-wall lowpass + post-lowpass noise floor: fake clips' >4 kHz
    # energy should be at least 5x lower than real clips'.
    assert np.mean(fake_hf) < np.mean(real_hf) * 0.20


# ================================================================ features

def test_aggregate_dimension_matches_name_list():
    r = synth_real_voice(seed=1, duration_s=1.0)
    fv = extract_features(r, 16000)
    assert fv.aggregate.shape == (len(AGGREGATE_FEATURE_NAMES),)
    assert fv.frame_features.shape[1] == len(FRAME_FEATURE_NAMES)
    assert fv.frame_features.shape[0] >= 50


def test_extract_features_rejects_empty():
    with pytest.raises(ValueError):
        extract_features(np.zeros(0, dtype=np.float32), 16000)


def test_extract_features_handles_2d_input():
    """Stereo input should be flattened, not crash."""
    sig = synth_real_voice(seed=2, duration_s=0.6)
    fv = extract_features(sig.reshape(1, -1), 16000)
    assert fv.aggregate.shape == (len(AGGREGATE_FEATURE_NAMES),)


def test_features_are_finite():
    for seed in range(5):
        for synth_fn in (synth_real_voice, synth_fake_voice):
            fv = extract_features(synth_fn(seed=seed, duration_s=0.7), 16000)
            assert np.isfinite(fv.aggregate).all()
            assert np.isfinite(fv.frame_features).all()


def test_high_freq_energy_ratio_distinguishes_real_vs_fake():
    """Aggregate high-frequency-energy-ratio mean must be lower for fake."""
    name = "high_freq_energy_ratio_mean"
    idx = AGGREGATE_FEATURE_NAMES.index(name)
    real_vals, fake_vals = [], []
    for seed in range(8):
        real_vals.append(float(extract_features(
            synth_real_voice(seed=seed), 16000).aggregate[idx]))
        fake_vals.append(float(extract_features(
            synth_fake_voice(seed=seed), 16000).aggregate[idx]))
    assert np.mean(fake_vals) < np.mean(real_vals)


def test_short_clip_pads_and_returns_one_frame():
    """Sub-window clip should pad and produce >=1 frame, not crash."""
    tiny = np.zeros(100, dtype=np.float32)
    tiny += 0.1
    fv = extract_features(tiny, 16000)
    assert fv.frame_features.shape[0] >= 1


# =================================================================== model

@pytest.fixture(scope="module")
def trained_detector():
    det = DeepfakeVoiceDetector(n_estimators=80, max_depth=4)
    det.fit_synthetic(n_per_class=40, duration_s=1.0)
    return det


def test_fit_requires_2d_X():
    det = DeepfakeVoiceDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros(10), np.zeros(10))


def test_fit_requires_matching_lengths():
    det = DeepfakeVoiceDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros((5, len(AGGREGATE_FEATURE_NAMES))),
                  np.zeros(3))


def test_fit_requires_correct_feature_width():
    det = DeepfakeVoiceDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros((5, 7)), np.zeros(5))


def test_predict_requires_trained():
    det = DeepfakeVoiceDetector()
    with pytest.raises(RuntimeError):
        det.predict(synth_real_voice(seed=0), sr=16000)


def test_holdout_accuracy_is_high(trained_detector):
    holdout = generate_dataset(n_per_class=15, duration_s=1.0, base_seed=999)
    X, y = DeepfakeVoiceDetector._batch_extract(holdout)
    metrics = trained_detector.evaluate(X, y)
    # Synthetic distributions are separable; require >=0.9 to catch regressions
    assert metrics["accuracy"] >= 0.90, metrics


def test_predict_returns_well_formed_result(trained_detector):
    res = trained_detector.predict(
        synth_fake_voice(seed=11, duration_s=2.0), sr=16000)
    assert isinstance(res, DetectionResult)
    assert 0.0 <= res.p_fake <= 1.0
    assert res.label in {"fake", "real"}
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.timeline) >= 1
    assert all(0.0 <= p <= 1.0 for p in res.timeline)
    assert res.duration_seconds == pytest.approx(2.0, abs=0.01)
    assert len(res.top_features) > 0
    assert all("name" in tf for tf in res.top_features)


def test_predict_real_clip_is_real(trained_detector):
    res = trained_detector.predict(
        synth_real_voice(seed=12, duration_s=1.5), sr=16000)
    assert res.label == "real"
    assert res.p_fake < 0.5


def test_predict_fake_clip_is_fake(trained_detector):
    res = trained_detector.predict(
        synth_fake_voice(seed=12, duration_s=1.5), sr=16000)
    assert res.label == "fake"
    assert res.p_fake > 0.5


def test_save_load_roundtrip(trained_detector, tmp_path):
    out = tmp_path / "model.joblib"
    trained_detector.save(out)
    assert out.exists()
    loaded = DeepfakeVoiceDetector.load(out)
    # Predictions must agree
    audio = synth_fake_voice(seed=77, duration_s=1.5)
    a = trained_detector.predict(audio, sr=16000)
    b = loaded.predict(audio, sr=16000)
    assert a.p_fake == pytest.approx(b.p_fake, abs=1e-9)
    assert a.label == b.label


def test_save_untrained_raises(tmp_path):
    det = DeepfakeVoiceDetector()
    with pytest.raises(RuntimeError):
        det.save(tmp_path / "x.joblib")


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeepfakeVoiceDetector.load(tmp_path / "does_not_exist.joblib")


def test_load_rejects_mismatched_feature_schema(trained_detector, tmp_path,
                                                       monkeypatch):
    out = tmp_path / "schema.joblib"
    trained_detector.save(out)
    # Mutate the saved schema and confirm load() refuses
    import joblib
    blob = joblib.load(out)
    blob["feature_names"] = ["wrong"] * 3
    joblib.dump(blob, out)
    with pytest.raises(ValueError):
        DeepfakeVoiceDetector.load(out)


def test_timeline_window_count_makes_sense(trained_detector):
    audio = synth_real_voice(seed=33, duration_s=3.0)
    res = trained_detector.predict(audio, sr=16000,
                                          window_s=0.5, hop_s=0.25)
    # 3 s with 0.5 s window, 0.25 s hop -> floor((3 - 0.5)/0.25) + 1 = 11
    assert len(res.timeline) == 11


# ================================================================ advisor

class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def chat(self, messages, *, model, temperature, max_tokens):
        self.calls.append({"messages": messages, "model": model,
                              "temperature": temperature,
                              "max_tokens": max_tokens})
        return types.SimpleNamespace(content=self.payload)


def _detection(p_fake=0.95):
    return DetectionResult(
        p_fake=p_fake,
        label="fake" if p_fake >= 0.5 else "real",
        confidence=abs(p_fake - 0.5) * 2,
        timeline=[0.1, 0.4, 0.85, 0.92, 0.88],
        window_seconds=0.75, hop_seconds=0.25, duration_seconds=2.5,
        top_features=[{"name": "high_freq_energy_ratio_mean",
                          "value": 0.0, "importance": 0.31}],
    )


def test_advisor_parses_pure_json():
    payload = json.dumps({
        "verdict": "likely_synthetic", "confidence": 0.82,
        "evidence": ["high p_fake", "missing high freq"],
        "recommendations": ["isolate caller", "callback verification"],
        "risk_to_organisation": "vishing / CEO fraud",
    })
    adv = LLMVoiceForensics(_FakeLLM(payload))
    rep = adv.analyse(_detection(0.92))
    assert rep.verdict == "likely_synthetic"
    assert rep.confidence == 0.82
    assert "isolate caller" in rep.recommendations
    assert isinstance(rep, ForensicReport)


def test_advisor_parses_fenced_json():
    payload = "Sure, here is the analysis:\n```json\n" + json.dumps({
        "verdict": "likely_authentic", "confidence": 0.6,
        "evidence": ["jitter present"],
        "recommendations": ["log call"],
        "risk_to_organisation": "low",
    }) + "\n```\n"
    adv = LLMVoiceForensics(_FakeLLM(payload))
    rep = adv.analyse(_detection(0.2))
    assert rep.verdict == "likely_authentic"
    assert rep.confidence == 0.6


def test_advisor_invalid_verdict_becomes_inconclusive():
    payload = json.dumps({"verdict": "definitely-real", "confidence": 0.99,
                              "evidence": [], "recommendations": [],
                              "risk_to_organisation": ""})
    rep = LLMVoiceForensics(_FakeLLM(payload)).analyse(_detection(0.5))
    assert rep.verdict == "inconclusive"


def test_advisor_clamps_confidence():
    payload = json.dumps({"verdict": "likely_synthetic", "confidence": 7.0,
                              "evidence": [], "recommendations": [],
                              "risk_to_organisation": ""})
    rep = LLMVoiceForensics(_FakeLLM(payload)).analyse(_detection(0.9))
    assert rep.confidence == 1.0
    payload2 = json.dumps({"verdict": "likely_synthetic", "confidence": -3.0,
                                 "evidence": [], "recommendations": [],
                                 "risk_to_organisation": ""})
    rep2 = LLMVoiceForensics(_FakeLLM(payload2)).analyse(_detection(0.9))
    assert rep2.confidence == 0.0


def test_advisor_handles_garbled_response():
    rep = LLMVoiceForensics(_FakeLLM("nonsense not json")).analyse(_detection(0.5))
    assert rep.verdict == "inconclusive"
    assert rep.confidence == 0.0


def test_advisor_summary_extracts_suspect_windows():
    """The detection summary passed to the LLM must flag windows >=0.7."""
    fake = _FakeLLM('{"verdict":"likely_synthetic","confidence":0.5,'
                       '"evidence":[],"recommendations":[],'
                       '"risk_to_organisation":""}')
    adv = LLMVoiceForensics(fake)
    adv.analyse(_detection(0.9))
    user_msg = fake.calls[0]["messages"][1]["content"]
    assert "suspect_windows_count" in user_msg
    # Of timeline [0.1, 0.4, 0.85, 0.92, 0.88], 3 are >= 0.7
    assert '"suspect_windows_count": 3' in user_msg


def test_advisor_passes_metadata_to_llm():
    fake = _FakeLLM('{"verdict":"inconclusive","confidence":0.5,'
                       '"evidence":[],"recommendations":[],'
                       '"risk_to_organisation":""}')
    adv = LLMVoiceForensics(fake)
    adv.analyse(_detection(0.5),
                  metadata={"caller": "ceo@corp.example",
                              "scenario": "wire-transfer-request"})
    user_msg = fake.calls[0]["messages"][1]["content"]
    assert "ceo@corp.example" in user_msg
    assert "wire-transfer-request" in user_msg


# ============================================================ live LLM smoke

@pytest.mark.skipif(not os.environ.get("LLM_LIVE"),
                          reason="set LLM_LIVE=1 to run")
def test_llm_live_advisor_smoke():
    """End-to-end: train, predict on a fake clip, ask real LLM for forensics."""
    from llm_client import LLMClient
    det = DeepfakeVoiceDetector(n_estimators=60, max_depth=4)
    det.fit_synthetic(n_per_class=30, duration_s=1.0)
    audio = synth_fake_voice(seed=2025, duration_s=2.0)
    result = det.predict(audio, sr=16000)
    assert result.label == "fake"
    advisor = LLMVoiceForensics(LLMClient(timeout=180), model="glm-5.1")
    report = advisor.analyse(result, metadata={
        "caller_claim": "CFO Karen Smith",
        "channel": "inbound_pstn",
        "scenario": "urgent_wire_transfer_request",
    })
    assert report.verdict in advisor.VALID_VERDICTS
    assert 0.0 <= report.confidence <= 1.0
    # The LLM should at minimum flag *something* given p_fake ~0.99
    assert (report.verdict == "likely_synthetic"
              or len(report.evidence) > 0)
    print("\nLLM verdict:", report.verdict,
            "confidence:", report.confidence,
            "rec_count:", len(report.recommendations))
