"""voicedet command-line tool: train | predict | advise."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from voicedet.detector import DeepfakeVoiceDetector  # noqa: E402
from voicedet.synth import generate_dataset, synth_real_voice, synth_fake_voice  # noqa: E402


def _load_audio(path: str):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _save_audio(path: str, audio, sr: int):
    import soundfile as sf
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def cmd_train(args):
    det = DeepfakeVoiceDetector(n_estimators=args.n_estimators,
                                       max_depth=args.max_depth)
    det.fit_synthetic(n_per_class=args.n_per_class,
                          duration_s=args.duration_s)
    det.save(args.model_out)
    print(f"saved trained detector to {args.model_out}")
    # Quick eval on holdout
    holdout = generate_dataset(n_per_class=20, duration_s=args.duration_s,
                                   base_seed=99)
    X, y = DeepfakeVoiceDetector._batch_extract(holdout)
    metrics = det.evaluate(X, y)
    print("holdout metrics:", json.dumps(metrics, indent=2))


def cmd_predict(args):
    det = DeepfakeVoiceDetector.load(args.model)
    audio, sr = _load_audio(args.audio)
    result = det.predict(audio, sr=sr)
    print(json.dumps(result.to_dict(), indent=2))


def cmd_advise(args):
    from llm_client import LLMClient
    from voicedet.advisor import LLMVoiceForensics
    det = DeepfakeVoiceDetector.load(args.model)
    audio, sr = _load_audio(args.audio)
    result = det.predict(audio, sr=sr)
    client = LLMClient()
    advisor = LLMVoiceForensics(client, model=args.llm_model)
    metadata = json.loads(args.metadata) if args.metadata else {}
    report = advisor.analyse(result, metadata=metadata)
    print(json.dumps(report.to_dict(), indent=2))


def cmd_demo(args):
    """Generate one real + one fake clip and write them to disk."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    real = synth_real_voice(seed=42)
    fake = synth_fake_voice(seed=42)
    _save_audio(out_dir / "real.wav", real, 16000)
    _save_audio(out_dir / "fake.wav", fake, 16000)
    print(f"wrote {out_dir/'real.wav'} and {out_dir/'fake.wav'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="voicedet")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="train detector on synthetic data")
    pt.add_argument("--model-out", default="models/voicedet.joblib")
    pt.add_argument("--n-per-class", type=int, default=80)
    pt.add_argument("--duration-s", type=float, default=1.5)
    pt.add_argument("--n-estimators", type=int, default=120)
    pt.add_argument("--max-depth", type=int, default=4)
    pt.set_defaults(func=cmd_train)

    pp = sub.add_parser("predict", help="score a wav file")
    pp.add_argument("--model", required=True)
    pp.add_argument("--audio", required=True)
    pp.set_defaults(func=cmd_predict)

    pa = sub.add_parser("advise", help="LLM forensic analysis of a wav")
    pa.add_argument("--model", required=True)
    pa.add_argument("--audio", required=True)
    pa.add_argument("--metadata", default="",
                       help="JSON object describing the recording")
    pa.add_argument("--llm-model", default="glm-5.1")
    pa.set_defaults(func=cmd_advise)

    pd = sub.add_parser("demo", help="emit synthetic real + fake wavs")
    pd.add_argument("--out-dir", default="data/demo")
    pd.set_defaults(func=cmd_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
