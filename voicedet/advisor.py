"""LLM forensic advisor.

Given a `DetectionResult` and optional metadata (file name, claimed
speaker, scenario), produce a structured forensic interpretation:

    {
      "verdict": "likely_synthetic" | "likely_authentic" | "inconclusive",
      "confidence": 0..1,
      "evidence": [str, ...],
      "recommendations": [str, ...],
      "risk_to_organisation": str,
    }

The advisor never executes the model itself; it only frames its output.
The result is *advisory*. The detector's quantitative output remains the
authoritative score.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import re
from typing import Any, Dict, List, Optional

from .detector import DetectionResult


SYSTEM_PROMPT = """You are a forensic audio analyst assisting a SOC.
You receive the output of a synthetic-voice (deepfake) detection model
together with claimed metadata about the recording. You must:

  1. Combine the model's quantitative score with the human-context cues
     (claimed speaker, scenario, channel) to produce a defensive verdict.
  2. NEVER claim certainty that the model itself does not warrant.
  3. Assume an adversarial threat model: the recording may be a
     vishing / CEO-fraud / credential-extraction attempt.
  4. Output ONLY a JSON object matching the schema described.

Schema (no extra keys, no comments):
{
  "verdict": "likely_synthetic" | "likely_authentic" | "inconclusive",
  "confidence": float in [0,1],
  "evidence": [string, ...],
  "recommendations": [string, ...],
  "risk_to_organisation": string
}
"""


@dataclass
class ForensicReport:
    verdict: str
    confidence: float
    evidence: List[str]
    recommendations: List[str]
    risk_to_organisation: str
    raw_llm_output: str = ""
    detection_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class LLMVoiceForensics:
    VALID_VERDICTS = {"likely_synthetic", "likely_authentic", "inconclusive"}

    def __init__(self, llm_client: Any, *, model: str = "glm-5.1",
                  temperature: float = 0.2, max_tokens: int = 800):
        self.client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def analyse(self, detection: DetectionResult,
                  *, metadata: Optional[Dict[str, Any]] = None) -> ForensicReport:
        metadata = metadata or {}
        summary = self._summarise_detection(detection)
        user_prompt = self._build_user_prompt(summary, metadata)
        resp = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = (getattr(resp, "content", None) or str(resp)).strip()
        parsed = self._parse_json(raw)
        return self._coerce_report(parsed, raw=raw,
                                       detection_summary=summary)

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _summarise_detection(d: DetectionResult) -> Dict[str, Any]:
        timeline = d.timeline or []
        peak = max(timeline) if timeline else d.p_fake
        # Find suspicious window time-range (above 0.7)
        suspect_windows = []
        for i, p in enumerate(timeline):
            if p >= 0.7:
                start_s = i * d.hop_seconds
                end_s = start_s + d.window_seconds
                suspect_windows.append({"start_s": round(start_s, 3),
                                            "end_s": round(end_s, 3),
                                            "p_fake": round(p, 3)})
        return {
            "label": d.label,
            "p_fake": round(d.p_fake, 4),
            "confidence": round(d.confidence, 4),
            "duration_s": round(d.duration_seconds, 3),
            "peak_p_fake": round(peak, 4),
            "n_windows": len(timeline),
            "suspect_windows_count": len(suspect_windows),
            "suspect_windows_preview": suspect_windows[:5],
            "top_features": d.top_features[:5],
        }

    def _build_user_prompt(self, summary: Dict[str, Any],
                              metadata: Dict[str, Any]) -> str:
        return (
            "Detector output (JSON):\n"
            + json.dumps(summary, indent=2)
            + "\n\nClaimed metadata (JSON):\n"
            + json.dumps(metadata, indent=2)
            + "\n\nReturn ONLY a JSON object matching the schema. "
              "Be specific in 'evidence' (cite p_fake, top_features, "
              "suspect_windows). 'recommendations' must be concrete defensive "
              "actions the SOC can take in the next 24 hours."
        )

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        # Try plain parse, then ```json fence, then first {...} block
        s = raw.strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {}

    def _coerce_report(self, parsed: Dict[str, Any], *, raw: str,
                          detection_summary: Dict[str, Any]) -> ForensicReport:
        verdict = str(parsed.get("verdict", "inconclusive")).lower().strip()
        if verdict not in self.VALID_VERDICTS:
            verdict = "inconclusive"
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        evidence = parsed.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        evidence = [str(x) for x in evidence][:20]
        recs = parsed.get("recommendations", [])
        if not isinstance(recs, list):
            recs = [str(recs)]
        recs = [str(x) for x in recs][:20]
        risk = str(parsed.get("risk_to_organisation", ""))[:1000]
        return ForensicReport(
            verdict=verdict, confidence=confidence,
            evidence=evidence, recommendations=recs,
            risk_to_organisation=risk,
            raw_llm_output=raw,
            detection_summary=detection_summary,
        )
