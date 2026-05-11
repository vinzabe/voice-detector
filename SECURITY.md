# Security Policy

## Reporting

Report vulnerabilities responsibly to the repository owner by email -- do not open public issues.

## Defensive use only

`voicedet` is a **defensive synthetic-voice detector** for SOC, fraud,
and trust-and-safety teams. It does not generate synthetic voices for
the purpose of impersonating real people. The bundled synthesiser
(`voicedet.synth`) produces clearly-synthetic parametric audio used
*only* for training the detector and for reproducible tests.

## Limits of the detector

- Models trained only on the bundled synthetic data **will not
  generalise** to real-world TTS / vocoder output. Re-train on a real
  corpus (e.g. ASVspoof2019, WaveFake, In-the-Wild Audio Deepfakes)
  before relying on the score for any consequential decision.
- A "likely_authentic" verdict is not proof of authenticity. Pair this
  tool with out-of-band callback verification, voiceprint enrollment,
  or content-of-call sanity checks for high-value workflows (wire
  transfers, credential resets, access changes).
- The LLM advisor is a contextualiser, not a classifier. Its `verdict`
  is bounded by the model's quantitative score; its `recommendations`
  are advisory.

## Data handling

- Audio is processed locally; no audio bytes are sent to the LLM.
  The advisor receives only the detection summary
  (`p_fake`, top features, suspect-window timestamps) and any
  metadata the caller chooses to include.
- The CLI does not log audio paths or model outputs anywhere beyond
  stdout. Wrap it in your own pipeline if you need an audit trail.

## Threat model

- Assumed operator: an analyst or automated SOC pipeline.
- Out of scope: an adversary with white-box access to the trained
  model performing gradient-based adversarial audio attacks. The
  handcrafted-feature design provides *some* robustness against
  pixel-domain audio adversaries, but cannot defend against an
  attacker who tunes their TTS pipeline against this specific model.
