# Training and evaluation data

Supply your own permitted, aligned pairs. The repository does not download training datasets, restricted checkpoints, or teacher outputs. Record materials in [the provenance ledger](../docs/provenance.md) before use.

## Manifest format

Pass manifests through Hydra when training:

```bash
python -m realtime_use.train experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  trainer.accelerator=gpu trainer.devices=1
```

These paths select user-provided files; they do not trigger downloads. Absolute manifest paths avoid ambiguity when running sweeps. Audio paths within each manifest are still resolved relative to that manifest.

Use one JSON object per line in a UTF-8 JSONL file:

```json
{"noisy":"noisy/a.wav","clean":"clean/a.wav","sample_rate":16000}
{"noisy":"noisy/b.wav","clean":"clean/b.wav","sample_rate":16000}
```

Paths are relative to the manifest location unless absolute. The initial implementation requires:

- Mono noisy and clean recordings.
- A real and declared sample rate of 16,000 Hz for the supplied initial configurations.
- Equal sample counts and explicit time alignment for each pair.
- Audio the loader can read through its supported audio library.

The loader rejects mismatched rates and lengths. It does not resample or estimate alignment silently. Training crops noisy, clean, and any optional teacher audio at exactly the same positions and pads them consistently.

For optional cached-output distillation, a row may add a teacher path:

```json
{"noisy":"noisy/a.wav","clean":"clean/a.wav","teacher":"teacher/a.wav","sample_rate":16000}
```

Teacher audio must have the same rate, channels, length, and alignment. Providing the path does not enable regression-stage distillation by itself: its configured loss weight must also be enabled. Distillation is initially disabled; adversarial fine-tuning ignores teacher tensors.

## Targets and synthetic degradations

For simulated reverberation, the referenced offline paper motivates an anechoic target shifted to align with the direct path of the simulated room response. Prepare and inspect that alignment during dataset construction. The loader's length check cannot establish perceptual or physical alignment.

The paired-data loader offers optional training-only corruptions through [audiomentations](https://github.com/iver56/audiomentations): Gaussian noise with sampled SNR, clipping, and low-pass filtering. Install and enable them with:

```bash
python -m pip install -e '.[augmentation]'
python -m realtime_use.train experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  data.augmentation.enabled=true \
  trainer.accelerator=gpu trainer.devices=1
```

Transform probabilities and ranges are recorded under `data.augmentation` in the resolved configuration. Corruption changes only the noisy training input, preserves sample count, and leaves clean targets and validation audio unchanged. Gaussian SNR is relative to the supplied input waveform. This can further degrade already-noisy input; it is not a calibrated clean-speech/noise-corpus mixer. Python/NumPy worker randomness follows Lightning's worker seeding; exact mid-epoch replay is not promised.

Enabled augmentation is incompatible with cached-output distillation. Cached teacher outputs describe a specific input; additional random corruption would invalidate that correspondence. The loader rejects this combination instead of applying unrelated transforms to teacher and clean audio.

These three library transforms are not a reproduction of the full URGENT corruption generator. RIR synthesis and direct-path target shifts, codecs, packet loss, wind-noise simulation, and curated corpus mixing remain unimplemented. Do not use an arbitrary library reverb transform without defining target delay and tail handling.

Clean targets need listening and signal checks. Residual noise in a target can teach the model to preserve it. Keep clean speech examples in the validation plan to detect unnecessary distortion.

## Splits and records

Use separate manifests such as `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Exclude shared speakers and duplicate recordings where the evaluation design requires independence. Record corpus versions, checksums, split construction, sample-rate conversion performed before manifest creation, and any filtering.

Local audio, generated teacher outputs, and training artifacts should remain outside version control unless the repository owner has explicitly selected and permitted their inclusion. Do not assume that a dataset's availability or use in a cited paper permits redistribution or commercial training.

## Smoke data

The training override `smoke=true` generates tiny deterministic tones/noise locally and substitutes a one-block, width-4 model. It needs no manifest or external data. It exists to exercise software behavior, not to create a useful pretrained speech model or measure the supplied full-size configurations.
