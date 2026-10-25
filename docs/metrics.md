# Optional speech quality metrics

Evaluation always reports the existing TorchMetrics signal metrics and noisy-input baseline. Published perceptual and content scorers are optional: none is enabled by default, none downloads a model, and missing dependencies, missing local assets, undefined values, or backend failures raise an error. A requested metric is never replaced with a zero or another model.

These adapters follow the [URGENT 2025 evaluation implementations](https://github.com/urgent-challenge/urgent2025_challenge/tree/main/evaluation_metrics), which provide the practical metric settings used as our reference. They do not establish numerical reproduction of either enhancement paper. The learned scorers have not been executed against pretrained assets in this project. Unit tests use mock scorer contracts; ESTOI additionally has a real-library parity check. No training or pretrained-model download is needed for those tests.

## Evaluate an existing checkpoint

```bash
python -m realtime_use.evaluate \
  --checkpoint /path/to/checkpoint.ckpt \
  --manifest /path/to/heldout.jsonl \
  --device cuda --exits 4,8,12 --lookaheads 0,1,2 \
  --metrics utmos,estoi,dnsmos \
  --metrics-config /path/to/local-metrics.json \
  --output /path/to/evaluation.json
```

Choose only exits, sample rates and lookaheads supported by the checkpoint. Evaluation uses full, unpadded utterances, their native sampling rates, and fresh streaming state when `--stream` is selected. Quality scorers resample their own input as described below; the enhancement model is not silently resampled. Results include each exit/lookahead combination, the noisy baseline, per-rate summaries, backend identities and versions. `--check-clean` additionally evaluates clean-input preservation. Existing top-level `exits` fields refer to `primary_lookahead`; `by_lookahead` contains every requested setting.

A minimal asset configuration is:

```json
{
  "utmos": {
    "repo": "/local/SpeechMOS-v1.2.0",
    "checkpoint": "/local/utmos22_strong_step7459_v1.pt"
  },
  "dnsmos": {
    "repo": "/local/DNS-Challenge",
    "primary_model": "/local/sig_bak_ovr.onnx",
    "p808_model": "/local/model_v8.onnx"
  }
}
```

Paths are explicit local paths, resolved relative to the command's working directory. Obtain source and assets separately, under their applicable licenses. There is no automatic download flag or cache fallback. A missing file must be provisioned before requesting the metric. The `metrics` optional dependency group contains the smaller metric dependencies; it does not include weights. Torch and torchaudio versions must match. Heavy content metrics have separate dependencies.

Programmatic API:

```python
from realtime_use.quality_metrics import QualityEvaluator

scorer = QualityEvaluator(metrics=["utmos"], device="cuda", config=asset_settings)
# Constructor validates/loads the requested UTMOS assets before any training.
scores = scorer.score(estimate_1d, reference_1d, sample_rate=48000)
assert "utmos" in scores
provenance = scorer.metadata
```

`QualityEvaluator` is an ordinary Python object, not a trainable module. Scoring runs under `torch.inference_mode()` with caller autocast disabled and FP32 inputs/learned weights, so Lightning mixed precision does not silently change the metric precision. It accepts optional `transcript` and `language` arguments for CAcc. UTMOS can be used for full-utterance validation checkpoint selection; it is not a training loss. Batch size is one utterance, avoiding padding-dependent UTMOS shifts. Long utterances can make these evaluation models substantially more expensive than the enhancer.

## Backends and local assets

- **`utmos`: original UTMOS22 strong learner.** The adapter uses the exact [SpeechMOS v1.2.0 entry point](https://github.com/tarepan/SpeechMOS/blob/ed25eacbfa42b99156c36ebec67a733b5dbb9b79/hubconf.py) called by [URGENT](https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics/calculate_nonintrusive_mos.py). This is a maintained conversion of the [authors' UTMOS22](https://github.com/sarulab-speech/UTMOS22), not the full ensemble and not UTMOSv2. Source commit `ed25eacbfa42b99156c36ebec67a733b5dbb9b79` is checked against Git blob hashes for every imported model source file. The local hub call sets `source="local", pretrained=False`; the local checkpoint supplies both SSL frontend and MOS head through strict `weights_only=True` loading. Its expected release asset is `utmos22_strong_step7459_v1.pt`. Upstream torchaudio resamples to 16 kHz. Code and converted checkpoint provenance are MIT; preserve both SpeechMOS and original UTMOS22 notices. No additional metric package beyond matching torch/torchaudio is needed.

- **`estoi`: pystoi extended STOI.** `pystoi==0.4.1` supplies `stoi(..., extended=True)`, with soxr HQ resampling to 10 kHz as in URGENT. [pystoi is MIT](https://github.com/mpariente/pystoi/blob/v0.4.1/LICENSE). Short/silent signals for which the library cannot form valid segments fail; the library's warning-based sentinel is not reported as a quality score. No pretrained assets are needed.

- **`dnsmos`: Microsoft DNSMOS overall score.** Requires local [DNS-Challenge source](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py), commit `591184a9fcb2cbdec02520fed81a32bbbf9d73ff`, plus the nonpersonalized `sig_bak_ovr.onnx` and `model_v8.onnx`. The adapter calls the author's `ComputeScore` directly and returns calibrated `OVRL`; upstream segmentation, repetition of short audio and calibration remain in that implementation. Inputs are resampled to 16 kHz and written temporarily as float WAV. ONNX Runtime uses upstream default providers. Source is pinned by content hash, weights are hashed in reports. [DNS-Challenge's license is CC-BY-4.0](https://github.com/microsoft/DNS-Challenge/blob/591184a9fcb2cbdec02520fed81a32bbbf9d73ff/LICENSE). Suggested runtime pins: `onnxruntime==1.22.1`, `soxr==0.5.0.post1`; upstream also imports librosa, numpy, pandas, requests, tqdm and soundfile.

- **`pesq`: optional separately licensed PESQ.** The adapter delegates to an already installed `pesq==0.0.4`, using narrowband mode at 8 kHz and wideband mode at 16 kHz; higher rates are resampled to 16 kHz. Other rates below 16 kHz are rejected. The wrapper's MIT license does not replace the [ITU C code's separate IP notice](https://github.com/ludlows/PESQ/blob/master/pesq/pesq.h). This backend requires `"pesq": {"license_statement": "your applicable permission/provenance"}` and is not installed by the project's metric extras. The acknowledgement records supplied permission; it does not grant rights.

- **`speechbert_score`: URGENT's mHuBERT-147, layer 8, precision score.** Supply `repo` pointing to [URGENT's selected DiscreteSpeechMetrics fork](https://github.com/kohei0209/DiscreteSpeechMetrics/tree/e7c2d365b289403e4a0afe3444147b96089b4664), commit `e7c2d365b289403e4a0afe3444147b96089b4664`, `model_dir` pointing to the complete local Hugging Face model directory, and `assets_license`. The adapter calls the published `bert_score` function without copying its formula, with hidden states from `AutoModel.from_pretrained(..., local_files_only=True, trust_remote_code=False)`. This avoids the upstream class constructor's implicit remote download. The official expected asset is [utter-project/mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147), whose published weights are **CC-BY-NC-SA-4.0**; code is MIT. It is excluded from the default commercial configuration. An explicitly documented separate license can apply to user-provided assets; selecting the backend does not grant that license. Inputs use soxr HQ to 16 kHz. `transformers==4.57.6` is the project's compatible pin.

- **`nisqa`: author NISQA MOS.** Supply pinned [NISQA source](https://github.com/gabrielmittag/NISQA/tree/fe84f0f252abec382b24367d5b22498a7ce34dbb) in `repo`, a local `checkpoint` such as `nisqa.tar`, and `assets_license`. The source is MIT, but the [official weights are CC-BY-NC-SA-4.0](https://github.com/gabrielmittag/NISQA/blob/fe84f0f252abec382b24367d5b22498a7ce34dbb/weights/LICENSE_model_weights). The adapter runs the author's `run_predict.py --mode predict_file` in a temporary directory and reads `mos_pred`. It does not reimplement the model or preprocessing. `trust_checkpoint: true` is required because the original CLI loads a legacy pickle; its subprocess alone sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`. Only trust assets whose provenance you have verified. Optional `python` selects a separate environment with the author's dependencies; `timeout_seconds` defaults to 180. It loads the author model once per utterance and is slower than a persistent scorer. Its upstream feature/resampling behavior is retained. There is no claim this legacy environment has been validated locally.

- **`lps`: phoneme similarity.** Requires a complete local `model_dir` for [facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft), declared Apache-2.0, and local-only Transformers processor/model loading. The adapter removes spaces from decoded phonemes and uses `1 - jiwer.cer`, equivalent to URGENT's character-level Levenshtein normalization for those strings. It rejects an empty reference phoneme sequence. The input is resampled to 16 kHz. Suggested dependencies: `transformers==4.57.6`, `jiwer==4.0.0`, `phonemizer==3.3.0`; phonemizer and eSpeak-NG have GPL terms separate from the model and Transformers licenses. No phoneme recognizer has been downloaded or numerically validated here.

- **`cacc`: ASR character accuracy.** Supply local `train_config`, `checkpoint`, `bpemodel`, and `nltk_data` paths for [espnet/owsm_v3.1_ebf](https://huggingface.co/espnet/owsm_v3.1_ebf). The model card declares CC-BY-4.0; ESPnet is Apache-2.0. The adapter uses the public [ESPnet v.202509 Speech2Text API](https://github.com/espnet/espnet/blob/v.202509/espnet2/bin/s2t_inference.py), beam size 5, the official `whisper_basic` normalizer, and `jiwer` character error rate. Its accepted configuration is OWSM's default audio frontend/E-Branchformer/Transformer with inline tokens and explicit local BPE assets; it does not accept a remote pretrained frontend. ESPnet imports g2p_en even when ASR uses a BPE tokenizer; that dependency can download NLTK data at import time. The adapter requires local `taggers/averaged_perceptron_tagger_eng.zip` and `corpora/cmudict.zip` under `nltk_data`, restricts NLTK searches to that directory during import, and blocks unexpected NLTK download calls. Provision these resources separately; missing resources fail before ESPnet import. Each evaluation manifest row must contain `transcript` and `language` (for example `eng`); missing text/language fails. Longer than 30 seconds uses `decode_long` and joins the text segments without timestamps. A decoding failure never truncates to the first 30 seconds. This differs deliberately from URGENT's old fallback/timestamp formatting, so long-form exact parity is not claimed. Reports include both utterance-mean CAcc and corpus CAcc weighted by normalized reference characters. Suggested dependencies are `espnet==202509`, `openai-whisper==20250625`, and `jiwer==4.0.0`.

Example settings for opt-in content metrics:

```json
{
  "speechbert_score": {
    "repo": "/local/DiscreteSpeechMetrics",
    "model_dir": "/local/mHuBERT-147",
    "assets_license": "record the actual applicable license/permission"
  },
  "nisqa": {
    "repo": "/local/NISQA",
    "checkpoint": "/local/nisqa.tar",
    "assets_license": "record the actual applicable license/permission",
    "trust_checkpoint": true,
    "python": "/local/nisqa-env/bin/python"
  },
  "lps": {"model_dir": "/local/wav2vec2-lv-60-espeak-cv-ft"},
  "cacc": {
    "train_config": "/local/owsm/config.yaml",
    "checkpoint": "/local/owsm/model.pth",
    "bpemodel": "/local/owsm/bpe.model",
    "nltk_data": "/local/nltk_data"
  }
}
```

Source pins identify the inspected implementation. Weight-file SHA256 values identify provided bytes; source checks and architecture names alone do not prove that user-provided weights are the original trained assets. Backend metadata labels expected model identity separately. A commercial deployment must preserve each applicable notice and verify rights to its chosen weights and data; this provenance record is not legal clearance. No weights, restricted implementation code, or external dataset samples are included in this repository.
