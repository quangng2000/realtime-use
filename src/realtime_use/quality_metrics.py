"""Optional published quality scorers. Assets are local and never downloaded.

The adapters contain no metric models or learned weights. Constructor errors are
intentional: a requested score must never become a zero or a substitute metric.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import importlib
import importlib.util
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import torch


class MetricUnavailableError(RuntimeError):
    """The requested published metric cannot run with the supplied local setup."""


SOURCE_PINS = {
    "utmos": {
        "repository": "https://github.com/tarepan/SpeechMOS",
        "revision": "ed25eacbfa42b99156c36ebec67a733b5dbb9b79",
        "files": {
            "hubconf.py": "2fa3c1dce1890ba7bf2cda2a6eea5627c9cd80e1",
            "speechmos/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
            "speechmos/utmos22/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
            "speechmos/utmos22/strong/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
            "speechmos/utmos22/fairseq_alt.py": "4c0fa5e0920a7dcdf96455b79a5b1c5875720428",
            "speechmos/utmos22/strong/model.py": "8a2a4408f82dd2cc39c3361f25148677cef09273",
        },
    },
    "dnsmos": {
        "repository": "https://github.com/microsoft/DNS-Challenge",
        "revision": "591184a9fcb2cbdec02520fed81a32bbbf9d73ff",
        "files": {"DNSMOS/dnsmos_local.py": "e32032e97541fde9a98fdc12745d5c1e09e5c7f1"},
    },
    "speechbert_score": {
        "repository": "https://github.com/kohei0209/DiscreteSpeechMetrics",
        "revision": "e7c2d365b289403e4a0afe3444147b96089b4664",
        "files": {"discrete_speech_metrics/speechbertscore.py": "ede8fdb29ac010335c052d8cbb7646fb6f879fbe"},
    },
    "nisqa": {
        "repository": "https://github.com/gabrielmittag/NISQA",
        "revision": "fe84f0f252abec382b24367d5b22498a7ce34dbb",
        "files": {
            "run_predict.py": "47ef398ce4c55f5be82ea6ccf247b54f6365d7a1",
            "nisqa/NISQA_model.py": "665b5afb7e55657856b0d17c292dc08819c0d089",
            "nisqa/NISQA_lib.py": "e813e441cdbf88a078d38302df914fa06955e5e5",
        },
    },
}

SUPPORTED_METRICS = ("pesq", "utmos", "estoi", "dnsmos", "nisqa", "speechbert_score", "lps", "cacc")
UNAVAILABLE_METRICS = {
    "utmosv2": "UTMOSv2 is a different metric from the paper's original UTMOS22 strong learner and is not substituted.",
}


def _dependency(module: str, instruction: str) -> Any:
    try:
        return importlib.import_module(module)
    except (ImportError, OSError) as exc:
        raise MetricUnavailableError(f"Missing/incompatible optional dependency {module}: {instruction}") from exc


def _path(config: Mapping[str, Any], key: str, *, directory: bool = False) -> Path:
    value = config.get(key)
    if not isinstance(value, (str, Path)) or not str(value):
        raise MetricUnavailableError(f"Explicit local {key} is required; automatic downloads are disabled")
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise MetricUnavailableError(f"Local {key} does not exist: {path}; automatic downloads are disabled")
    return path


def _pinned_repo(name: str, config: Mapping[str, Any]) -> Path:
    root = _path(config, "repo", directory=True)
    pin = SOURCE_PINS[name]
    for relative, expected in pin["files"].items():
        path = root / relative
        if not path.is_file():
            raise MetricUnavailableError(f"{name}: missing {path}; use source revision {pin['revision']}")
        raw = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != expected:
            raise MetricUnavailableError(f"{name}: source differs from pinned revision {pin['revision']}: {relative}")
    return root


def _source_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_realtime_use_metric_{name}", path)
    if spec is None or spec.loader is None:
        raise MetricUnavailableError(f"Cannot import local metric source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _asset_metadata(path: Path) -> dict[str, str]:
    # Hash local bytes once at construction; reports can identify exact weights.
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}



def _directory_metadata(path: Path) -> dict[str, Any]:
    assets = [item for item in sorted(path.iterdir()) if item.is_file()
              and item.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}]
    return {"path": str(path), "files": {item.name: _asset_metadata(item)["sha256"] for item in assets},
            "identity": "caller-provided local files; hashes record bytes, not an assertion of original training provenance"}

def _resample(wave: torch.Tensor, sample_rate: int, target_rate: int) -> Any:
    audio = wave.detach().cpu().float().numpy()
    if sample_rate != target_rate:
        soxr = _dependency("soxr", "install soxr==0.5.0.post1")
        audio = soxr.resample(audio, sample_rate, target_rate, quality="HQ")
    return audio


def _license_ack(name: str, config: Mapping[str, Any]) -> str:
    declaration = config.get("assets_license")
    if not isinstance(declaration, str) or not declaration.strip():
        raise MetricUnavailableError(
            f"{name}: official pretrained assets carry CC-BY-NC-SA-4.0. "
            "Set assets_license to your applicable license/permission statement for explicitly provided local assets. "
            "This acknowledgement does not grant commercial rights.")
    return declaration


class _UTMOS:
    def __init__(self, config: Mapping[str, Any], device: str):
        repo = _pinned_repo("utmos", config)
        checkpoint = _path(config, "checkpoint")
        # All SSL/model architecture code in this pinned release is local.
        # pretrained=False avoids the sole load_state_dict_from_url call.
        for name, module in tuple(sys.modules.items()):
            if name == "speechmos" or name.startswith("speechmos."):
                filename = getattr(module, "__file__", None)
                if filename is None or not Path(filename).resolve().is_relative_to(repo):
                    raise MetricUnavailableError("Another SpeechMOS version is already imported; use a fresh process")
        self.model = torch.hub.load(str(repo), "utmos22_strong", source="local", pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = device
        self.metadata = {"backend": "SpeechMOS v1.2.0 utmos22_strong; original UTMOS22 strong learner, not UTMOSv2/full ensemble",
                         "source": SOURCE_PINS["utmos"], "checkpoint": _asset_metadata(checkpoint),
                         "license": "MIT; original UTMOS22 and converted checkpoint attribution required",
                         "resampling": "upstream torchaudio resample to 16 kHz", "batch_size": 1}

    def score(self, estimate, reference, sample_rate, **kwargs):
        return self.model(estimate.to(self.device, dtype=torch.float32).unsqueeze(0), sample_rate).item()


class _ESTOI:
    def __init__(self, config, device):
        self.stoi = _dependency("pystoi", "install pystoi==0.4.1").stoi
        self.metadata = {"backend": "pystoi.stoi(extended=True)", "license": "MIT",
                         "resampling": "soxr HQ to 10 kHz; no internal rate conversion afterwards"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        # pystoi returns 1e-5 with a warning for too-short/silent signals; treat
        # that condition as an error rather than a fabricated quality result.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Not enough STFT frames.*")
            try:
                return self.stoi(_resample(reference, sample_rate, 10000),
                                 _resample(estimate, sample_rate, 10000), 10000, extended=True)
            except Warning as exc:
                raise MetricUnavailableError("ESTOI is undefined: insufficient nonsilent frames") from exc


class _PESQ:
    def __init__(self, config, device):
        statement = config.get("license_statement")
        if not isinstance(statement, str) or not statement.strip():
            raise MetricUnavailableError(
                "PESQ's ITU C implementation has separate commercial-use restrictions despite the wrapper MIT license. "
                "An externally installed, appropriately licensed pesq backend and explicit license_statement are required; "
                "this statement does not grant rights.")
        self.pesq = _dependency("pesq", "provide your appropriately licensed pesq==0.0.4 installation").pesq
        self.metadata = {"backend": "python-pesq wrapper around ITU PESQ", "license_statement": statement,
                         "resampling": "native 8 kHz narrowband; otherwise soxr HQ to 16 kHz wideband",
                         "source": "https://github.com/ludlows/PESQ/tree/v0.0.4",
                         "license_notice": "https://github.com/ludlows/PESQ/blob/master/pesq/pesq.h"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        if sample_rate != 8000 and sample_rate < 16000:
            raise ValueError("URGENT PESQ requires 8 kHz or at least 16 kHz input")
        rate = 8000 if sample_rate == 8000 else 16000
        return self.pesq(rate, _resample(reference, sample_rate, rate),
                         _resample(estimate, sample_rate, rate), mode="nb" if rate == 8000 else "wb")


class _DNSMOS:
    def __init__(self, config, device):
        repo = _pinned_repo("dnsmos", config)
        primary, p808 = _path(config, "primary_model"), _path(config, "p808_model")
        source = _source_module("dnsmos", repo / "DNSMOS/dnsmos_local.py")
        self.model = source.ComputeScore(str(primary), str(p808))
        self.metadata = {"backend": "Microsoft DNSMOS ComputeScore, nonpersonalized calibrated OVRL",
                         "source": SOURCE_PINS["dnsmos"], "primary_model": _asset_metadata(primary),
                         "p808_model": _asset_metadata(p808), "license": "DNS-Challenge CC-BY-4.0",
                         "device": "upstream ONNX Runtime providers", "resampling": "soxr HQ to 16 kHz"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        sf = _dependency("soundfile", "install soundfile")
        with tempfile.TemporaryDirectory(prefix="realtime-use-dnsmos-") as folder:
            filename = Path(folder) / "audio.wav"
            sf.write(filename, _resample(estimate, sample_rate, 16000), 16000, subtype="FLOAT")
            return self.model(str(filename), 16000, False)["OVRL"]


class _SpeechBERTScore:
    def __init__(self, config, device):
        license_statement = _license_ack("speechbert_score", config)
        repo = _pinned_repo("speechbert_score", config)
        model_dir = _path(config, "model_dir", directory=True)
        source = _source_module("speechbert", repo / "discrete_speech_metrics/speechbertscore.py")
        transformers = _dependency("transformers", "install transformers==4.57.6")
        self.model = transformers.AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                                            trust_remote_code=False).to(device).eval()
        self.model.requires_grad_(False)
        self.bert_score, self.device = source.bert_score, device
        self.metadata = {"backend": "URGENT DiscreteSpeechMetrics mHuBERT-147 layer 8; precision",
                         "source": SOURCE_PINS["speechbert_score"], "model_dir": _directory_metadata(model_dir),
                         "assets_license": license_statement, "expected_asset": "utter-project/mHuBERT-147",
                         "resampling": "soxr HQ to 16 kHz"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        features = []
        for wave in (estimate, reference):
            tensor = torch.as_tensor(_resample(wave, sample_rate, 16000), device=self.device).float()[None]
            features.append(self.model(tensor, output_hidden_states=True).hidden_states[8].squeeze(0))
        return self.bert_score(*features)[0]


class _NISQA:
    def __init__(self, config, device):
        license_statement = _license_ack("nisqa", config)
        self.repo = _pinned_repo("nisqa", config)
        self.checkpoint = _path(config, "checkpoint")
        if config.get("trust_checkpoint") is not True:
            raise MetricUnavailableError("NISQA author API loads a legacy pickle; trust_checkpoint=true is required for your trusted local checkpoint")
        self.python = str(_path(config, "python")) if config.get("python") else sys.executable
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise MetricUnavailableError("NISQA author CLI supports CPU/CUDA only")
        self.timeout = float(config.get("timeout_seconds", 180))
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("NISQA timeout_seconds must be finite and positive")
        self.metadata = {"backend": "author NISQA run_predict.py predict_file; mos_pred", "source": SOURCE_PINS["nisqa"],
                         "checkpoint": _asset_metadata(self.checkpoint), "assets_license": license_statement,
                         "execution": "isolated author CLI per utterance, slower than a persistent scorer",
                         "resampling": "upstream NISQA preprocessing"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        sf = _dependency("soundfile", "install soundfile")
        with tempfile.TemporaryDirectory(prefix="realtime-use-nisqa-") as folder:
            path = Path(folder) / "audio.wav"
            sf.write(path, estimate.cpu().float().numpy(), sample_rate, subtype="FLOAT")
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
            # The upstream CLI chooses the first visible GPU automatically.
            if self.device.type == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            elif self.device.index is not None:
                visible = env.get("CUDA_VISIBLE_DEVICES")
                env["CUDA_VISIBLE_DEVICES"] = (visible.split(",")[self.device.index] if visible else str(self.device.index))
            process = subprocess.run([self.python, str(self.repo / "run_predict.py"),
                "--mode", "predict_file", "--pretrained_model", str(self.checkpoint),
                "--deg", str(path), "--output_dir", folder, "--num_workers", "0", "--bs", "1"],
                cwd=self.repo, env=env, capture_output=True, text=True, timeout=self.timeout)
            if process.returncode:
                raise MetricUnavailableError("NISQA author CLI failed: " + process.stderr[-2000:])
            result = Path(folder) / "NISQA_results.csv"
            if not result.is_file():
                raise MetricUnavailableError("NISQA author CLI did not produce NISQA_results.csv")
            with result.open() as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or "mos_pred" not in rows[0]:
                raise MetricUnavailableError("Unexpected NISQA output schema")
            return float(rows[0]["mos_pred"])


class _LPS:
    def __init__(self, config, device):
        folder = _path(config, "model_dir", directory=True)
        transformers = _dependency("transformers", "install transformers==4.57.6 and phonemizer==3.3.0; eSpeak-NG runtime may be needed")
        self.cer = _dependency("jiwer", "install jiwer==4.0.0").cer
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(str(folder), local_files_only=True)
        self.model = transformers.Wav2Vec2ForCTC.from_pretrained(str(folder), local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.metadata = {"backend": "HF Wav2Vec2ForCTC phonemes + jiwer.cer; 1 - character-level phoneme error rate",
                         "expected_asset": "facebook/wav2vec2-lv-60-espeak-cv-ft", "model_dir": _directory_metadata(folder),
                         "license": "model Apache-2.0; transformers Apache-2.0; jiwer Apache-2.0; phonemizer/eSpeak licenses separate",
                         "resampling": "soxr HQ to 16 kHz", "normalization": "remove spaces in decoded phonemes, following URGENT"}

    def score(self, estimate, reference, sample_rate, **kwargs):
        text = []
        for wave in (reference, estimate):
            values = self.processor(_resample(wave, sample_rate, 16000), return_tensors="pt", sampling_rate=16000).input_values
            logits = self.model(values.to(self.device)).logits
            text.append(self.processor.batch_decode(logits.argmax(dim=-1))[0].replace(" ", ""))
        if not text[0]:
            raise MetricUnavailableError("LPS is undefined: reference phoneme sequence is empty")
        return 1 - self.cer(text[0], text[1])



@contextmanager
def _local_nltk_resources(config: Mapping[str, Any]):
    """ESPnet's g2p_en import may download corpora even for non-G2P ASR.

    Require explicitly provisioned resources, and reject any unexpected download
    call before it reaches NLTK's downloader. Restore the library state on exit.
    """
    root = _path(config, "nltk_data", directory=True)
    nltk = _dependency("nltk", "install the ESPnet dependencies and provision local NLTK resources separately")
    resources = ("taggers/averaged_perceptron_tagger_eng.zip", "corpora/cmudict.zip")
    for resource in resources:
        try:
            nltk.data.find(resource, paths=[str(root)])
        except LookupError as exc:
            raise MetricUnavailableError(
                f"CAcc requires local NLTK resource {resource} under {root}; automatic NLTK downloads are disabled") from exc
    previous_path, previous_download = nltk.data.path, nltk.download
    def reject_download(*args, **kwargs):
        raise MetricUnavailableError("ESPnet dependency attempted an NLTK download; provision compatible local resources before scoring")
    nltk.data.path = [str(root)]
    nltk.download = reject_download
    try:
        yield root
    finally:
        nltk.data.path = previous_path
        nltk.download = previous_download

class _CharacterAccuracy:
    def __init__(self, config, device):
        train_config, checkpoint = _path(config, "train_config"), _path(config, "checkpoint")
        bpemodel = _path(config, "bpemodel")
        import yaml
        architecture = yaml.safe_load(train_config.read_text())
        if not isinstance(architecture, dict) or any(architecture.get(key) != value for key, value in {
            "frontend": "default", "encoder": "e_branchformer", "decoder": "transformer", "token_type": "bpe"
        }.items()) or not isinstance(architecture.get("token_list"), list):
            raise MetricUnavailableError("CAcc expects the local OWSM v3.1 EBF configuration with default frontend, inline token_list, and BPE tokenizer; remote/pretrained frontends are not loaded")
        with _local_nltk_resources(config) as nltk_root:
            inference = _dependency("espnet2.bin.s2t_inference", "install espnet==202509 plus openai-whisper and jiwer==4.0.0")
            self.cleaner = _dependency("espnet2.text.cleaner", "install espnet==202509").TextCleaner("whisper_basic")
            self.process_characters = _dependency("jiwer", "install jiwer==4.0.0").process_characters
            self.model = inference.Speech2Text(s2t_train_config=str(train_config), s2t_model_file=str(checkpoint),
                                              device=device, task_sym="<asr>", beam_size=5, predict_time=False, bpemodel=str(bpemodel))
        self.metadata = {"backend": "ESPnet OWSM v3.1 E-Branchformer + whisper_basic normalization + jiwer CER",
                         "expected_asset": "espnet/owsm_v3.1_ebf", "train_config": _asset_metadata(train_config),
                         "checkpoint": _asset_metadata(checkpoint), "bpemodel": _asset_metadata(bpemodel), "nltk_data": str(nltk_root), "license": "OWSM CC-BY-4.0; ESPnet Apache-2.0",
                         "aggregation": "utterance mean; also report corpus CAcc weighted by normalized reference characters",
                         "long_form": "decode_long text joined without timestamps; no 30-second truncation fallback",
                         "resampling": "soxr HQ to 16 kHz"}
        self.last_reference_characters = 0

    def score(self, estimate, reference, sample_rate, *, transcript=None, language=None):
        if not isinstance(transcript, str) or not transcript or transcript == "<not-available>":
            raise MetricUnavailableError("CAcc requires a nonempty reference transcript per utterance")
        if not isinstance(language, str) or not language or language == "none":
            raise MetricUnavailableError("CAcc requires a language ID, e.g. eng; automatic language-model downloads are disabled")
        normalized = self.cleaner(transcript)
        if not normalized:
            raise MetricUnavailableError("CAcc is undefined: normalized reference transcript is empty")
        wave = _resample(estimate, sample_rate, 16000)
        if len(wave) > 30 * 16000:
            self.model.maxlenratio = -300
            decoded = self.model.decode_long(wave, condition_on_prev_text=False, init_text="",
                                            end_time_threshold="<29.00>", lang_sym=f"<{language}>", task_sym="<asr>")
            hypothesis = " ".join(part[2] for part in decoded)
        else:
            import numpy as np
            self.model.maxlenratio = -max(1, min(300, int(len(wave) / 16000 * 10)))
            wave = np.pad(wave, (0, 30 * 16000 - len(wave)))
            hypothesis = self.model(wave, "", lang_sym=f"<{language}>", task_sym="<asr>")[0][-2]
        measured = self.process_characters(normalized, self.cleaner(hypothesis))
        self.last_reference_characters = measured.hits + measured.substitutions + measured.deletions
        return 1 - measured.cer


_BACKENDS = {"pesq": _PESQ, "utmos": _UTMOS, "estoi": _ESTOI, "dnsmos": _DNSMOS, "nisqa": _NISQA,
             "speechbert_score": _SpeechBERTScore, "lps": _LPS, "cacc": _CharacterAccuracy}


class QualityEvaluator:
    """Score full unpadded utterances with explicitly enabled optional backends.

    No metric is enabled by default. Local source/asset availability is checked
    during construction (before training if used for checkpoint selection).
    ``config`` maps each metric name to its local-asset settings.
    """
    def __init__(self, metrics: Sequence[str] = (), device: str = "cpu", config: Mapping[str, Any] | None = None):
        if isinstance(metrics, str):
            raise ValueError("metrics must be a sequence of metric names, not a comma-separated string")
        self.metrics = tuple(metrics)
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metric names must be unique")
        self.device, self.config = str(device), dict(config or {})
        self.backends = {}
        for name in self.metrics:
            if name in UNAVAILABLE_METRICS:
                raise MetricUnavailableError(UNAVAILABLE_METRICS[name])
            if name not in _BACKENDS:
                raise ValueError(f"Unknown quality metric {name!r}; choose {SUPPORTED_METRICS}")
            settings = self.config.get(name) or {}
            if not isinstance(settings, Mapping):
                raise ValueError(f"{name} settings must be a mapping")
            try:
                self.backends[name] = _BACKENDS[name](settings, self.device)
            except MetricUnavailableError:
                raise
            except Exception as exc:
                raise MetricUnavailableError(f"Could not initialize requested {name} backend: {exc}") from exc
        versions = {}
        for package in ("torch", "torchaudio", "pystoi", "soxr", "pesq", "onnxruntime", "transformers", "jiwer", "espnet", "openai-whisper"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.metadata = {name: {**backend.metadata, "installed_package_versions": versions, "precision": "FP32 learned scorer inputs/weights; caller autocast disabled"}
                         for name, backend in self.backends.items()}

    def validate_assets(self) -> None:
        """Construction already validates requested assets; convenient lifecycle API."""

    @torch.inference_mode()
    def score(self, estimate: torch.Tensor, reference: torch.Tensor | None = None, sample_rate: int = 16000,
              *, transcript: str | None = None, language: str | None = None) -> dict[str, float]:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        for name, waveform in (("estimate", estimate), ("reference", reference)):
            if waveform is None and name == "reference":
                continue
            if not isinstance(waveform, torch.Tensor) or waveform.ndim != 1 or not waveform.numel():
                raise ValueError(f"{name} must be a nonempty 1D tensor")
            if not bool(torch.isfinite(waveform).all()):
                raise ValueError(f"{name} must contain finite audio")
        if reference is not None and reference.shape != estimate.shape:
            raise ValueError("quality scoring requires aligned equal-length estimate/reference")
        results = {}
        for name, backend in self.backends.items():
            if reference is None and name in ("pesq", "estoi", "speechbert_score", "lps"):
                raise ValueError(f"{name} requires reference audio")
            try:
                with torch.autocast(device_type=torch.device(self.device).type, enabled=False):
                    value = float(backend.score(estimate.float(), reference.float() if reference is not None else None,
                                                sample_rate, transcript=transcript, language=language))
            except MetricUnavailableError:
                raise
            except Exception as exc:
                raise MetricUnavailableError(f"Requested {name} scoring failed: {exc}") from exc
            if not math.isfinite(value):
                raise MetricUnavailableError(f"Requested {name} returned a nonfinite/undefined score; no fallback is reported")
            results[name] = value
        return results
