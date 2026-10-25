"""Offline adapter contracts; mocked learned scorers are not metric validation."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from realtime_use import quality_metrics as qm


def test_no_metrics_requires_no_optional_backend_or_assets(monkeypatch):
    monkeypatch.setattr(qm, "_dependency", lambda *args: pytest.fail("unexpected import"))
    scorer = qm.QualityEvaluator()
    assert scorer.score(torch.ones(8), sample_rate=48000) == {}
    assert scorer.metadata == {}
    scorer.validate_assets()


def test_utmos_missing_assets_fails_before_hub_or_any_network(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: pytest.fail("hub must not run"))
    with pytest.raises(qm.MetricUnavailableError, match="Explicit local repo"):
        qm.QualityEvaluator(["utmos"])
    with pytest.raises(qm.MetricUnavailableError, match="missing"):
        qm.QualityEvaluator(["utmos"], config={"utmos": {"repo": str(tmp_path)}})


def test_pinned_source_content_is_checked_not_directory_name(tmp_path, monkeypatch):
    (tmp_path / "backend.py").write_text("original")
    raw = b"original"
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setitem(qm.SOURCE_PINS, "fixture", {"revision": "fixed", "files": {"backend.py": blob}})
    assert qm._pinned_repo("fixture", {"repo": str(tmp_path)}) == tmp_path
    (tmp_path / "backend.py").write_text("changed")
    with pytest.raises(qm.MetricUnavailableError, match="differs from pinned"):
        qm._pinned_repo("fixture", {"repo": str(tmp_path)})


def test_utmos_contract_local_pretrained_false_strict_weights_and_whole_utterance(tmp_path, monkeypatch):
    # A tiny fake scorer verifies loading/calling, not official UTMOS numerics.
    class FakeUTMOS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))
        def forward(self, audio, rate):
            assert audio.shape == (1, 23)
            assert rate == 44100
            return audio.mean(dim=-1) * self.scale
    checkpoint = tmp_path / "utmos.pt"
    torch.save(FakeUTMOS().state_dict(), checkpoint)
    monkeypatch.setattr(qm, "_pinned_repo", lambda name, config: tmp_path)
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeUTMOS()
    monkeypatch.setattr(torch.hub, "load", load)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: pytest.fail("network download"))
    scorer = qm.QualityEvaluator(["utmos"], config={"utmos": {"checkpoint": str(checkpoint)}})
    assert calls == [((str(tmp_path), "utmos22_strong"), {"source": "local", "pretrained": False})]
    assert scorer.score(torch.ones(23), torch.ones(23), 44100) == {"utmos": 2.0}
    assert not scorer.backends["utmos"].model.training
    assert not scorer.backends["utmos"].model.scale.requires_grad
    assert "not UTMOSv2" in scorer.metadata["utmos"]["backend"]
    assert len(scorer.metadata["utmos"]["checkpoint"]["sha256"]) == 64
    json.dumps(scorer.metadata)


@pytest.mark.parametrize("name", ["nisqa", "speechbert_score"])
def test_restricted_asset_backend_needs_explicit_permission_statement(name):
    with pytest.raises(qm.MetricUnavailableError, match="does not grant commercial rights"):
        qm.QualityEvaluator([name])


def test_pesq_needs_separate_license_and_keeps_reference_order(monkeypatch):
    with pytest.raises(qm.MetricUnavailableError, match="separate commercial-use restrictions"):
        qm.QualityEvaluator(["pesq"])
    calls = []
    def pesq(rate, reference, estimate, mode):
        calls.append((rate, reference.copy(), estimate.copy(), mode))
        return 2.5
    monkeypatch.setattr(qm, "_dependency", lambda *args: SimpleNamespace(pesq=pesq))
    scorer = qm.QualityEvaluator(["pesq"], config={"pesq": {"license_statement": "test-only fake backend"}})
    assert scorer.score(torch.ones(10), torch.ones(10) * 2, 8000) == {"pesq": 2.5}
    assert calls[0][0] == 8000 and calls[0][3] == "nb"
    np.testing.assert_array_equal(calls[0][1], 2)
    np.testing.assert_array_equal(calls[0][2], 1)


def test_estoi_real_library_adapter_and_undefined_short_audio():
    pystoi = pytest.importorskip("pystoi")
    pytest.importorskip("soxr")
    generator = torch.Generator().manual_seed(31)
    reference = torch.randn(30000, generator=generator)
    estimate = reference + .1 * torch.randn(30000, generator=generator)
    scorer = qm.QualityEvaluator(["estoi"])
    actual = scorer.score(estimate, reference, 10000)["estoi"]
    expected = pystoi.stoi(reference.numpy(), estimate.numpy(), 10000, extended=True)
    assert actual == pytest.approx(expected)
    with pytest.raises(qm.MetricUnavailableError, match="estoi|ESTOI"):
        scorer.score(torch.zeros(50), torch.zeros(50), 10000)


def test_dnsmos_contract_uses_official_local_files_and_full_audio(tmp_path, monkeypatch):
    primary, p808 = tmp_path / "primary.onnx", tmp_path / "p808.onnx"
    primary.write_bytes(b"fake"); p808.write_bytes(b"fake")
    calls = []
    class ComputeScore:
        def __init__(self, first, second):
            assert first == str(primary) and second == str(p808)
        def __call__(self, filename, rate, personalized):
            import soundfile as sf
            audio, saved_rate = sf.read(filename)
            calls.append((len(audio), saved_rate, rate, personalized))
            return {"OVRL": 3.2}
    monkeypatch.setattr(qm, "_pinned_repo", lambda *args: tmp_path)
    monkeypatch.setattr(qm, "_source_module", lambda *args: SimpleNamespace(ComputeScore=ComputeScore))
    scorer = qm.QualityEvaluator(["dnsmos"], config={"dnsmos": {"primary_model": str(primary), "p808_model": str(p808)}})
    assert scorer.score(torch.ones(321), sample_rate=16000) == {"dnsmos": 3.2}
    assert calls == [(321, 16000, 16000, False)]


def test_nisqa_author_cli_contract_isolated_trusted_local_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "nisqa.tar"; checkpoint.write_bytes(b"fake")
    monkeypatch.setattr(qm, "_pinned_repo", lambda *args: tmp_path)
    settings = {"checkpoint": str(checkpoint), "assets_license": "test fixture permission"}
    with pytest.raises(qm.MetricUnavailableError, match="trust_checkpoint"):
        qm.QualityEvaluator(["nisqa"], config={"nisqa": settings})
    settings["trust_checkpoint"] = True
    def run(command, **kwargs):
        assert "run_predict.py" in command[1]
        assert command[command.index("--pretrained_model") + 1] == str(checkpoint)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        result = Path(command[command.index("--output_dir") + 1]) / "NISQA_results.csv"
        result.write_text("mos_pred\n3.4\n")
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(qm.subprocess, "run", run)
    scorer = qm.QualityEvaluator(["nisqa"], config={"nisqa": settings})
    assert scorer.score(torch.ones(100), sample_rate=48000) == {"nisqa": 3.4}


def test_speechbert_contract_local_model_layer8_official_precision(tmp_path, monkeypatch):
    calls = []
    class Model(torch.nn.Module):
        def forward(self, audio, output_hidden_states):
            assert output_hidden_states
            return SimpleNamespace(hidden_states=[torch.ones(1, 3, 2) * i for i in range(13)])
    def from_pretrained(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return Model()
    def bert_score(estimate, reference):
        assert torch.all(estimate == 8) and torch.all(reference == 8)
        calls.append(True)
        return .8, .7, .75
    monkeypatch.setattr(qm, "_pinned_repo", lambda *args: tmp_path)
    monkeypatch.setattr(qm, "_source_module", lambda *args: SimpleNamespace(bert_score=bert_score))
    monkeypatch.setattr(qm, "_dependency", lambda *args: SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=from_pretrained)))
    scorer = qm.QualityEvaluator(["speechbert_score"], config={"speechbert_score": {
        "model_dir": str(tmp_path), "assets_license": "test fixture permission"}})
    assert scorer.score(torch.ones(100), torch.ones(100), 16000) == {"speechbert_score": .8}
    assert calls == [True]


def test_requested_backend_nonfinite_or_failure_is_never_silently_replaced(monkeypatch):
    class Backend:
        metadata = {"backend": "fake"}
        def __init__(self, settings, device): pass
        def score(self, *args, **kwargs): return float("nan")
    monkeypatch.setitem(qm._BACKENDS, "utmos", Backend)
    scorer = qm.QualityEvaluator(["utmos"])
    with pytest.raises(qm.MetricUnavailableError, match="nonfinite/undefined"):
        scorer.score(torch.ones(10))
    with pytest.raises(ValueError, match="finite audio"):
        scorer.score(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="equal-length"):
        scorer.score(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError, match="positive integer"):
        scorer.score(torch.ones(3), sample_rate=True)


def test_metric_names_are_explicit_no_v2_substitution():
    with pytest.raises(qm.MetricUnavailableError, match="different metric"):
        qm.QualityEvaluator(["utmosv2"])
    with pytest.raises(ValueError, match="unique"):
        qm.QualityEvaluator(["utmos", "utmos"])
    with pytest.raises(ValueError, match="Unknown"):
        qm.QualityEvaluator(["not_a_metric"])


def test_lps_local_transformers_api_and_empty_reference(monkeypatch, tmp_path):
    texts = iter(["a b", "a c", "", "a"])
    calls = []
    class Processor:
        def __call__(self, audio, **kwargs):
            assert kwargs == {"return_tensors": "pt", "sampling_rate": 16000}
            return SimpleNamespace(input_values=torch.as_tensor(audio)[None])
        def batch_decode(self, ids):
            return [next(texts)]
    class Model(torch.nn.Module):
        def forward(self, values): return SimpleNamespace(logits=torch.ones(1, 2, 3))
    def processor_load(path, **kwargs):
        calls.append((path, kwargs)); return Processor()
    def model_load(path, **kwargs):
        calls.append((path, kwargs)); return Model()
    modules = {
        "transformers": SimpleNamespace(Wav2Vec2Processor=SimpleNamespace(from_pretrained=processor_load),
                                         Wav2Vec2ForCTC=SimpleNamespace(from_pretrained=model_load)),
        "jiwer": SimpleNamespace(cer=lambda ref, hyp: .5 if (ref, hyp) == ("ab", "ac") else pytest.fail("wrong phoneme order")),
    }
    monkeypatch.setattr(qm, "_dependency", lambda name, *args: modules[name])
    scorer = qm.QualityEvaluator(["lps"], config={"lps": {"model_dir": str(tmp_path)}})
    assert calls == [(str(tmp_path), {"local_files_only": True})] * 2
    assert scorer.score(torch.ones(100), torch.ones(100), 16000) == {"lps": .5}
    with pytest.raises(qm.MetricUnavailableError, match="reference phoneme sequence is empty"):
        scorer.score(torch.ones(100), torch.ones(100), 16000)


def test_cacc_local_owsm_short_long_and_no_language_download(monkeypatch, tmp_path):
    config, weights = tmp_path / "config.yaml", tmp_path / "model.pth"
    config.write_text("frontend: default\nencoder: e_branchformer\ndecoder: transformer\ntoken_type: bpe\ntoken_list: [a, b]\n")
    weights.write_bytes(b"fake")
    bpemodel = tmp_path / "bpe.model"; bpemodel.write_bytes(b"fake")
    calls = []
    class Speech2Text:
        def __init__(self, **kwargs):
            assert kwargs["s2t_train_config"] == str(config)
            assert kwargs["s2t_model_file"] == str(weights)
            assert kwargs["beam_size"] == 5
        def __call__(self, audio, prompt, **kwargs):
            assert len(audio) == 480000
            calls.append("short")
            return [("unused", [], [], "hi", None)]
        def decode_long(self, audio, **kwargs):
            assert len(audio) > 480000
            calls.append("long")
            return [(0, 25, "hi"), (25, 31, "there")]
    def process(reference, hypothesis):
        assert reference == "hi"
        assert hypothesis in ("hi", "hi there")
        return SimpleNamespace(cer=0.0 if hypothesis == "hi" else 3.0, hits=2, substitutions=0, deletions=0)
    modules = {"espnet2.bin.s2t_inference": SimpleNamespace(Speech2Text=Speech2Text),
               "espnet2.text.cleaner": SimpleNamespace(TextCleaner=lambda name: str.lower),
               "jiwer": SimpleNamespace(process_characters=process),
               "nltk": SimpleNamespace(data=SimpleNamespace(find=lambda *args, **kwargs: True, path=[]),
                                        download=lambda *args: pytest.fail("unexpected download"))}
    monkeypatch.setattr(qm, "_dependency", lambda name, *args: modules[name])
    scorer = qm.QualityEvaluator(["cacc"], config={"cacc": {"train_config": str(config), "checkpoint": str(weights), "bpemodel": str(bpemodel), "nltk_data": str(tmp_path)}})
    assert scorer.score(torch.ones(1000), sample_rate=16000, transcript="Hi", language="eng") == {"cacc": 1.0}
    assert scorer.score(torch.ones(496000), sample_rate=16000, transcript="Hi", language="eng") == {"cacc": -2.0}
    assert calls == ["short", "long"]
    assert scorer.backends["cacc"].last_reference_characters == 2
    with pytest.raises(qm.MetricUnavailableError, match="language ID"):
        scorer.score(torch.ones(1000), sample_rate=16000, transcript="Hi", language="none")
    with pytest.raises(qm.MetricUnavailableError, match="reference transcript"):
        scorer.score(torch.ones(1000), sample_rate=16000, language="eng")


def test_nltk_resource_preflight_fails_without_downloading(monkeypatch, tmp_path):
    attempts = []
    def missing(resource, paths):
        assert paths == [str(tmp_path)]
        raise LookupError(resource)
    fake = SimpleNamespace(data=SimpleNamespace(find=missing, path=["original"]),
                           download=lambda *args: attempts.append(args))
    monkeypatch.setattr(qm, "_dependency", lambda *args: fake)
    with pytest.raises(qm.MetricUnavailableError, match="averaged_perceptron_tagger_eng.zip"):
        with qm._local_nltk_resources({"nltk_data": str(tmp_path)}):
            pytest.fail("must fail before ESPnet import")
    assert attempts == []
    assert fake.data.path == ["original"]


def test_nltk_import_time_download_attempt_is_blocked_and_state_restored(monkeypatch, tmp_path):
    attempts = []
    downloader = lambda *args: attempts.append(args)
    fake = SimpleNamespace(data=SimpleNamespace(find=lambda *args, **kwargs: True, path=["original"]),
                           download=downloader)
    monkeypatch.setattr(qm, "_dependency", lambda *args: fake)
    with pytest.raises(qm.MetricUnavailableError, match="attempted an NLTK download"):
        with qm._local_nltk_resources({"nltk_data": str(tmp_path)}):
            assert fake.data.path == [str(tmp_path)]
            # Simulates an import-time dependency call, before network I/O.
            fake.download("unexpected-resource")
    assert attempts == []
    assert fake.download is downloader
    assert fake.data.path == ["original"]


def test_metric_scoring_disables_caller_amp_and_uses_float32(monkeypatch):
    class Backend:
        metadata = {"backend": "fake precision contract"}
        def __init__(self, settings, device): pass
        def score(self, estimate, reference, sample_rate, **kwargs):
            assert not torch.is_autocast_enabled("cpu")
            assert estimate.dtype == reference.dtype == torch.float32
            return 3.0
    monkeypatch.setitem(qm._BACKENDS, "utmos", Backend)
    scorer = qm.QualityEvaluator(["utmos"], device="cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        values = scorer.score(torch.ones(12, dtype=torch.bfloat16), torch.ones(12, dtype=torch.bfloat16))
        assert torch.is_autocast_enabled("cpu")
    assert values == {"utmos": 3.0}
    assert "autocast disabled" in scorer.metadata["utmos"]["precision"]
