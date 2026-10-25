import json
from types import SimpleNamespace

import pytest
import torch

from realtime_use.benchmark import summarize_timings
from realtime_use.config import ModelConfig
from realtime_use.evaluate import evaluate_dataset, signal_metrics, summarize_metrics
from realtime_use.infer import enhance_waveform, load_model
from realtime_use.model import Enhancer


def test_si_sdr_gain_invariance_and_snr_amplitude_sensitivity():
    generator = torch.Generator().manual_seed(10)
    reference = torch.randn(2000, generator=generator, dtype=torch.float64)
    estimate = reference + 0.1 * torch.randn(2000, generator=generator, dtype=torch.float64)
    metrics = signal_metrics(estimate, reference)
    scaled = signal_metrics(2 * estimate, reference)
    assert metrics["si_sdr_db"] == pytest.approx(scaled["si_sdr_db"], abs=1e-9)
    assert metrics["snr_db"] > scaled["snr_db"] + 10
    assert 15 < metrics["si_sdr_db"] < 25


def test_silence_metrics_remain_defined_only_when_meaningful():
    silent_reference = signal_metrics(torch.ones(5), torch.zeros(5))
    assert silent_reference["mse"] == 1
    assert silent_reference["si_sdr_db"] is None
    assert silent_reference["snr_db"] is None
    silent_estimate = signal_metrics(torch.zeros(5), torch.tensor([1.0, -1, 1, -1, 1]))
    assert silent_estimate["si_sdr_db"] is None
    assert silent_estimate["snr_db"] == pytest.approx(0)
    summary = summarize_metrics([silent_reference])
    assert summary["si_sdr_db"] == {"mean": None, "valid_utterances": 0}
    json.dumps(summary, allow_nan=False)


def test_metrics_reject_implicit_alignment_and_nonfinite_audio():
    with pytest.raises(ValueError):
        signal_metrics(torch.ones(3), torch.ones(4))
    with pytest.raises(ValueError, match="finite"):
        signal_metrics(torch.tensor([float("nan")]), torch.ones(1))


class ScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.config = SimpleNamespace(sample_rate=16000, hop_ms=20.0, window_ms=40.0, exits=(1, 2))

    def enhance(self, waveform, exit_depths=None):
        return {depth: waveform * self.scale for depth in exit_depths}


def test_evaluation_has_bypass_clean_preservation_and_no_quality_claim():
    clean = torch.tensor([0.2, -0.4, 0.3, -0.1])
    dataset = [{"noisy": clean * 2, "clean": clean, "id": "heldout-a"}]
    result = evaluate_dataset(ScaleModel(), dataset, check_clean=True)
    assert result["summary"]["bypass"]["mse"]["mean"] > 0
    assert result["summary"]["exits"]["1"]["mse"]["mean"] == 0
    assert result["summary"]["clean_input_preservation"]["1"]["mse"]["mean"] > 0
    assert "production readiness" in result["limitations"]
    assert "si_sdr_db" in result["metric_implementations"]
    json.dumps(result, allow_nan=False)


def test_inference_stream_state_does_not_leak_across_files(monkeypatch):
    class StatefulStream:
        def __init__(self, model, exit_depth):
            self.accumulated = 0
            self.batch = 1

        def push(self, chunk):
            self.batch = chunk.shape[0]
            output = chunk + self.accumulated
            self.accumulated += float(chunk.sum())
            return output

        def flush(self):
            return torch.empty(self.batch, 0)

    monkeypatch.setattr("realtime_use.infer.StreamingEnhancer", StatefulStream)
    model = ScaleModel()
    waveform = torch.arange(8, dtype=torch.float32)
    first = enhance_waveform(model, waveform, 1, streaming=True, chunk_samples=3)
    second = enhance_waveform(model, waveform, 1, streaming=True, chunk_samples=3)
    torch.testing.assert_close(first, second)
    assert first.shape == (1, 8)


def test_timing_quantiles_and_deadline_misses():
    result = summarize_timings([1, 2, 3, 25], hop_ms=20)
    assert result["p50_ms"] == 2.5
    assert result["p95_ms"] == pytest.approx(21.7)
    assert result["p99_ms"] == pytest.approx(24.34)
    assert result["deadline_misses"] == 1
    assert result["deadline_miss_fraction"] == 0.25
    with pytest.raises(ValueError):
        summarize_timings([], 20)


def test_checkpoint_must_describe_model_and_contain_weights(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"step": 3}, path)
    with pytest.raises(ValueError, match="model_config and model_state"):
        load_model(path)


def test_cuda_scan_checkpoint_loads_and_runs_on_cpu(tmp_path):
    config = ModelConfig(sample_rate=8000, window_ms=4, hop_ms=2, depth=1,
                         exits=(1,), width=4, state_dim=2, expand=1,
                         conv_kernel=2, scan_backend="mamba_ssm")
    original = Enhancer(config)
    path = tmp_path / "cuda-backend.pt"
    torch.save({"model_config": config.to_dict(), "model_state": original.state_dict()}, path)
    loaded = load_model(path, "cpu")
    assert loaded.config.scan_backend == "reference"
    output = enhance_waveform(loaded, torch.randn(97), exit_depth=1)
    assert output.shape == (1, 97)
    assert torch.isfinite(output).all()
    assert original.config.scan_backend == "mamba_ssm"
