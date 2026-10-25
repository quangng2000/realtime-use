"""Real two-process CPU/Gloo integration, including equivalent global gradients."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from realtime_use.config import ModelConfig
from realtime_use.data import PairedAudioDataset, collate_audio
from realtime_use.losses import ReconstructionLoss
from realtime_use.model import Enhancer


pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="CPU distributed integration requires PyTorch Gloo",
)
ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("ddp_worker.py")


def run_process(arguments, *, timeout=180):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    process = subprocess.Popen(arguments, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        pytest.fail(f"Distributed subprocess timed out after {timeout}s:\n{output[-20000:]}")
    assert process.returncode == 0, output[-24000:]
    return output


def torchrun_prefix():
    # Explicit IPv4 avoids platform-specific localhost/IPv6 rendezvous delays.
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        port = endpoint.getsockname()[1]
    return [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
            "--master_addr=127.0.0.1", f"--master_port={port}"]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def assert_nested_close(left, right, *, atol, rtol):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_close(left[key], right[key], atol=atol, rtol=rtol)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_close(a, b, atol=atol, rtol=rtol)
    else:
        assert left == right


def assert_rank_agreement(directory, label):
    first, second = (load(directory / f"{label}_rank{rank}.pt") for rank in (0, 1))
    for name in ("state", "gradients", "optimizers"):
        assert_nested_close(first[name], second[name], atol=0, rtol=0)
    assert first["updates"] == second["updates"]
    assert first["global_step"] == second["global_step"]
    assert first["trace"] == second["trace"]
    assert first["validation"] == second["validation"]
    return first


def assert_equivalent_update(actual, reference):
    for name, value in actual["state"].items():
        expected = reference["state"][name]
        # A constant real residual over all FFT bins is an impulse at the zero
        # analysis-window endpoint. This bias component has zero analytic
        # gradient; FP32 FFT roundoff (~1e-9) can change its first Adam update
        # substantially because Adam's epsilon is 1e-8. Check that null gradient
        # explicitly, while comparing every observable component tightly.
        gradient = actual["gradients"].get(name)
        reference_gradient = reference["gradients"].get(name)
        if (name.endswith("decoder.output.bias") and gradient is not None
                and gradient[0].abs() < 1e-7 and reference_gradient[0].abs() < 1e-7):
            assert abs(float(value[0] - expected[0])) <= 4e-4
            torch.testing.assert_close(value[1:], expected[1:], atol=2e-5, rtol=3e-4)
        else:
            torch.testing.assert_close(value, expected, atol=2e-5, rtol=3e-4)
    assert_nested_close(actual["gradients"], reference["gradients"], atol=5e-7, rtol=3e-4)
    assert_nested_close(actual["optimizers"], reference["optimizers"], atol=2e-7, rtol=3e-4)
    # The unobservable-bias exception must not conceal a changed enhancement.
    models = []
    for record in (actual, reference):
        model = Enhancer(ModelConfig.from_dict(record["portable"]["model_config"])).eval()
        model.load_state_dict(record["portable"]["model_state"], strict=True)
        models.append(model)
    waveform = 0.1 * torch.randn(1, 128, generator=torch.Generator().manual_seed(44))
    with torch.no_grad():
        left, right = (model.enhance(waveform)[1] for model in models)
    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize("stage", ["regression", "adversarial"])
def test_ddp_gradients_match_single_process_equivalent_global_batch(tmp_path, stage):
    distributed, single = tmp_path / "distributed", tmp_path / "single"
    common = [str(WORKER), "--mode", "equivalence", "--stage", stage]
    run_process(torchrun_prefix() + common + ["--output", str(distributed)])
    run_process([sys.executable] + common + ["--output", str(single)], timeout=90)
    actual = assert_rank_agreement(distributed, stage)
    reference = load(single / f"{stage}_rank0.pt")
    assert actual["updates"] == reference["updates"] == 1
    assert actual["global_step"] == (2 if stage == "adversarial" else 1)
    # Compare gradients and Adam moments too: similar first-step Adam parameter
    # changes alone can hide an incorrect reduction or accumulation scale.
    assert_equivalent_update(actual, reference)
    assert any(name.startswith("model.") for name in actual["gradients"])
    if stage == "adversarial":
        assert any(name.startswith("discriminator.") for name in actual["gradients"])


def write_manifests(directory):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(8):
        length = 128 if index < 5 else (99, 115, 128)[index - 5]
        time = np.arange(length, dtype=np.float32) / 800
        clean = (0.1 + index * 0.01) * np.sin(2 * np.pi * (50 + index * 9) * time)
        noisy = clean + np.random.default_rng(index).normal(0, 0.025, length).astype(np.float32)
        sf.write(directory / f"clean{index}.wav", clean, 800, subtype="FLOAT")
        sf.write(directory / f"noisy{index}.wav", noisy, 800, subtype="FLOAT")
        rows.append({"noisy": f"noisy{index}.wav", "clean": f"clean{index}.wav", "sample_rate": 800})
    train, validation = directory / "train.jsonl", directory / "validation.jsonl"
    train.write_text("\n".join(json.dumps(row) for row in rows[:4]) + "\n")
    validation.write_text("\n".join(json.dumps(row) for row in rows[5:]) + "\n")
    validation.with_name("validation-single.jsonl").write_text(json.dumps(rows[5]) + "\n")
    return train, validation


def full_validation_loss(portable, manifest):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = Enhancer(ModelConfig.from_dict(portable["model_config"])).eval()
        model.load_state_dict(portable["model_state"], strict=True)
        dataset = PairedAudioDataset(manifest, 800, crop_samples=128, training=False, load_teacher=False)
        batch = collate_audio([dataset[index] for index in range(len(dataset))])
        criterion = ReconstructionLoss(waveform=1, stft=0.1, fft_sizes=[16, 32])
        with torch.no_grad():
            predictions = model.enhance(batch["noisy"])
            return float(torch.stack([criterion(value, batch["clean"], lengths=batch["lengths"])[0]
                                      for value in predictions.values()]).mean())
    finally:
        torch.set_num_threads(previous_threads)


def test_ddp_sampled_exits_decoder_transition_resume_and_exact_validation(tmp_path):
    train, validation = write_manifests(tmp_path / "data")
    output = tmp_path / "lifecycle"
    run_process(torchrun_prefix() + [str(WORKER), "--mode", "lifecycle", "--output", str(output),
                "--train-manifest", str(train), "--validation-manifest", str(validation)], timeout=240)
    for stage in ("regression", "adversarial"):
        first = assert_rank_agreement(output, f"{stage}_first")
        resumed = assert_rank_agreement(output, f"{stage}_resumed")
        assert first["updates"] == 1 and resumed["updates"] == 3
        factor = 2 if stage == "adversarial" else 1
        assert first["global_step"] == factor and resumed["global_step"] == 3 * factor
        assert first["decoder_mode"] == "shared"
        assert resumed["decoder_mode"] == "per_exit"
        assert {depth for row in first["trace"] + resumed["trace"] for depth in row["exits"]} == {1, 2}
        assert any(row["partial_microbatches"] == 1 for row in resumed["trace"])
        assert resumed["validation"]["validation_mean_clean_loss"] == pytest.approx(
            full_validation_loss(resumed["portable"], validation), rel=3e-5, abs=3e-6)
        checkpoint = load(output / f"{stage}_resumed.ckpt")
        assert checkpoint["step"] == 3 and checkpoint["global_step"] == 3 * factor
        assert len(checkpoint["optimizer_states"]) == factor
        assert checkpoint["world_size"] == 2
        assert len(checkpoint["rng_states_by_rank"]) == 2
    singleton = assert_rank_agreement(output, "validation_single")
    assert singleton["validation"]["validation_mean_clean_loss"] == pytest.approx(
        full_validation_loss(singleton["portable"], validation.with_name("validation-single.jsonl")),
        rel=3e-5, abs=3e-6)


def test_hydra_internal_ddp_and_external_torchrun_gan_entrypoints(tmp_path):
    train, validation = write_manifests(tmp_path / "data")
    common = ["experiment=fixed4", "model.sample_rate=800", "model.depth=1", "model.exits=[1]",
              "model.width=4", "model.state_dim=2", "model.expand=1", "model.conv_kernel=2",
              "model.scan_backend=reference", "model.mixer_impl=legacy", "training.steps=1", "training.batch_size=1",
              "training.crop_seconds=0.16", "training.grad_accum=2", "training.workers=0",
              "training.checkpoint_every=1", "training.validate_every=1", "training.validation_batches=20",
              "training.log_every=1", "loss.fft_sizes=[16,32]", "loss.stft=0.1",
              "adversarial.channels=4", "adversarial.scales=1",
              "trainer.accelerator=cpu", "trainer.devices=2", "trainer.strategy=ddp",
              "trainer.enable_progress_bar=false", f"data.train_manifest={train}",
              f"data.validation_manifest={validation}"]
    regression = tmp_path / "cli_regression"
    run_process([sys.executable, "-m", "realtime_use.train"] + common + [f"output_dir={regression}"], timeout=180)
    checkpoint = regression / "checkpoints" / "last.ckpt"
    assert load(checkpoint)["step"] == 1
    adversarial = tmp_path / "cli_adversarial"
    run_process(torchrun_prefix() + ["-m", "realtime_use.train"] + common + [
        "training.stage=adversarial", f"init_from={checkpoint}", f"output_dir={adversarial}"], timeout=180)
    saved = load(adversarial / "checkpoints" / "last.ckpt")
    assert saved["step"] == 1 and saved["global_step"] == 2
    assert len(saved["optimizer_states"]) == 2
    for directory in (regression, adversarial):
        assert (directory / "checkpoints" / "best.ckpt").is_file()
        assert len(list((directory / "metrics").glob("version_*/metrics.csv"))) == 1
