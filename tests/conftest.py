"""Opt-in test mode that permits gradients but prevents training execution."""

from __future__ import annotations

import shlex
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption("--no-training", action="store_true", default=False,
                     help="Skip training_loop tests and block Trainer.fit/optimizer steps")


def _forbid_training(*args, **kwargs):
    pytest.fail("Training execution is forbidden by --no-training; mark this test training_loop", pytrace=True)


def _install_training_guards():
    """Install test-process guards before collection, without changing production."""
    from lightning.pytorch import Trainer
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    patch = pytest.MonkeyPatch()
    patch.setattr(Trainer, "fit", _forbid_training)
    # The global hook covers concrete optimizer subclasses, including AdamW,
    # instead of patching only the unused Optimizer.step base implementation.
    hook = register_optimizer_step_pre_hook(_forbid_training)
    original_popen = subprocess.Popen
    class GuardedPopen(original_popen):
        def __init__(self, args, *positional, **kwargs):
            # Known child training tests are marked individually. This extra
            # check prevents an accidentally unmarked CLI/worker launch; pure
            # Hydra config rendering remains available without a fit loop.
            command = shlex.split(args) if isinstance(args, str) else [str(arg) for arg in args]
            text = " ".join(command)
            launcher = any(token in text for token in (
                "realtime_use.train", "realtime-use-train", "ddp_worker.py",
                "--ddp-worker", "torch.distributed.run", "torchrun"))
            render_only = ("realtime_use.train" in command and "--cfg" in command
                           and not kwargs.get("shell"))
            if launcher and not render_only:
                pytest.fail("Training subprocess blocked by --no-training; mark this test training_loop")
            super().__init__(args, *positional, **kwargs)

    patch.setattr(subprocess, "Popen", GuardedPopen)
    return patch, hook


def pytest_configure(config):
    if config.getoption("--no-training"):
        config._no_training_guards = _install_training_guards()


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-training"):
        skip = pytest.mark.skip(reason="--no-training excludes optimizer updates and training loops")
        for item in items:
            if item.get_closest_marker("training_loop") is not None:
                item.add_marker(skip)


def pytest_report_header(config):
    if config.getoption("--no-training"):
        return "No-training mode: training_loop cases skipped; Trainer.fit, optimizer steps and child launchers guarded"


def pytest_unconfigure(config):
    if guards := getattr(config, "_no_training_guards", None):
        patch, hook = guards
        hook.remove()
        patch.undo()
