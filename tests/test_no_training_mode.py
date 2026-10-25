"""Verify the guards abort before a fit loop or parameter update can execute."""

import subprocess
import sys

from lightning.pytorch import Trainer
import pytest
import torch


@pytest.fixture(autouse=True)
def require_no_training_mode(request):
    if not request.config.getoption("--no-training"):
        pytest.skip("Guard checks apply only when --no-training is enabled")


@pytest.mark.parametrize("optimizer_class", [torch.optim.SGD, torch.optim.AdamW])
def test_optimizer_guard_blocks_before_parameter_or_state_changes(optimizer_class):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = optimizer_class([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(pytest.fail.Exception, match="Training execution is forbidden"):
        optimizer.step()
    assert parameter.item() == 1.0
    assert not optimizer.state


def test_fit_guard_blocks_before_trainer_or_model_setup():
    with pytest.raises(pytest.fail.Exception, match="Training execution is forbidden"):
        Trainer.fit(None)


def test_unreviewed_worker_launcher_is_blocked_before_launch():
    with pytest.raises(pytest.fail.Exception, match="Training subprocess blocked"):
        subprocess.run([sys.executable, "tests/ddp_worker.py"])
