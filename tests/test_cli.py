"""Exercise Hydra composition, real Lightning checkpoints, and command-line sweeps."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_module
import numpy as np
from omegaconf import OmegaConf
import pytest
import soundfile as sf
import torch

from realtime_use.export import export_exit
from realtime_use.infer import load_model
from realtime_use.train import prepare_config, run_training


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config_for(path, experiment="fixed4"):
    with initialize_config_module(version_base="1.3", config_module="realtime_use.conf"):
        return compose(config_name="config", overrides=[f"experiment={experiment}", f"output_dir={path}"])


@pytest.mark.parametrize("experiment,depth,exits", [("fixed4",4,[4]),("fixed8",8,[8]),
    ("fixed12",12,[12]),("flexible12",12,[4,8,12])])
def test_hydra_experiment_composition(tmp_path, experiment, depth, exits):
    cfg = config_for(tmp_path, experiment)
    assert cfg.model.depth == depth
    assert list(cfg.model.exits) == exits
    assert cfg.model.mixer_impl == "mamba_ssm"
    assert cfg.model.codec_impl == "nemotron_causal"
    assert cfg.training.split_decoders_at == (50000 if experiment == "flexible12" else None)


def tiny_paired_config(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    cfg = config_for(tmp_path / "regression", "flexible12")
    cfg.model.update(sample_rate=800, depth=2, exits=[1,2], width=4, state_dim=2,
                     expand=1, conv_kernel=2, scan_backend="reference", mixer_impl="legacy", codec_impl="legacy")
    cfg.training.update(steps=2, batch_size=2, crop_seconds=0.16, workers=0,
        grad_accum=2, checkpoint_every=1, validate_every=1, validation_batches=1,
        log_every=1, split_decoders_at=1)
    cfg.loss.update(fft_sizes=[16,32], kd=0.5)
    cfg.adversarial.update(channels=4, scales=1, implementation="legacy")
    cfg.trainer.update(enable_progress_bar=False)
    rows=[]
    for index, length in enumerate([99,128]):
        x=np.arange(length,dtype=np.float32)/800
        clean=0.2*np.sin(2*np.pi*120*x)
        noisy=clean+0.02*np.random.default_rng(index).normal(size=length).astype(np.float32)
        sf.write(tmp_path/f"clean{index}.wav",clean,800,subtype="FLOAT")
        sf.write(tmp_path/f"noisy{index}.wav",noisy,800,subtype="FLOAT")
        rows.append({"noisy":f"noisy{index}.wav","clean":f"clean{index}.wav",
                     "teacher":f"clean{index}.wav","sample_rate":800})
    manifest=tmp_path/"pairs.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    cfg.data.train_manifest=str(manifest)
    cfg.data.validation_manifest=str(manifest)
    return cfg,manifest


@pytest.mark.training_loop
def test_native_kd_split_gan_resume_and_inference_export(tmp_path):
    cfg,manifest=tiny_paired_config(tmp_path)
    first=run_training(cfg)
    saved=torch.load(first,weights_only=True)
    assert saved["step"] == 2 and saved["global_step"] == 2
    assert saved["model_config"]["decoder_mode"] == "per_exit"
    assert (first.parent/"best.ckpt").is_file()
    assert "optimizer_states" in saved and len(saved["optimizer_states"]) == 1
    assert not any(key.startswith("shared_decoder.") for key in saved["model_state"])
    assert any(key.startswith("model.shared_decoder.") for key in saved["state_dict"])
    inference=load_model(first)
    assert inference.config.exits == (1,2)
    exported=export_exit(first,1,tmp_path/"export.pt")
    assert load_model(exported).config.depth == 1
    # A cached teacher path may remain in the manifest without being accessed by GAN.
    rows=[json.loads(line) for line in manifest.read_text().splitlines()]
    for row in rows:
        row["teacher"]="missing-teacher.wav"
    manifest.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    gan=copy.deepcopy(cfg)
    gan.training.stage="adversarial"
    gan.training.steps=1
    gan.init_from=str(first)
    gan.output_dir=str(tmp_path/"gan")
    second=run_training(gan)
    checkpoint=torch.load(second,weights_only=True)
    assert checkpoint["step"] == 1 and checkpoint["global_step"] == 2
    assert len(checkpoint["optimizer_states"]) == 2
    assert checkpoint["optimizer_states"][1]["state"]
    resumed=copy.deepcopy(gan)
    resumed.init_from=None
    resumed.resume=str(second)
    resumed.training.steps=2
    resumed.output_dir=str(tmp_path/"continued")
    last=run_training(resumed)
    state=torch.load(last,weights_only=True)
    assert state["step"] == 2 and state["global_step"] == 4
    assert state["model_config"]["decoder_mode"] == "per_exit"
    bad=copy.deepcopy(resumed)
    bad.training.stage="regression"
    with pytest.raises(ValueError,match="preserve stage"):
        run_training(bad)


def test_official_run_rejects_legacy_weights_before_creating_output(tmp_path):
    cfg, _ = tiny_paired_config(tmp_path)
    old_model_config = OmegaConf.to_container(cfg.model, resolve=True)
    old_model_config.pop("mixer_impl")
    source = tmp_path / "legacy.pt"
    torch.save({"model_config": old_model_config, "model_state": {},
                "stage": "regression", "step": 0}, source)
    cfg.model.mixer_impl = "mamba_ssm"
    cfg.model.scan_backend = "auto"
    cfg.init_from = str(source)
    with pytest.raises(ValueError, match="architecture differs at mixer_impl"):
        run_training(cfg)
    assert not Path(cfg.output_dir).exists()


@pytest.mark.training_loop
def test_legacy_weights_initialize_but_cannot_native_resume(tmp_path):
    cfg,_=tiny_paired_config(tmp_path)
    cfg.training.steps=1
    native=run_training(cfg)
    checkpoint=torch.load(native,weights_only=True)
    legacy={key:checkpoint[key] for key in ("model_config","model_state","stage","step")}
    path=tmp_path/"legacy.pt"
    torch.save(legacy,path)
    cfg.output_dir=str(tmp_path/"initialized")
    cfg.init_from=str(path)
    result=run_training(cfg)
    assert torch.load(result,weights_only=True)["step"] == 1
    cfg.init_from=None
    cfg.resume=str(path)
    with pytest.raises(ValueError,match="native Lightning"):
        run_training(cfg)


def test_entrypoint_accepts_ddp_and_rejects_unsafe_strategy_and_missing_data(tmp_path):
    cfg=config_for(tmp_path)
    cfg.trainer.devices=2
    assert prepare_config(cfg)["trainer"]["devices"] == 2
    cfg.trainer.strategy="ddp_find_unused_parameters_false"
    with pytest.raises(ValueError,match="unused-parameter"):
        prepare_config(cfg)
    cfg.trainer.devices=1
    cfg.trainer.strategy="auto"
    with pytest.raises(ValueError,match="train_manifest"):
        run_training(cfg)


def test_hydra_cli_render_without_training(tmp_path):
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    render=subprocess.run([sys.executable,"-m","realtime_use.train","experiment=flexible12",
        "--cfg","job","--resolve"],capture_output=True,text=True,env=env,timeout=60)
    assert render.returncode == 0, render.stdout+render.stderr
    rendered=OmegaConf.create(render.stdout)
    assert list(rendered.model.exits)==[4,8,12]


@pytest.mark.training_loop
def test_hydra_two_job_smoke_sweep(tmp_path):
    env=dict(os.environ,OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    sweep=subprocess.run([sys.executable,"-m","realtime_use.train","-m",
        "experiment=fixed4,fixed8","smoke=true","training.steps=1",
        "trainer.enable_progress_bar=false",f"hydra.sweep.dir={tmp_path}/sweep"],
        capture_output=True,text=True,env=env,timeout=90)
    assert sweep.returncode == 0,sweep.stdout+sweep.stderr
    for number in (0,1):
        checkpoint=tmp_path/"sweep"/str(number)/"checkpoints"/"last.ckpt"
        assert torch.load(checkpoint,weights_only=True)["step"] == 1
        assert (checkpoint.parent.parent/".hydra"/"config.yaml").exists()
