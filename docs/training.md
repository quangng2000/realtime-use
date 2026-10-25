# Training with Lightning and Hydra

Training uses a LightningModule, an aligned-audio LightningDataModule, and Hydra configuration composition. The model, losses, and method limitations described in [method.md](method.md) are unchanged by this orchestration choice.

## Library responsibilities and project-specific code

Hydra composes configurations and launches independent sweeps. Lightning owns process setup, optimizer and loop restoration, precision handling, CSV logging, and checkpoint selection and serialization. Two native `ModelCheckpoint` callbacks manage periodic snapshots and the lowest validation loss; the project does not maintain a separate best-score tracker.

New model presets use official Mamba-1 and the licensed NVIDIA dense encoder/magnitude/phase decoder components adapted for causality. GAN presets import DAC's spectral discriminator. PyTorch supplies STFT/iSTFT and waveform L1, `auraloss` supplies multi-resolution spectral losses, TorchMetrics supplies validation means and signal-quality formulas, and optional `audiomentations` transforms supply training corruptions. Project adapters handle causal state, valid utterance lengths, paired-data semantics, and exit selection. Legacy components remain available for existing checkpoints; they are not the default Hydra training recipe.

`loss.implementation=auraloss` uses centered Hann STFTs with hop sizes one quarter of each configured FFT size, spectral-convergence and log-magnitude weights of one, and no phase/perceptual term. Valid utterance prefixes are evaluated separately so batch composition and DDP sharding do not alter their relative weights. Very short prefixes receive deterministic zero extension for the library's reflection padding. This objective differs from the previous uncentered custom loss. Resume must preserve the checkpoint's `model.mixer_impl`, `model.codec_impl`, `loss.implementation` and `adversarial.implementation`; missing identity fields mean legacy. Runs from before the library migration use all legacy components, while runs from the intermediate Mamba migration may use official Mamba with the legacy codec. Inspect the saved configuration rather than switching all fields blindly. `init_from` can deliberately start a new objective/critic with compatible enhancement architecture and fresh optimizer state; it cannot convert codec or mixer weights.

The remaining checkpoint adapter checks that manual gradient accumulation has finished before delegating a periodic save. A small run callback writes the resolved configuration, rejects accidental reuse of an existing run, signals the generator-update budget, and refreshes `last.ckpt` on every rank before distributed teardown. Signaling the budget also lets Lightning run any final validation inside `fit`, including when the budget is not a validation-interval multiple.

Model-specific code still defines the causal enhancement network, losses, two-optimizer GAN update, decoder transition, synchronized exit sampling, and valid-example accounting for distributed validation. These are project implementation choices; using Lightning does not establish equivalence to the papers or add missing data augmentation.

## Official Mamba CUDA environment

The CUDA extra pins the complete upstream Mamba-1 implementation to `mamba-ssm==2.2.6.post3` and its causal convolution extension to `causal-conv1d==1.6.0`. The adapter calls the upstream [Mamba class](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/mamba_ssm/modules/mamba_simple.py), including its initialization, sequence forward pass, inference cache, and token step. This dependency is separate from NVIDIA's RE-USE implementation and pretrained weights. Mamba carries the [Apache-2.0 license](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/LICENSE); causal-conv1d carries the [BSD-3-Clause license](https://github.com/Dao-AILab/causal-conv1d/blob/v1.6.0/LICENSE).

The following is a compatible installation candidate for **Linux x86_64, Python 3.11, CUDA 12.8, and PyTorch 2.7.1**. It has not been installed or runtime-validated on a CUDA machine by this project. The [Mamba release](https://github.com/state-spaces/mamba/releases/tag/v2.2.6.post3) and [causal-conv1d release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.6.0) publish matching Python 3.11 / PyTorch 2.7 / CUDA 12 wheels. The PyTorch command follows its [versioned installation instructions](https://pytorch.org/get-started/previous-versions/#v271).

Start in the repository root with an NVIDIA driver compatible with CUDA 12.8 and a CUDA 12.8 development toolkit, including `nvcc`, available. A CUDA runtime-only image may lack the compiler needed by extension metadata or a source-build fallback.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
nvcc --version
python -m pip install --no-build-isolation -e ".[cuda,gan,test,augmentation]"
python -m pip check
```

Install PyTorch before the extensions and retain `--no-build-isolation`, so their build uses that installed CUDA-enabled PyTorch. These commands target the pinned releases and their [release build setup](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/setup.py); newer upstream `main` installation flags can differ. The Mamba-1 pin avoids adding the unrelated Mamba-3 dependency stack. It is not a claim that this is the newest Mamba package.

For the complete Linux CPU test suite, first install matching CPU wheels (`python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu`), then `python -m pip install -e ".[test,augmentation,gan]"`. CI uses this combination. The author codec and actual DAC tests execute on CPU with explicitly selected reference mixers; official Mamba numerical tests require CUDA. Do not install the CUDA extra for CPU smoke tests. Smoke selects the legacy codec, reference mixer and legacy critic; it does not exercise the new architecture.

The `gan` extra pins `descript-audio-codec==1.0.0` and `descript-audiotools==0.7.2`. Install torchaudio alongside the matching PyTorch build: dependency resolution alone does not guarantee matching extension binaries. The adapter uses DAC's `MRD` directly, avoiding the enclosing wrapper's periodic critics and amplitude normalization. `adversarial.fft_sizes` and `adversarial.band_width_hz` select STFT scales and approximate band width; `channels` and `scales` apply only to the legacy critic.

Official Mamba checkpoints require `--device cuda` for inference, evaluation, benchmarking, and exit export. Loading one on CPU raises an error rather than changing its architecture. Export retains the chosen official Mamba blocks and verifies full/stream outputs on the requested CUDA device. Legacy checkpoints keep their original parameter layout and CPU reference support; they cannot initialize the official architecture through a backend switch.

Run `python -m pytest -q tests/test_official_mamba.py` on that CUDA host before a training job. These tests exercise the actual class, gradients, streaming state, exit export, and regression/GAN checkpoint continuation. Without CUDA, numerical tests are skipped; on a CUDA host, missing or broken Mamba dependencies fail instead of passing through a mock.

## Configuration groups

Canonical configs are packaged under `src/realtime_use/conf`, so the installed command can compose them outside the repository root. The groups are:

- `model`: audio framing, depth, exits, dimensions, decoder mode, codec/mixer implementations, and legacy scan backend.
- `data`: training/validation manifests and data-loading options.
- `training`: stage, generator-update budget, batch/crop/worker settings, optimizer settings, accumulation, and decoder transition.
- `loss`: supervised reconstruction and optional cached-output distillation.
- `adversarial`: critic implementation, STFT scales/bands (or legacy size), and adversarial/feature-matching weights.
- `trainer`: Lightning accelerator, devices, precision, and runtime options.
- `experiment`: the composed `fixed4`, `fixed8`, `fixed12`, or `flexible12` starting recipe.

Inspect a fully resolved configuration without training:

```bash
python -m realtime_use.train experiment=fixed4 --cfg job --resolve
```

Override individual values with Hydra's `key=value` syntax:

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  training.steps=10000 \
  training.batch_size=2 training.crop_seconds=2.0 \
  training.workers=2 \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/fixed4-pilot
```

The training driver resolves manifest paths against the launch directory. Audio paths inside a manifest remain relative to that manifest. Use absolute manifest/checkpoint paths when scripting sweeps or running from another directory.

## Runs and sweeps

Without an explicit `output_dir`, the run uses Hydra's runtime output directory. The configured run/sweep directories are under `runs/`; sweep jobs receive separate numbered subdirectories. Hydra records the composed config and overrides. Lightning CSV metrics are written below the run's `metrics/version_*` directory.

To train all fixed baselines with one command:

```bash
python -m realtime_use.train -m \
  experiment=fixed4,fixed8,fixed12 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  trainer.accelerator=gpu trainer.devices=1
```

The default Hydra launcher runs these jobs sequentially. A multirun is a set of independent training jobs, not distributed training of one model. Do not override all jobs to share one output directory. Hydra `-m` is supported for single-device jobs; combining it with DDP is currently rejected because Lightning's child-process re-execution can repeat the sweep.

## Distributed training

Both regression and adversarial stages support PyTorch DDP. Lightning's internal launcher can start one process per GPU:

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  trainer.accelerator=gpu trainer.devices=4 trainer.strategy=ddp \
  training.batch_size=2 training.grad_accum=4 \
  output_dir=/absolute/path/to/runs/fixed4-ddp
```

Alternatively, launch the same job with external `torchrun`. Supply one shared, explicit `output_dir` so independently started Hydra processes agree on the run destination:

```bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
  --master_addr=127.0.0.1 --master_port=29500 \
  -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  trainer.accelerator=gpu trainer.devices=4 trainer.strategy=ddp \
  training.batch_size=2 training.grad_accum=4 \
  output_dir=/absolute/path/to/runs/fixed4-torchrun
```

Choose another unused master port when running concurrent local jobs. The explicit IPv4 address also avoids localhost/IPv6 resolver delays on some CPU development hosts.

The generator-update batch size is `world_size × training.batch_size × training.grad_accum` for full microbatches. `training.batch_size` is per rank. Partial batches can reduce the actual number of examples, and the training distributed sampler may repeat a few rows to give ranks equal work. Increasing GPU count does not automatically justify a proportional learning-rate change.

All ranks use the same sampled exit for each training microbatch. Decoder heads are registered before DDP setup, and the shared-to-per-exit transition copies weights without replacing registered parameters. The strategy enables unused-parameter handling and disables static-graph assumptions because active exits and GAN optimizer phases change which parameters receive gradients.

Validation pads rank workloads where necessary but marks padding rows invalid. Global loss is computed from summed valid-example losses and counts, so a complete pass counts every original validation example once, including when the dataset size is not divisible by world size. `training.validation_batches` limits batches **per rank**; a capped pass may cover only a subset. Use a sufficient limit for a full validation-set result.

Checkpoint participation is collective, while checkpoint files and run logs are written by rank zero. Native resume preserves each rank's main-process Python, NumPy, and PyTorch random states and requires the same world size. Worker prefetch/crop order is not replayed exactly; changing world size is a fresh initialization experiment. Keep architecture, optimizer structure, accumulation, and stage compatible with the checkpoint.

For a CPU-only execution check, use `trainer.accelerator=cpu trainer.devices=2 trainer.strategy=ddp` with `smoke=true` and a unique output directory. The smoke model remains tiny and uses the legacy reference mixer. The integration tests also compare actual gradients and Adam moments against an equivalent single-process batch for both regression and GAN; merely completing without a hang would not establish correct gradient reduction.

CUDA/NCCL and multi-node execution have not been exercised by the CPU/Gloo integration tests. Measure GPU memory, mixed-precision behavior, throughput, and scaling separately before a longer run. Launch separate DDP jobs instead of a distributed Hydra multirun.

## Update counts and accumulation

`training.steps` always means **generator updates**. `training.grad_accum` is the number of microbatches accumulated for each update. The module uses manual optimization so the regression and adversarial stages share that definition.

In regression, one completed generator update corresponds to one optimizer step. In the adversarial stage, each update includes one discriminator step and one generator step. Lightning therefore counts two optimizer steps for each completed generator update. The driver derives `trainer.max_steps` from the stage and requested budget; use `training.steps` to set the budget rather than overriding the derived limit.

Use the logged `completed_updates` or the portable checkpoint's `step` field when comparing stages. Do not compare raw Lightning `global_step` as if it had the same meaning in both stages. For timing estimates, measure a complete generator update, including its accumulated microbatches and any discriminator work.

Use `training.grad_accum` rather than Lightning's automatic `accumulate_grad_batches` setting. Checkpointing and validation must occur at completed-update boundaries so that a checkpoint does not silently discard a partially accumulated update.

## Checkpoints and stage transitions

The run writes native Lightning checkpoints under `checkpoints/`:

- `last.ckpt`: the latest saved training state, refreshed at normal training completion.
- `step_*.ckpt`: periodic snapshots named by completed generator updates.
- `best.ckpt`: selected by `validation_mean_clean_loss` when validation is configured.

The native checkpoint also includes `model_config`, `model_state`, `stage`, and `step` compatibility fields. Existing `infer`, `evaluate`, `benchmark`, and `export` commands can read the enhancement weights without constructing the training loop.

Continue a compatible interrupted Lightning run using `resume`:

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  resume=/absolute/path/to/runs/fixed4-regression/checkpoints/last.ckpt \
  training.steps=20000 \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/fixed4-regression-continued
```

The budget is the total desired generator-update count, including completed updates. Lightning restores optimizer/loop state for a compatible run; exact data-order continuation is not guaranteed.

Native checkpoint-selection history is restored when resuming into the same checkpoint directory with the same callback configuration. Lightning resets the score ranking when the checkpoint directory changes, so the example above selects a new best within the continued run. For checkpoints from the earlier custom `TrainingCheckpoints` callback, a small migration imports the saved best score into Lightning only when resuming from that same directory and its `best.ckpt` still exists. Otherwise selection starts a new history; compatible model, optimizer, and loop state can still resume. Keep the earlier `best.ckpt` when comparing across a directory change.

Start a new adversarial stage with `init_from` pointing to a regression checkpoint:

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  training.stage=adversarial \
  init_from=/absolute/path/to/runs/fixed4-regression/checkpoints/last.ckpt \
  training.steps=10000 \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/fixed4-adversarial
```

Initialization transfers enhancement weights while starting new optimizers and a new update budget. It does not continue the previous optimizer state. The enhancement network remains trainable, and no teacher is loaded for the adversarial stage.

For flexible training, the decoder transition preserves the learned shared decoder by copying it into the matching exit heads and adjusts the preceding modules' learning rate. This is separate from changing the training loss stage. Checkpoint state must preserve which transition has occurred.

## Migration from the earlier training CLI

The former argparse training flags are not preserved. Update scripts as follows:

- `--config configs/fixed4.yaml` becomes `experiment=fixed4`.
- `--manifest PATH` becomes `data.train_manifest=PATH`.
- `--validation-manifest PATH` becomes `data.validation_manifest=PATH`.
- `--output DIR` becomes `output_dir=DIR`.
- `--stage adversarial` becomes `training.stage=adversarial`.
- `--device cuda` becomes `trainer.accelerator=gpu trainer.devices=1`.
- `--device cpu` becomes `trainer.accelerator=cpu trainer.devices=1`.
- `--steps N` becomes `training.steps=N`.
- `--smoke` becomes `smoke=true`.
- `--init-from PATH` becomes `init_from=PATH`.
- `--resume PATH` becomes `resume=PATH` for a compatible native Lightning checkpoint.

Earlier portable `.pt` checkpoints can initialize compatible enhancement weights through `init_from`; they do not contain native Lightning loop state and cannot provide a full Lightning resume. Retain them for inference/export or fresh-stage initialization. New training checkpoints use `.ckpt`; exported inference-only checkpoints may still use `.pt`.

Inference, evaluation, export, and benchmark commands retain their existing flag-based interfaces. Changing the training runner does not require changing those commands beyond selecting the new checkpoint path.

## Smoke and validation scope

`smoke=true` substitutes a one-block, width-4 legacy reference model and tiny generated data. It caps the generator-update budget at three; set `training.steps=1` or `training.steps=2` for a shorter check. Synthetic validation checks the execution path; it is not a speech-quality result or validation of official Mamba CUDA execution. A sweep of smoke jobs tests configuration and lifecycle handling, not the actual 4-/8-/12-block training cost.

CPU/Gloo tests do not validate CUDA kernels, NCCL, GPU precision modes, full-size training memory, multi-node execution, or production latency. Measure those independently before allocating a longer run.
