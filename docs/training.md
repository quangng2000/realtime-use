# Training with Lightning and Hydra

Training uses a LightningModule, an aligned-audio LightningDataModule, and Hydra configuration composition. The model, losses, and method limitations described in [method.md](method.md) are unchanged by this orchestration choice.

## Configuration groups

Canonical configs are packaged under `src/realtime_use/conf`, so the installed command can compose them outside the repository root. The groups are:

- `model`: audio framing, depth, exits, dimensions, decoder mode, and scan backend.
- `data`: training/validation manifests and data-loading options.
- `training`: stage, generator-update budget, batch/crop/worker settings, optimizer settings, accumulation, and decoder transition.
- `loss`: supervised reconstruction and optional cached-output distillation.
- `adversarial`: discriminator size and adversarial/feature-matching weights.
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

The default Hydra launcher runs these jobs sequentially. A multirun is a set of independent training jobs, not distributed training of one model. Do not override all jobs to share one output directory. The current implementation supports one CPU or one GPU device and rejects distributed execution; DDP and multi-GPU scaling remain future work.

## Update counts and accumulation

`training.steps` always means **generator updates**. `training.grad_accum` is the number of microbatches accumulated for each update. The module uses manual optimization so the regression and adversarial stages share that definition.

In regression, one completed generator update corresponds to one optimizer step. In the adversarial stage, each update includes one discriminator step and one generator step. Lightning therefore counts two optimizer steps for each completed generator update. The driver derives `trainer.max_steps` from the stage and requested budget; use `training.steps` to set the budget rather than overriding the derived limit.

Use the logged `completed_updates` or the portable checkpoint's `step` field when comparing stages. Do not compare raw Lightning `global_step` as if it had the same meaning in both stages. For timing estimates, measure a complete generator update, including its accumulated microbatches and any discriminator work.

Use `training.grad_accum` rather than Lightning's automatic `accumulate_grad_batches` setting. Checkpointing and validation must occur at completed-update boundaries so that a checkpoint does not silently discard a partially accumulated update.

## Checkpoints and stage transitions

The run writes native Lightning checkpoints under `checkpoints/`:

- `last.ckpt`: the latest saved training state.
- `step_*.ckpt`: periodic snapshots.
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

`smoke=true` substitutes a one-block, width-4 model and tiny generated data. It caps the generator-update budget at three; set `training.steps=1` or `training.steps=2` for a shorter check. Synthetic validation checks the execution path; it is not a speech-quality result. A sweep of smoke jobs tests configuration and lifecycle handling, not the actual 4-/8-/12-block training cost.

CPU tests do not validate CUDA kernels, GPU precision modes, full-size training memory, distributed execution, or production latency. Measure those independently before allocating a longer run.
