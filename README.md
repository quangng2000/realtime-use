# realtime-use

An independent PyTorch research implementation for small, causal speech enhancement models with **4, 8, or 12 time–frequency blocks**, trained with **PyTorch Lightning and Hydra**. The first target is mono 16 kHz audio with a 40 ms analysis window, a 20 ms hop, and zero look-ahead.

The project uses the methods in [One Model, Many Latencies](https://arxiv.org/abs/2606.25621) and [Rethinking Training Targets, Architectures and Data Quality](https://arxiv.org/abs/2603.02641) as research references. It does not download or include NVIDIA checkpoints or restricted model source code. Models start from random initialization unless you explicitly supply your own compatible checkpoint.

**Status:** a training and streaming baseline for experiments. It is not an exact reproduction of either paper, a pretrained enhancement product, or evidence of published quality or latency. The complete URGENT degradation pipeline, sample-rate-independent training, and adjustable look-ahead remain future work. The initial Python reference scan prioritizes correctness; deployment speed must be measured on the intended hardware.

## Set up

Python 3.11 or 3.12 is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

For a CPU-only PyTorch install, install the appropriate wheel first:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

The CPU reference backend needs neither a paid GPU nor a pretrained model. The CI workflow is configured to test that path on Ubuntu with Python 3.11 and 3.12.

For accelerated training, first configure a compatible CUDA PyTorch and build environment, then install the optional scan dependencies:

```bash
python -m pip install -e '.[cuda]'
```

Pass `model.scan_backend=mamba_ssm` to require the CUDA scan kernel and fail clearly if it is unavailable. The supplied `auto` setting can fall back to the much slower reference implementation. Initial verification is CPU-only: CUDA kernel compatibility, mixed-precision behavior, GPU training throughput, and GPU latency still need measurements on the actual environment.

The trainer supports one device or PyTorch DDP for regression and adversarial training. The integration suite exercises two CPU/Gloo processes, gradient accumulation, decoder transitions, checkpoint resume, and comparison with an equivalent single-process batch. CUDA/NCCL correctness, memory use, and scaling still require GPU validation.

## Prepare your data

Training requires aligned noisy/clean pairs that you are permitted to use. No dataset is downloaded automatically. See [the data contract](data/README.md) before constructing a manifest and record each dataset's terms in [the provenance ledger](docs/provenance.md).

Each JSONL row has paths relative to the manifest and an explicit sample rate:

```json
{"noisy":"noisy/example.wav","clean":"clean/example.wav","sample_rate":16000}
```

The two recordings must be mono, have the same sample rate and length, and already be aligned. The initial dataset loader rejects other rates instead of silently resampling. Keep validation speakers and recordings separate from training.

## Run a smoke test

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  smoke=true training.steps=2 \
  trainer.accelerator=cpu trainer.devices=1 \
  output_dir=runs/smoke4
```

Smoke mode overrides the supplied architecture with **one block at width 4** and runs at most three tiny synthetic steps. Its data contains tones and noise. A successful run checks the software path only; it cannot establish speech enhancement quality or the memory/runtime requirements of the real 4-, 8-, or 12-block configurations. Benchmark those configurations separately on the intended GPU.

## Train fixed-depth baselines

Start by training each fixed depth independently. This isolates the quality and compute cost of the 4-, 8-, and 12-block choices.

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  training.stage=regression \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/fixed4-regression
```

Use `experiment=fixed8` or `experiment=fixed12` and a separate output directory for the other baselines. Set the training budget, crop duration, batch size, and validation plan for your dataset; the supplied settings are starting configurations, not the authors' recovered training schedule.

Hydra composes the `model`, `data`, `training`, `loss`, `adversarial`, `trainer`, and `experiment` groups. Inspect the resolved job before training:

```bash
python -m realtime_use.train experiment=fixed4 --cfg job --resolve
```

Run the three fixed baselines as a Hydra sweep:

```bash
python -m realtime_use.train -m \
  experiment=fixed4,fixed8,fixed12 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  trainer.accelerator=gpu trainer.devices=1
```

Use isolated output directories for each job. A sweep starts multiple training runs; it does not split one model across GPUs. See [training configuration and migration](docs/training.md) for overrides, checkpoints, and update-count semantics.

## Train on several GPUs

For one four-GPU DDP job, use an explicit shared output directory:

```bash
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  data.validation_manifest=/absolute/path/to/validation.jsonl \
  trainer.accelerator=gpu trainer.devices=4 trainer.strategy=ddp \
  training.batch_size=2 training.grad_accum=4 \
  output_dir=/absolute/path/to/runs/fixed4-ddp
```

Batch size is per rank: this example accumulates **4 ranks × 2 examples × 4 microbatches = 32 examples per generator update** when all microbatches are full. The same DDP settings apply to adversarial fine-tuning with `training.stage=adversarial` and `init_from` pointing to a regression checkpoint.

Hydra `-m` with a distributed job is currently rejected; launch each DDP experiment separately. External `torchrun`, validation sharding, and resume requirements are documented in [training.md](docs/training.md#distributed-training).

The regression loss combines waveform L1 with multi-resolution spectral convergence and log-magnitude losses. These are independently chosen objectives. Their definitions live in `src/realtime_use/losses.py`; they should not be described as a verified copy of the papers' loss implementation.

The trainer writes a native Lightning checkpoint at `checkpoints/last.ckpt` inside the run's output directory. With a validation manifest, it can select `checkpoints/best.ckpt` using held-out reconstruction loss averaged over exits. Validation uses fixed crops and the configured validation limit; this selection metric is not UTMOS, a listening-test result, or a full official benchmark.

## Add adversarial fine-tuning

First obtain and evaluate a regression checkpoint. Then initialize adversarial training from that checkpoint:

```bash
REGRESSION_CHECKPOINT=/absolute/path/to/runs/fixed4-regression/checkpoints/last.ckpt
python -m realtime_use.train \
  experiment=fixed4 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  training.stage=adversarial \
  "init_from=$REGRESSION_CHECKPOINT" \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/fixed4-adversarial
```

Here the enhancement network remains trainable, with hinge-GAN and feature-matching losses in addition to clean-reference reconstruction. Inference still uses one enhancement network. Teacher tensors are ignored in this stage. Discriminators and the precise adversarial objective are our implementation choices.

This follows the real-time paper's broad regression-then-adversarial approach. The offline RE-USE paper instead freezes a regression network and trains a separate residual generator; that two-network inference design is not implemented here. See [method boundaries](docs/method.md).

Use `resume=/absolute/path/to/checkpoint.ckpt` to continue a compatible Lightning run, including optimizer and loop state. Exact data-order continuation is not guaranteed. `training.steps` is the total target number of generator updates, including prior updates on resume. Use `init_from=/absolute/path/to/checkpoint.ckpt` to initialize a new stage with a fresh optimizer and update count. Do not treat these as interchangeable operations.

## Train one model with several exits

After the fixed baselines are understood, `experiment=flexible12` enables a depth-12 model with exits at 4, 8, and 12. Shared-decoder training and a later shared-to-per-exit decoder transition are separate from the regression-to-adversarial loss transition.

```bash
python -m realtime_use.train \
  experiment=flexible12 \
  data.train_manifest=/absolute/path/to/train.jsonl \
  training.stage=regression \
  trainer.accelerator=gpu trainer.devices=1 \
  output_dir=runs/flexible12-regression
```

`training.sample_exit` controls exit sampling. `training.split_decoders_at` schedules cloning the learned shared decoder into per-exit decoders; `training.predecoder_lr_scale` controls the preceding modules' learning-rate scale at that transition. Choose these settings explicitly for the experiment. The complete authors' schedule, and its precise ordering relative to adversarial fine-tuning, are not available in the reviewed materials.

Knowledge distillation during regression training is optional and disabled initially. If enabled later, it uses explicitly supplied cached teacher outputs with documented permissions; neither training stage fetches or runs a teacher automatically.

## Export a small inference checkpoint

After training a flexible model, extract one of its trained exits:

```bash
python -m realtime_use.export \
  --checkpoint runs/flexible12-regression/checkpoints/last.ckpt \
  --exit-depth 4 \
  --output runs/export/exit4.pt
```

This retains the encoder, first four blocks, and the matching trained decoder. It removes the deeper blocks, other decoder heads, and training optimizer/discriminator state. The resulting inference checkpoint can be used by the inference and benchmark commands below. Selecting an exit at runtime alone does not remove the unused weights from a full model's checkpoint.

Export verifies full/streaming output equivalence on a short deterministic CPU sample and refuses to overwrite an existing destination. This is a structural correctness check, not a quality or speed benchmark. Exported checkpoints are for inference; preserve the original training checkpoint for resuming training.

## Enhance and evaluate audio

Use a checkpoint from your own run. Input must be mono and match its configured sample rate.

Native Lightning `.ckpt` files retain `model_config` and `model_state` compatibility fields for these existing tools. Inference, evaluation, export, and benchmarking keep their flag-based interfaces; the Hydra migration applies to training.

```bash
python -m realtime_use.infer \
  --checkpoint runs/fixed4-regression/checkpoints/last.ckpt \
  --input data/noisy/example.wav \
  --output runs/example-enhanced.wav \
  --exit-depth 4 --stream --device cpu
```

Evaluate only exits configured in that checkpoint:

```bash
python -m realtime_use.evaluate \
  --checkpoint runs/fixed4-regression/checkpoints/last.ckpt \
  --manifest data/validation.jsonl \
  --output runs/fixed4-evaluation.json \
  --exits 4 --stream --check-clean --device cpu
```

Benchmark the actual streaming path independently from quality evaluation:

```bash
python -m realtime_use.benchmark \
  --checkpoint runs/fixed4-regression/checkpoints/last.ckpt \
  --output runs/fixed4-timing.json \
  --exit-depth 4 --seconds 10 --warmup-seconds 1 \
  --device cpu --threads 1
```

The benchmark uses synthetic timing input unless `--input` supplies a recording. Its timing results do not establish enhancement quality. Increase the measurement duration for a deployment decision and repeat on the actual target hardware.

## Measure before scaling

With the supplied width-64, expansion-4, state-dimension-16 configurations, the current implementation has these measured active parameter counts:

- Four-block exit: **863,426**.
- Eight-block exit: **1,687,746**.
- Twelve-block exit: **2,512,066**.

An enhancement-only flexible 12-block model stores 2,512,066 parameters with a shared decoder and 2,586,438 with three per-exit decoders. Active counts include one selected decoder. The Lightning training module/checkpoint can additionally retain transition heads, discriminators, and optimizer state; these inference-model counts are not its total memory footprint. Re-measure after architecture changes:

```python
from realtime_use.config import ModelConfig
from realtime_use.model import Enhancer

model = Enhancer(ModelConfig(depth=12, exits=(4, 8, 12)))
for depth in model.config.exits:
    print(depth, model.active_parameter_count(depth))
```

These are counts for our independent encoder/decoder and are not directly comparable to the paper's reported architecture. The decoder adds a learned complex residual to the noisy spectrum, allowing nonzero predictions in missing input bins. That capability does not establish restoration quality. This baseline retains every frequency bin instead of downsampling frequency; smaller parameter counts therefore do not guarantee lower compute or memory use than the published model.

Measure streaming p50/p95/p99 processing time at batch size one on the target machine, including the audio framing path. With a 20 ms hop, sustained processing must fit within 20 ms per hop. The 40 ms analysis window is an algorithmic latency component; audio-device buffering and processing time add to it.

For a training-time estimate, benchmark the intended crop length, sample rate, precision, batch size, and training stage. Multiply measured seconds per generator update by `training.steps`, then add data preparation, validation, and checkpoint overhead. An adversarial update includes both discriminator and generator work. Published inference timing is not a training-throughput benchmark.

See [the experiment roadmap and acceptance gates](docs/experiments.md) for validation and deployment criteria.

## Repository and materials

The repository is public at [quangng2000/realtime-use](https://github.com/quangng2000/realtime-use). No public software license has been selected. See [NOTICE.md](NOTICE.md) and [the provenance ledger](docs/provenance.md) for the boundary between this project's code, third-party dependencies, research references, and future training materials.
