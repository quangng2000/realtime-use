# Experiments and acceptance gates

This roadmap defines what to measure before expanding the training budget or making deployment claims. All numerical targets below are project decisions to be filled or adopted for a specific use case, not reported results from this repository. No training or optimizer updates were launched for the current parity update. Later training gates are a proposed workflow, not completed work.

## 1. Verify the software path

First run the [configured preflight](training.md#preflight-without-training), then forward/backward and inference-only tests without optimizer steps. Preflight loads local assets and checks every configured rate/exit/lookahead on one device in FP32; it does not exercise AMP, NCCL, gradients or optimizer steps. Verify finite outputs/gradients, checkpoint load/export, exact paired cropping, and waveform length preservation. Use `python -m pytest -q --no-training` to skip optimizer/lifecycle cases and activate the test-local guards. Default push/PR CI uses the same mode. The full suite without that flag and the smoke training command contain optimizer/lifecycle tests; run those only when that execution is intended. Compare streaming and whole-utterance evaluation under the same native rate, exit, lookahead and precision. Reset recurrent state between independent recordings.

Before training, verify the actual official Mamba class, full/step outputs and gradients on CUDA at the chosen precision. The independent legacy recurrence is not a substitute for that architecture. CPU success does not validate that path. Check that streaming boundary behavior stays stable with learned spectral changes and that persistent state storage remains bounded across long input chunks.

Passing this gate means the tested implementation paths work. It does not demonstrate speech enhancement quality.

Smoke mode uses a one-block, width-4 network even when given a fixed-depth experiment. Full-size 4-/8-/12-block GPU memory and throughput require separate runs. The opt-in CPU/Gloo integration tests exercise DDP gradients, accumulation, sampled exits, decoder transitions, resume, and exact validation weighting; earlier results are historical coverage and these training-loop tests are excluded from current no-training verification. CUDA/NCCL and multi-GPU scaling still need GPU measurements. The optional CUDA parity test is skipped in a CPU-only environment, so that skip is not CUDA validation.

## 2. Establish a licensed data pilot

Select a manageable set of aligned clean/noisy recordings with documented training rights. A pilot subset size is an experiment choice; neither paper guarantees a quality level from a particular number of pilot hours.

Before the pilot:

- Record sources, versions, applicable terms, intended uses, and preprocessing in the provenance ledger.
- Separate training and validation speakers and recordings. Prevent duplicates and near-duplicates across splits.
- Inspect clean targets for noise, reverberation, clipping, and non-speech artifacts.
- For simulated reverberation, construct and check the direct-path-aligned anechoic target. Do not replace it with an arbitrary shift inferred from waveform length.
- Include representative real microphone examples in a held-out evaluation set where the relevant permissions allow this.

Train the fixed four-block regression model first. Inspect clean-input behavior, speech preservation, artifacts, and improvement across each included degradation. Only increase the budget after the model can learn a small diagnostic subset and performs plausibly on held-out audio.

## 3. Compare fixed depths

Run fixed 4-, 8-, and 12-block models with the same data split and preprocessing. Record the comparison basis: equal generator updates, equal audio exposure, or equal compute budget. These answer different questions. A Hydra multirun can schedule the experiment configurations; it is not distributed training of one model.

For each run, retain:

- Repository revision, full config, random seed, checkpoint identifier, and data manifest/version.
- Trainable and per-exit active parameter counts measured from the model.
- Configured and sampled native rates, lookahead branches, crop length, batch size, gradient accumulation, precision, optimizer, stage, and generator-update count. In the adversarial stage, Lightning's optimizer-step counter includes both optimizers; do not equate it with generator updates.
- GPU model/count, peak memory, measured optimizer-step time, wall-clock time, and validation/checkpoint overhead.
- Held-out waveform/spectral metrics, listening notes, and any separately installed speech-quality or ASR metrics with their versions and evaluation settings.

Do not label a surrogate spectral metric as PESQ, DNSMOS, UTMOS, or ASR accuracy. Optional metric adapters need their documented local assets and independent validation before reported scores are treated as evidence; see [metrics.md](metrics.md). UTMOS22 strong is the paper-profile checkpoint monitor, not UTMOSv2 or a full ensemble.

## 4. Evaluate adversarial fine-tuning

Initialize from a regression checkpoint and train the same enhancement network with discriminator losses. A teacher is absent by default. Record generator/discriminator learning rates, update policy, loss weights, and the stopping criterion.

Accept the stage only if a held-out comparison shows the intended perceptual benefit without unacceptable changes in linguistic content, speaker identity, or clean-input behavior. Set task-specific thresholds before examining the final test set. Preserve the regression checkpoint as the control.

This stage is not the offline paper's frozen-regressor-plus-separate-generator pipeline.

## 5. Evaluate the flexible model

Compare its exits at 4, 8, and 12 against the independently trained fixed baselines at matched rates and lookahead. `paper_flexible12` spans seven rates and three branches; compare all nine exit/lookahead settings, not only the largest exit. Record exit/lookahead sampling probabilities, the shared-decoder duration, the decoder split step, and the pre-decoder learning-rate scale. Save checkpoints before and after the split. The paper profile leaves the split unset until convergence is reviewed, selects UTMOS22 strong, evaluates full utterances, and removes the validation batch cap.

The regression/adversarial transition and the shared/per-exit transition need separate labels in run metadata. Test their ordering explicitly rather than claiming the complete author schedule is known.

## 6. Establish streaming feasibility

Benchmark batch size one with the full streaming audio path on the intended deployment machine. Use warm-up, device synchronization for CUDA measurements, and sufficient iterations to report p50, p95, and p99 processing time. Include hardware, operating system, dependency versions, precision, mixer implementation, sample rate, exit, lookahead, warm-up, and measurement counts with each result.

The initial hop deadline is **20 ms**. Sustained processing must remain below that budget, including overhead outside the neural network. A useful initial engineering target is p99 below 10 ms to leave room for integration overhead; this is our proposed margin, not an author result or a production guarantee. Also measure sustained overruns and long-run buffer behavior, because a percentile alone cannot rule out accumulated delay.

Report algorithmic latency, processing latency, and device/transport buffering separately. Algorithmic latency is 40/60/80 ms for lookahead 0/1/2; those are not microphone-to-speaker latency guarantees. Verify first-output timing, exact tail length after flush, stream resets and long-lived bounded state at every intended rate/branch. CPU, laptop, phone, and server-GPU results cannot be substituted for one another.

## 7. Scale only after measuring throughput

Measure several hundred representative generator updates after warm-up. Include the intended data loader, augmentation, crop duration, sample rate, precision, and stage. Adversarial training needs its own measurement; each generator update includes discriminator and feature-matching work.

Estimate elapsed training time as:

```text
planned generator updates × measured seconds per generator update
+ validation + checkpointing + data preparation + scheduling overhead
```

State the assumptions and a range. Neither model parameter count nor published inference timing is sufficient to predict training time. Multi-GPU speedup also requires measurement; it is not automatically linear.

## Later experiments

Native multi-rate and nonzero-lookahead paths are implemented and tested structurally, but their learned quality has not been established. Completing the full degradation generator, shortening windows, optimizing kernels, quantization and mobile deployment require separate quality and latency experiments. A 40 ms window cannot meet an under-20-ms end-to-end requirement regardless of GPU speed. Optional distillation must have a separate baseline, licensed teacher/output provenance, and measured preprocessing cost.

No acceptance gate here authorizes cloud resource rental, paid training jobs, dataset acquisition, or publication of weights.
