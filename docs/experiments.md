# Experiments and acceptance gates

This roadmap defines what to measure before expanding the training budget or making deployment claims. All numerical targets below are project decisions to be filled or adopted for a specific use case, not reported results from this repository.

## 1. Verify the software path

Run the CPU test suite and the tiny synthetic smoke experiment. Verify finite losses, parameter updates, checkpoint save/load, exact paired cropping, and waveform length preservation. Compare streaming and whole-utterance evaluation under the same exit and precision. Reset recurrent state between independent recordings.

Before a CUDA run, also verify the optional scan kernel against the reference recurrence and compare gradients at the chosen precision. CPU success does not validate that path. Check that streaming boundary behavior stays stable with learned spectral changes and that persistent state storage remains bounded across long input chunks.

Passing this gate means the tested implementation paths work. It does not demonstrate speech enhancement quality.

Smoke mode uses a one-block, width-4 network even when given a fixed-depth config. Full-size 4-/8-/12-block GPU memory and throughput require separate runs. The current trainer is single-device; multi-GPU scaling is future work. The optional CUDA parity test is skipped in a CPU-only environment, so that skip is not CUDA validation.

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

Run fixed 4-, 8-, and 12-block models with the same data split and preprocessing. Record the comparison basis: equal optimizer steps, equal audio exposure, or equal compute budget. These answer different questions.

For each run, retain:

- Repository revision, full config, random seed, checkpoint identifier, and data manifest/version.
- Trainable and per-exit active parameter counts measured from the model.
- Training sample rate, crop length, batch size, gradient accumulation, precision, optimizer, stage, and step count.
- GPU model/count, peak memory, measured optimizer-step time, wall-clock time, and validation/checkpoint overhead.
- Held-out waveform/spectral metrics, listening notes, and any separately installed speech-quality or ASR metrics with their versions and evaluation settings.

Do not label a surrogate spectral metric as PESQ, DNSMOS, UTMOS, or ASR accuracy. A metric absent from the evaluator needs a separate implementation and validation before it appears in a result.

## 4. Evaluate adversarial fine-tuning

Initialize from a regression checkpoint and train the same enhancement network with discriminator losses. A teacher is absent by default. Record generator/discriminator learning rates, update policy, loss weights, and the stopping criterion.

Accept the stage only if a held-out comparison shows the intended perceptual benefit without unacceptable changes in linguistic content, speaker identity, or clean-input behavior. Set task-specific thresholds before examining the final test set. Preserve the regression checkpoint as the control.

This stage is not the offline paper's frozen-regressor-plus-separate-generator pipeline.

## 5. Evaluate the flexible model

Compare its exits at 4, 8, and 12 against the independently trained fixed baselines. Record exit-sampling probabilities, the shared-decoder duration, the decoder split step, and the pre-decoder learning-rate scale. Save checkpoints before and after the split.

The regression/adversarial transition and the shared/per-exit transition need separate labels in run metadata. Test their ordering explicitly rather than claiming the complete author schedule is known.

## 6. Establish streaming feasibility

Benchmark batch size one with the full streaming audio path on the intended deployment machine. Use warm-up, device synchronization for CUDA measurements, and sufficient iterations to report p50, p95, and p99 processing time. Include hardware, operating system, dependency versions, precision, scan backend, sample rate, exit, warm-up, and measurement counts with each result.

The initial hop deadline is **20 ms**. Sustained processing must remain below that budget, including overhead outside the neural network. A useful initial engineering target is p99 below 10 ms to leave room for integration overhead; this is our proposed margin, not an author result or a production guarantee. Also measure sustained overruns and long-run buffer behavior, because a percentile alone cannot rule out accumulated delay.

Report algorithmic latency, processing latency, and device/transport buffering separately. The current 40 ms window is not a promise of 40 ms microphone-to-speaker latency. CPU, laptop, phone, and server-GPU results cannot be substituted for one another.

## 7. Scale only after measuring throughput

Measure several hundred representative optimizer steps after warm-up. Include the intended data loader, augmentation, crop duration, sample rate, precision, and stage. Adversarial training needs its own measurement; it adds discriminator and feature-matching work.

Estimate elapsed training time as:

```text
planned optimizer steps × measured seconds per optimizer step
+ validation + checkpointing + data preparation + scheduling overhead
```

State the assumptions and a range. Neither model parameter count nor published inference timing is sufficient to predict training time. Multi-GPU speedup also requires measurement; it is not automatically linear.

## Later experiments

Add multi-rate support and the full degradation generator only after the fixed-rate baselines pass the gates above. Nonzero look-ahead, shorter windows, optimized scan kernels, quantization, and mobile deployment each require fresh quality and latency checks. Optional distillation must have a separate baseline, licensed teacher/output provenance, and measured preprocessing cost.

No acceptance gate here authorizes cloud resource rental, paid training jobs, dataset acquisition, or publication of weights.
