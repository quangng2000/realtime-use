# Method references and implementation boundaries

This project implements a trainable research model and explicit streaming contracts. Reusing author components and matching architectural settings do not establish published quality, measured runtime, or a completed paper reproduction. No training or optimizer updates were launched for the current parity update.

## What matches the real-time architecture

The reference is [One Model, Many Latencies, sections III-B and IV-B/C](https://arxiv.org/html/2606.25621v1), with architecture lineage from [USEMamba](https://arxiv.org/abs/2505.21198) and [RE-USE](https://arxiv.org/abs/2603.02641).

- **Mamba:** production presets instantiate the complete official Mamba-1 class, including upstream initialization, sequence execution, inference-cache allocation and stepping. Width 64, state dimension 16, expansion 4 and convolution width 4 match the released configuration. Temporal processing is unidirectional and precedes bidirectional frequency processing, with directional residuals, projections, channel LayerNorm and outer residuals. The legacy CPU mixer is a separate architecture, not validation of CUDA Mamba.
- **Author codec:** the actual `DenseEncoder`, `MagDecoder` and `PhaseDecoder` classes are reused from [NemotronLabs revision 31f82f2](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/models/codec_module_time_d4.py). Source and OpenMDW license are retained unchanged. The adapter changes child modules: causal temporal padding, channel-only normalization, removal of temporal down/up-sampling, and 3×3 input convolutions. Dense connections, frequency down/up-sampling and separate magnitude/phase heads retain the author structure. No restricted Real-time RE-USE wrapper or weights are copied.
- **Features:** input magnitude uses `log1p(abs(spectrum))` with phase; mapped magnitude reconstructs with `expm1(ReLU(mapped))`. These settings follow the released configurations. The model has no whole-utterance energy normalization.
- **Lookahead:** `model.lookahead_frames=[0,1,2]` registers three independent first-convolution branches. Full-sequence execution applies the selected left/right padding. Streaming primes only that branch's bounded input history, delays downstream computation until its future frames arrive, and drains pending outputs at the end. Other temporal layers remain causal. A branch switch requires resetting the stream.
- **Rates:** one model can select native 8, 16, 22.05, 24, 32, 44.1 or 48 kHz transforms per call, retaining a 40 ms window and 20 ms hop. The learned codec handles odd and even bin counts with explicit frequency-boundary padding/cropping. Supported rates are declared in the checkpoint; no implicit resampling occurs.
- **Exits:** depth 12 and the requested 4/8/12 exits are implemented. Shared decoders can transition to separately trained exit decoders by copying their weights and reducing the preceding modules' learning rate. The paper samples every exit from 3–12; the supplied profile uses the requested subset rather than claiming all thirty exit/lookahead settings.

Tests cover future-frame dependency limits, selected-branch gradients, full/streaming outputs and gradients, short utterances, partial tails, interleaved streams, all seven native rates, export pruning, and compatibility with old checkpoints. CPU tests use an explicitly selected reference mixer. Actual official-Mamba CUDA, precision and NCCL validation remains a separate prerequisite; interface mocks and CUDA skips are not numerical evidence.

## Framing and latency

New presets use ordinary Hann windows. The pinned [released STFT source](https://huggingface.co/nvidia/Real-time_RE-USE/blob/4c0c186272ca853cd18d832fc5c28777b62f32d1/models/stfts.py#L34) also uses Hann, but invokes centered STFT with reflect padding. Its [online example](https://huggingface.co/nvidia/Real-time_RE-USE/blob/4c0c186272ca853cd18d832fc5c28777b62f32d1/online_inference.py#L367) explicitly precomputes the waveform STFT and leaves online STFT integration for future work.

This project's real audio streaming path uses past zero padding for overlap initialization, drops artificial past output, and pads the end only during flush. Synthesis normalizes overlap by squared window weights. Full and streaming paths share those boundary conventions. They are intentional differences from the released centered-reflect edges, not an assertion that the release's model calls alone provide an end-to-end microphone pipeline.

The first valid output waits for **40 + 20 × lookahead ms** of input: 40/60/80 ms for the three branches. A finite clip shorter than that budget is completed at end-of-stream. Processing and device/network buffering add to these algorithmic latencies. The benchmark includes transforms, the model, overlap-add, host/device transfers and CUDA synchronization in per-hop service time. It does not measure microphone/playback scheduling or guarantee a real-time deadline. CPU/phone execution of official Mamba and a sub-20-ms end-to-end design remain unimplemented.

Checkpoints missing the new configuration fields retain the previous square-root-Hann, zero-lookahead, singleton-rate behavior and state-dict layout. Changing the window or adding branches is an architecture/configuration change for a new experiment; do not silently reinterpret an existing checkpoint.

## Profiles and training choices

`fixed4`, `fixed8`, `fixed12` and `flexible12` remain simple 16 kHz, zero-lookahead baselines, now using Hann. `paper_flexible12` adds all seven native rates, three lookahead branches, rate-scaled losses/critics, sampled exits and lookahead, full-utterance validation over all batches, and UTMOS22-strong selection. Its default exits remain 4/8/12, giving nine latency/depth settings.

Regression pretraining followed by adversarial fine-tuning updates one enhancement network. AdamW uses a starting learning rate of 0.0002; the shared-to-per-exit transition scales preceding-module learning rates by 0.1. These two transitions are separate. The baseline flexible recipe's 50,000-update decoder split is an example project schedule. The paper profile leaves it unset so its timing can be chosen after reviewing convergence. Changing the split schedule requires `init_from` into a new run with fresh optimizer/update state; native resume preserves the original schedule. The full author schedule, stopping rules, loss weights and exact ordering of transitions have not been recovered.

The offline RE-USE paper instead freezes its regression network and trains a separate residual generator. That two-network inference design is not implemented here. Optional cached-output KD is a separate, disabled-by-default experiment with explicit teacher/output provenance; GAN training does not require or run a teacher.

## Loss, discriminator and evaluation limits

Waveform L1 uses PyTorch; multi-resolution spectral convergence and log-magnitude losses use `auraloss`. Valid utterance prefixes receive equal weight. With rate scaling enabled, FFT sizes scale from `model.sample_rate` to each batch rate and round to the nearest integer; odd FFT sizes are allowed. This preserves approximate frame durations, not a byte-for-byte author loss implementation. Phase/perceptual-loss details and weights are not established as exact reproductions.

Adversarial training uses the official DAC multiband spectral CNN modules, with a maximum-rate set of bands registered before distributed setup and corresponding lower-band weights reused at lower rates. FFT sizes adapt to native rate. This is an identified library substitute: neither reviewed RE-USE release supplies the exact real-time critic, and DAC's final head differs from the offline paper's described two-convolution/global-pooling head. Periodic critics and DAC's peak-normalizing wrapper are not used. Hinge-GAN and feature-matching weights remain project choices.

The paper profile selects `best.ckpt` by maximizing UTMOS22 strong, the original learner used by URGENT, rather than UTMOSv2 or a full ensemble. Source and checkpoint assets must be supplied locally. It computes equal-utterance global means for each exit/lookahead setting and then averages those setting means equally. Native-rate audio is converted only as required by the metric adapter. See [metric setup, versions and limits](metrics.md); installed adapters without validated assets do not constitute measured scores.

Prepared aligned pairs and optional library corruptions do not reproduce the complete seven-distortion URGENT mixture, multilingual data proportions, data-quality filtering, or anechoic-target construction. The dataset and its permissions must be audited independently. Neither a manifest nor a passing synthetic test establishes universal enhancement quality.

## Before a long run

1. Run the [one-device FP32 preflight](training.md#preflight-without-training) with the real configuration and local assets, then verify CUDA gradients and the intended precision without launching an optimizer loop.
2. Audit aligned data, split separation and local metric assets; inspect the resolved profile.
3. In a separately authorized pilot, verify useful gradients, overfit a diagnostic subset, review held-out speech, and measure memory/update throughput.
4. Choose regression/adversarial and decoder-transition schedules explicitly; retain comparison checkpoints.
5. Evaluate every intended rate, exit and lookahead on held-out degradations and clean input, then measure deployment latency.

No model parameter count, perceptual quality, GPU training time or hardware latency is claimed from the papers for this assembled implementation. Measure the actual selected architecture and runtime.
