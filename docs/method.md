# Method references and implementation boundaries

The papers establish research directions. They do not establish the quality, computational cost, or legal suitability of this independent implementation.

## Verification against the real-time paper

The reference for this comparison is [arXiv:2606.25621v1, sections III-B and IV-B/C](https://arxiv.org/html/2606.25621v1). The architecture lineage also includes [USEMamba, arXiv:2505.21198](https://arxiv.org/abs/2505.21198). The published real-time implementation [imports the full `mamba_ssm.Mamba` class](https://huggingface.co/nvidia/Real-time_RE-USE/blob/4c0c186272ca853cd18d832fc5c28777b62f32d1/models/streaming_mamba_block2_SEMamba.py#L16). This is evidence about dependency usage; that wrapper is not incorporated into this project.

- **Mamba dependency:** new Hydra presets use full official Mamba-1 blocks from [state-spaces/mamba](https://github.com/state-spaces/mamba). The adapter delegates sequence execution, inference-cache allocation, and recurrent stepping. The old custom mixer is retained only as an explicit legacy architecture for existing checkpoints and CPU smoke tests. Passing those tests does not validate the official CUDA kernels.
- **Causality and normalization:** the temporal path is unidirectional, frequency processing is bidirectional within an available frame, convolutions pad only past time, and LayerNorm operates on channels. Legacy tests cover future-frame perturbations and full/streaming agreement; CUDA-gated tests cover the official Mamba path when suitable hardware and dependencies are available.
- **Depth and model size:** the maximum configured depth is 12, but the independent encoder, feature representation, and complex-residual decoder differ from the authors' architecture. Using the same Mamba library does not reproduce the reported 3.7M-parameter network.
- **Exits and look-ahead:** the requested 4/8/12 exit subset is implemented. The paper samples exits 3–12 and look-ahead 0–2; this project implements only zero look-ahead and has no parallel look-ahead convolution branches.
- **STFT and sample rates:** the configured transform uses a 40 ms window and 20 ms hop. Window and hop sizes derive from the configured sample rate, but each run/checkpoint and paired-data loader use one fixed rate. This is not the paper's one-model, multiple-rate training/inference interface.
- **Training stages:** regression pretraining followed by adversarial fine-tuning updates one enhancement network. AdamW defaults to 0.0002; copying the shared decoder into per-exit decoders scales the preceding modules' learning rate by 0.1. The default transition at 50,000 updates is a project schedule, not a measured convergence criterion. Loss and discriminator definitions remain independent choices.
- **Data and quality evaluation:** prepared aligned pairs and optional `audiomentations` Gaussian noise, clipping, and low-pass filtering are supported. The full seven-distortion URGENT pipeline and data-quality filtering are not implemented. Manifests do not establish that clean targets are anechoic. Built-in evaluation delegates MSE to PyTorch and SNR/SI-SDR to TorchMetrics; it does not implement PESQ, ESTOI, SBERT, LPS, CAcc, DNSMOS, NISQA, or UTMOS. Checkpoint selection uses reconstruction loss, not those perceptual scores.
- **Latency:** the benchmark processes streaming chunks and reports timing quantiles. It does not invoke `torch.compile` or CUDA graphs. Results remain device-specific; no A100/3090/4090 or phone measurements have been established for this repository. The zero-look-ahead algorithmic latency is 40 ms, before processing and device buffering.

This verifies specific implementation properties and records the remaining gaps. It does not establish paper reproduction or published speech quality.

## What the papers support

[One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](https://arxiv.org/abs/2606.25621) describes causal temporal processing, controlled look-ahead, variable output depth, shared-decoder training followed by per-exit decoders, and regression pretraining followed by adversarial fine-tuning. Its real-time design avoids a second enhancement network during inference. The reported framing uses a 40 ms window and a 20 ms hop.

[Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641) describes aligned anechoic targets, training-data quality filtering, and a different two-stage architecture: train a regression network, freeze it, and train a separate residual generative network. The reported experiment uses a 30-layer regression network and a 6-layer generator; the design is non-causal. This is not the inference architecture implemented in this project.

These are distinct uses of “two stage.” In addition, shared-to-per-exit decoder training is another transition. A complete combined schedule, including its exact ordering with adversarial fine-tuning and all convergence criteria, could not be established from the materials reviewed. Our configurations must make their own choices explicit.

## Current implementation choices

- Fixed-depth baselines use 4, 8, or 12 time–frequency blocks. A block includes internal operations and should not be equated with one primitive neural-network layer.
- A separate flexible configuration uses a maximum depth of 12 and exits at 4, 8, and 12. It does not reproduce the paper's full range of sampled exits.
- The initial audio contract is mono 16 kHz, a 40 ms window, a 20 ms hop, and zero look-ahead. There is no whole-utterance normalization.
- The transform uses a square-root Hann window, past zero padding for overlap initialization, and end-of-stream zero padding. It discards the artificial past samples; the first valid streamed output still waits for a full 40 ms input window. These boundary conventions are our implementation choices.
- The new Mamba core is supplied by the official library. The encoder, decoders, loss weighting, and discriminators are developed for this repository. Architectural resemblance is not an exact-reproduction claim.
- The decoder predicts an additive complex residual over the noisy spectrum. It can express nonzero output in an absent input bin, but effective packet-loss concealment or bandwidth restoration must be learned and evaluated. It does not reproduce the authors' codec.
- The encoder keeps the full frequency resolution. State memory and compute therefore depend on frequency-bin count as well as depth; a smaller parameter count need not mean less runtime work than the published architecture.
- The official `mixer_impl=mamba_ssm` architecture requires CUDA and never changes architecture automatically. The legacy mixer has a Python reference scan and an optional CUDA scan kernel. Its `scan_backend=auto` can fall back to the Python scan; this legacy setting must not be confused with selecting the full official architecture. CUDA and mixed-precision execution remain unverified by CPU-only validation.
- Model depth and exit selection are explicit. Per-exit active parameter counts must be measured from the implementation; the papers' counts must not be relabeled as this model's counts.
- New regression presets combine PyTorch waveform L1 with `auraloss` multi-resolution spectral convergence and log-magnitude losses. The adapter evaluates valid utterance prefixes separately; library STFT conventions differ from the retained legacy loss, so they are distinct objectives. Adversarial training continues updating the enhancement network and adds hinge-GAN and feature-matching objectives. Discriminators are not part of enhancement inference.
- Optional cached-output distillation during regression is a separate experiment. Its default weight is zero, and adversarial training ignores teacher tensors. It is neither a prerequisite for the GAN stage nor an author-verified recipe.

## Suggested experiment order

1. Establish fixed 4-, 8-, and 12-block regression baselines on identical splits and an explicitly recorded training budget.
2. Continue each selected baseline with adversarial fine-tuning. Compare quality and content preservation before accepting the added objective.
3. Train a flexible model with a shared decoder, then clone it into per-exit decoders using a recorded transition point and learning-rate change.
4. Evaluate how that decoder transition interacts with adversarial training as a separate ablation. Record the sequence used; do not attribute it to the authors without supporting evidence.
5. Only then consider teacher distillation, additional sample rates, parallel look-ahead convolutions, or a smaller analysis window.

## Deliberately incomplete parts

The initial project does not reproduce all seven URGENT degradation types, the full multilingual training data mixture, sampling-frequency-independent STFT training, learned behavior at nonzero look-ahead, a complete author training schedule, or published perceptual scores. A fixed 16 kHz paired-data experiment does not justify calling the trained model universal.

Changing the window to pursue sub-40 ms algorithmic latency is a new model experiment. Shortening it after training can alter the model's input representation and quality. Device buffering and processing time must still be measured separately.

## Checkpoints versus papers

The [original RE-USE model card](https://huggingface.co/nvidia/RE-USE) explicitly says its released checkpoint differs from the paper. Paper architecture descriptions and checkpoint architecture descriptions must not be treated as interchangeable. No released checkpoint is required, downloaded, or included here.

The [NVIDIA NemotronLabs RE-USE card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) describes a separate release under OpenMDW-1.1. Naming that available release does not select it as this project's teacher or establish permission for a particular proposed training use. Any future external model requires a recorded, use-specific materials review.
