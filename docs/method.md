# Method references and implementation boundaries

The papers establish research directions. They do not establish the quality, computational cost, or legal suitability of this independent implementation.

## What the papers support

[One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](https://arxiv.org/abs/2606.25621) describes causal temporal processing, controlled look-ahead, variable output depth, shared-decoder training followed by per-exit decoders, and regression pretraining followed by adversarial fine-tuning. Its real-time design avoids a second enhancement network during inference. The reported framing uses a 40 ms window and a 20 ms hop.

[Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641) describes aligned anechoic targets, training-data quality filtering, and a different two-stage architecture: train a regression network, freeze it, and train a separate residual generative network. The reported experiment uses a 30-layer regression network and a 6-layer generator; the design is non-causal. This is not the inference architecture implemented in this project.

These are distinct uses of “two stage.” In addition, shared-to-per-exit decoder training is another transition. A complete combined schedule, including its exact ordering with adversarial fine-tuning and all convergence criteria, could not be established from the materials reviewed. Our configurations must make their own choices explicit.

## Current implementation choices

- Fixed-depth baselines use 4, 8, or 12 time–frequency blocks. A block includes internal operations and should not be equated with one primitive neural-network layer.
- A separate flexible configuration uses a maximum depth of 12 and exits at 4, 8, and 12. It does not reproduce the paper's full range of sampled exits.
- The initial audio contract is mono 16 kHz, a 40 ms window, a 20 ms hop, and zero look-ahead. There is no whole-utterance normalization.
- The transform uses a square-root Hann window, past zero padding for overlap initialization, and end-of-stream zero padding. It discards the artificial past samples; the first valid streamed output still waits for a full 40 ms input window. These boundary conventions are our implementation choices.
- The selective state-space implementation, encoder, decoders, loss weighting, and discriminators are developed for this repository. Architectural resemblance is not an exact-reproduction claim.
- The decoder predicts an additive complex residual over the noisy spectrum. It can express nonzero output in an absent input bin, but effective packet-loss concealment or bandwidth restoration must be learned and evaluated. It does not reproduce the authors' codec.
- The encoder keeps the full frequency resolution. State memory and compute therefore depend on frequency-bin count as well as depth; a smaller parameter count need not mean less runtime work than the published architecture.
- A Python reference scan supports correctness and CPU tests. An optional `mamba_ssm` scan accelerates sequence processing on CUDA, while `auto` permits fallback. The CUDA path and mixed-precision execution remain unverified by CPU-only validation; the reference backend's speed is not representative of a tuned deployment kernel.
- Model depth and exit selection are explicit. Per-exit active parameter counts must be measured from the implementation; the papers' counts must not be relabeled as this model's counts.
- Regression training combines waveform L1 with multi-resolution spectral convergence and log-magnitude losses. Adversarial training continues updating the enhancement network and adds hinge-GAN and feature-matching objectives. Discriminators are not part of enhancement inference.
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
