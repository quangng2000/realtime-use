# Materials provenance ledger

This ledger separates research references, project code, dependencies, training data, and optional external models. It records provenance and review status; it is **not legal clearance** for a commercial product or a blanket determination of downstream rights.

## Current project boundary

- **Project integration code:** developed for the public [quangng2000/realtime-use](https://github.com/quangng2000/realtime-use) repository. Public visibility does not select a software license for project-owned code; included third-party source retains its own license. See [NOTICE.md](../NOTICE.md).
- **NVIDIA source:** the commercially licensed NemotronLabs codec is vendored at the pinned revision below. Restricted source from the original `nvidia/RE-USE` and `nvidia/Real-time_RE-USE` releases is not included or used as an implementation dependency.
- **NVIDIA checkpoints:** none included, downloaded, or used as initialization/teacher weights.
- **Training audio:** none supplied as a production dataset. Synthetic smoke audio is generated locally for software tests and is not evidence of speech performance.
- **Teacher outputs:** none supplied. Distillation is disabled by default.
- **Paid compute:** not provisioned by this repository. Tests and smoke paths can run on CPU.

## Research references

1. [One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](https://arxiv.org/abs/2606.25621). Method reference for causal enhancement, controllable depth/look-ahead, shared-to-per-exit decoder training, and regression followed by adversarial training. No paper figures, model code, or weights are redistributed here.
2. [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641). Method reference for target alignment, data curation, and the distinct frozen-regression/residual-generator design. That offline two-network architecture is not the current project architecture.
3. [Universal Speech Enhancement with Regression and Generative Mamba](https://arxiv.org/abs/2505.21198). USEMamba architecture lineage. Its flow-matching variant is not implemented here.

The [original RE-USE card](https://huggingface.co/nvidia/RE-USE) identifies its release as research/development-only under NVIDIA's noncommercial license. The [NemotronLabs RE-USE card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) identifies a separate release under OpenMDW-1.1. Only the explicitly licensed codec source from the latter is selected here, not either release's weights. Do not infer identical permissions from similar names, shared authors, or related papers.

## Vendored author codec

Source: [NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b), revision `31f82f2efad1095034492f236e8dac9239ba713b`, reviewed 2026-10-25. The file `models/codec_module_time_d4.py` declares `SPDX-License-Identifier: OpenMDW-1.1`. The accompanying license defines the covered materials to include associated software and permits use/modification/distribution subject to retained license and origin notices.

The original source and complete license/third-party notices are retained unchanged under [the vendor directory](../src/realtime_use/_vendor/nemotron_codec/README.md), which records SHA-256 hashes and the adaptation boundary. Git attributes preserve their exact bytes; tests verify both hashes. The package includes those notices. `author_codec.py` instantiates the original classes and changes child modules to implement the published causal constraints. No pretrained tensors or restricted real-time source are copied.

This source decision covers reuse of those supplied software materials. It does not establish rights to training audio or future trained artifacts, or equality to the separately released real-time architecture/training recipe.

## Dependency records

Runtime and test dependencies are declared in `pyproject.toml` and installed separately. The author codec above is the explicit vendored-source exception. Before distributing software or model artifacts, record the exact resolved versions and review the applicable package licenses and notices, including transitive dependencies and any subsequently added acceleration kernels.

Adversarial presets use `descript-audio-codec==1.0.0` and `descript-audiotools==0.7.2`. DAC's [official discriminator source](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/dac/model/discriminator.py) and [license](https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/LICENSE) identify the MIT implementation; [audiotools](https://github.com/descriptinc/audiotools/blob/master/LICENSE) is also MIT. The project imports DAC's `MRD` class directly and does not load codec checkpoints or its full peak-normalizing/periodic-discriminator wrapper. The STFT configuration and band selection are recorded in each run; this is a reusable approximation to the cited discriminator family, not recovered real-time author training code.

New reconstruction presets use [auraloss 0.4.0](https://github.com/csteinmetz1/auraloss/tree/v0.4.0), under [Apache-2.0](https://github.com/csteinmetz1/auraloss/blob/v0.4.0/LICENSE). Optional input augmentation uses [audiomentations 0.43.1](https://github.com/iver56/audiomentations/tree/v0.43.1), under [MIT](https://github.com/iver56/audiomentations/blob/v0.43.1/LICENSE). Neither integration downloads audio or pretrained model weights. Evaluation records installed PyTorch/TorchMetrics versions because library metric numerics are part of the reported result.

The optional CUDA dependencies are the official [state-spaces/mamba v2.2.6.post3](https://github.com/state-spaces/mamba/tree/v2.2.6.post3), under [Apache-2.0](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/LICENSE), and [causal-conv1d v1.6.0](https://github.com/Dao-AILab/causal-conv1d/tree/v1.6.0), under [BSD-3-Clause](https://github.com/Dao-AILab/causal-conv1d/blob/v1.6.0/LICENSE). The project imports these packages; it does not vendor their implementations or use pretrained language-model weights. The new Mamba adapter delegates to the complete upstream Mamba-1 class. This dependency is distinct from NVIDIA's restricted enhancement wrapper and model weights.

## Required record for every future dataset

Create a dated entry before use with:

- Dataset title, provider, original source URL, and immutable version or file checksums.
- How the files were obtained and whether downloads, redistribution, modification, and the intended training use are permitted.
- The exact license/terms URL and a retained copy or version identifier of the applicable terms.
- Any restrictions relevant to commercial use, redistribution, attribution, privacy/consent, speaker data, or derived artifacts.
- Who reviewed the proposed use and the resulting status; unresolved questions stay unresolved.
- Audio composition, languages, sample rates, split method, and known overlap with evaluation data.
- Preprocessing, filtering, degradation synthesis, alignment, and any resulting derived dataset identifiers.

An accessible download URL is not evidence of training permission. A paper's data list does not transfer rights to the project. Do not automatically download the paper's training corpora.

## Required record for a future teacher or external checkpoint

Record the provider, exact model repository and revision, checkpoint hash, license/terms version, permitted training and output uses, restrictions on derived models, and any distribution obligations relevant to the proposed workflow. Record whether outputs are generated locally or through a service with separate terms.

Cached teacher outputs need a link to that record and to the source-audio permissions. The presence of a `teacher` field in a manifest does not establish permission. Keep optional distillation disabled until the relevant materials are approved for the intended use.

## Required record before distributing trained weights

Identify the training run, source revision, configuration, complete materials ledger, dependency versions, evaluation evidence, known limitations, and the license actually selected by the owner. Repository visibility, independent implementation, or a successful test suite alone does not settle rights to training data or trained artifacts.
