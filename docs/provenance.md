# Materials provenance ledger

This ledger separates research references, project code, dependencies, training data, and optional external models. It records provenance and review status; it is **not legal clearance** for a commercial product or a blanket determination of downstream rights.

## Current project boundary

- **Project code:** independently implemented for the private `quangng2000/realtime-use` repository. No public software license has been selected. See [NOTICE.md](../NOTICE.md).
- **NVIDIA model source and checkpoints:** none included or automatically downloaded. In particular, restricted source and tensors from the original `nvidia/RE-USE` and `nvidia/Real-time_RE-USE` releases are not implementation dependencies.
- **Training audio:** none supplied as a production dataset. Synthetic smoke audio is generated locally for software tests and is not evidence of speech performance.
- **Teacher outputs:** none supplied. Distillation is disabled by default.
- **Paid compute:** not provisioned by this repository. Tests and smoke paths can run on CPU.

## Research references

1. [One Model, Many Latencies: Universal Speech Enhancement for Diverse Real-Time Applications](https://arxiv.org/abs/2606.25621). Method reference for causal enhancement, controllable depth/look-ahead, shared-to-per-exit decoder training, and regression followed by adversarial training. No paper figures, model code, or weights are redistributed here.
2. [Rethinking Training Targets, Architectures and Data Quality for Universal Speech Enhancement](https://arxiv.org/abs/2603.02641). Method reference for target alignment, data curation, and the distinct frozen-regression/residual-generator design. That offline two-network architecture is not the current project architecture.

The [original RE-USE card](https://huggingface.co/nvidia/RE-USE) identifies its release as research/development-only under NVIDIA's noncommercial license. The [NemotronLabs RE-USE card](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE) identifies a separate release under OpenMDW-1.1. Neither has been selected as a training material. Do not infer identical permissions from similar names, shared authors, or related papers.

## Dependency records

Runtime and test dependencies are declared in `pyproject.toml`. They are installed separately rather than vendored. Before distributing software or model artifacts, record the exact resolved versions and review the applicable package licenses and notices, including transitive dependencies and any subsequently added acceleration kernels.

Changing from the reference scan to third-party optimized code is a new dependency decision. Record its source revision and terms before incorporation.

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

Identify the training run, source revision, configuration, complete materials ledger, dependency versions, evaluation evidence, known limitations, and the license actually selected by the owner. A private repository, independent implementation, or a successful test suite alone does not settle rights to training data or trained artifacts.
