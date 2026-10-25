# Notice

This is a research integration in the public `quangng2000/realtime-use` repository. No public software license has been selected for project-owned code; public visibility does not grant a license to redistribute that code or future model artifacts. Included third-party source retains its own license, independently of project-owned code.

The project cites research by the authors of [One Model, Many Latencies](https://arxiv.org/abs/2606.25621) and [Rethinking Training Targets, Architectures and Data Quality](https://arxiv.org/abs/2603.02641). Citation does not imply affiliation, endorsement, an exact reproduction, or matching experimental results.

No NVIDIA checkpoints or restricted NVIDIA model source files are included. The vendored NemotronLabs codec in `src/realtime_use/_vendor/nemotron_codec` is supplied under its retained OpenMDW-1.1 license and notices. Project adapters make explicitly documented causal modifications by replacing child modules; the vendored source itself is unchanged. DAC and other installed dependencies remain subject to their respective licenses and notices. Training datasets, cached teacher outputs, and trained weights require their own provenance and terms review before use or distribution.

See [the provenance ledger](docs/provenance.md) for the current materials boundary and required records. This notice is not legal clearance for commercial use.
