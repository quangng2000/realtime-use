# NVIDIA NemotronLabs codec source

`codec_module_time_d4.py` and `LICENSE` are unmodified copies from
[nvidia/NVIDIA-NemotronLabs-RE-USE](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/tree/31f82f2efad1095034492f236e8dac9239ba713b),
revision `31f82f2efad1095034492f236e8dac9239ba713b`.

The source file explicitly declares `SPDX-License-Identifier: OpenMDW-1.1`.
The accompanying license covers model architecture and associated software and
includes NVIDIA's third-party notices. Preserve these files and notices when
redistributing this source or substantial portions of it. This source is from
the commercial NemotronLabs release, not the separately licensed Real-time
RE-USE repository. No model weights are included.

The local `realtime_use.author_codec` adapter instantiates the actual upstream
classes and changes their child modules. It replaces instance normalization
with channel-only LayerNorm; changes the encoder's first convolution to a causal
3-by-3 convolution; removes factor-four temporal down/up-sampling; and makes
temporal convolutions causal. It retains the dense connections, separate mapped
magnitude and phase heads, frequency down/up-sampling, and two-stage decoder
structure. Both this release and the real-time recipe select `relu_log1p`:
the adapter uses `log1p(abs(spectrum))` at its input and `expm1(ReLU(mapped))`
at its output, following the commercial release's `models/stfts.py`.

When configured, the first convolution has independent branches for lookahead
0, 1 and 2. Full-sequence execution controls left/right padding in that layer;
streaming delays downstream computation until the selected future frames arrive
and drains pending outputs at end-of-stream. Remaining temporal convolutions
use only past context. Single-branch configurations retain the older parameter
layout. Convolution caches are external per-stream dictionaries; full-sequence
training and one-frame inference call the same upstream `forward` methods. Two
zero frequency bins are appended at the encoder boundary and decoded output is
cropped to the original bin count, including even counts. These adaptations are
local and do not import NVIDIA's separately licensed realtime wrapper.

SHA-256 hashes of the unmodified upstream files:

- `codec_module_time_d4.py`: `02bd1d478f36a9b38dce7d44368805298f42798ea239304580f1ccffafe766a0`
- `LICENSE`: `a3d882953f026d3e2fdd57842d9c522c8320d4c0388af2278bcf4d438d255a2c`
