# Materializing downloaded source pools

Downloaded source archives contain clean speech, noise, RIRs, or generated wind. They are not paired training examples. `realtime_use.materialize_sources` verifies and unpacks one source/split JSONL inventory into the local SCP files consumed by the [URGENT preparation adapter](../data/README.md). It does not download data, render corruptions, start training, or grant dataset permissions.

Use a JSONL inventory whose records contain `id`, `source_id`, `source`, `split`, `kind`, `sample_rate`, `frames`, `channels`, `sha256`, `shard`, and `path`. `shard` is relative to the dataset root; `path` names the audio member inside that archive. Speech records also require `speaker_id` and may contain `text`. The bridge preserves all original fields in its output records. Each invocation accepts one corpus and one split, and validates channel count, exact frame/rate metadata, finite decoded samples in every source channel, and the original file checksum. RIR inventories may use the unsplit `pool` label; speech/noise/wind require train, validation, or test. It never resamples or averages channels.

Inspect the deterministic first-20-record readiness plan without writing anything:

```bash
python -m realtime_use.materialize_sources \
  --root /workspace-global/realtime-use/urgent2025-research-v1 \
  --manifest /workspace-global/realtime-use/urgent2025-research-v1/sources/vctk/train.jsonl \
  --output-dir /workspace-global/realtime-use/materialized-check/vctk/train \
  --role speech --max-items 20
```

Add `--execute` to materialize that selection. For the complete pool, omit `--max-items` and choose a separate full-data output directory. First-N subsets are software-readiness fixtures, not representative evaluation sets. Other roles are `noise`, `rir`, and `wind`; wind source IDs must already begin with `wind_noise`.

The default `--channel-policy reject` requires mono sources. Use `--channel-policy first` explicitly for multichannel sources such as WHAM noise or measured RIRs. This follows official URGENT's `read_audio(force_1ch=True)` choice of channel zero. Original archived bytes are verified first, including finite samples in all channels. Multichannel sources become mono float32 WAV files containing exactly their decoded first channel. Mono sources retain their original bytes. No channel averaging or resampling occurs.

The output contains:

- `audio/`: verified original mono files or explicit first-channel float WAV derivatives, named by a hash of corpus and source identity.
- `verification/`: per-file source/output hash records for normalized multichannel files, enabling safe partial-run resume.
- `source.scp`: absolute audio paths in the official `uid sample_rate path` format.
- `utt2spk` and `text`: selected speech metadata; missing transcripts remain absent.
- `records.jsonl`: original source metadata plus stable UID, materialized filename, original and output SHA-256, channel transform, output channels/rate/frames, and emitted speaker identity. Original `sha256`, `channels`, and `speaker_id` continue to describe the archived source.
- `request.json`: the input manifest hash, channel policy, and selection parameters for safe resume.
- `materialization.json`: completion summary, counts, sample rates, and unverified permission status.

Rerun the identical command to reuse files after checking their checksum and decoded metadata. Existing files are not rewritten. A different manifest, selection, role, or channel policy requires a different output directory. A failed extraction leaves already verified audio available for the next identical run, but does not publish a new SCP or completion summary. Do not launch concurrent commands against one output directory.

Tar contents are read in bounded blocks and never extracted by archive-provided paths. Traversal paths, links, special entries, missing selected members, duplicate selected members, and checksum mismatches fail visibly. The bridge retains selected metadata in memory; run one source/split at a time or explicitly partition very large source inventories. It reads each required archive once per invocation; resumed archives with no missing files are not opened again.

Pass the resulting `source.scp` files to `speech_scps`, `noise_scps`, `rir_scps`, and `wind_noise_scps` in the simulator YAML. Match each speech SCP with its `utt2spk`, and optionally its `text`. Corpus/source UIDs deliberately omit the split: this preserves cross-split identity for the final leakage audit. Speaker IDs are retained unchanged except for MLS: its raw numeric speaker IDs can collide between languages, so emitted mappings use `mls:<language>:<raw-speaker>`, with language read from `mls_<language>_<utterance>` source IDs. Original raw speaker metadata is retained separately from `materialized_speaker_id`. No split label is added to speaker namespaces, so a reused speaker still triggers the final cross-split audit. Other acquisition pipelines must namespace speakers consistently across corpora and retain the same ID across splits.

Materialization does not invent `commercial_use:true`, license evidence, or audited RIR direct-path delays. Supply those records separately under the preparation route's selected permission policy. Native-rate audio coverage, source split design, RIR timing, and the official simulator environment still need validation before rendering paired manifests. Run `prepare_data audit-splits` on the generated train/validation manifests before use.

Independently generated wind pools may restart their filename counters for each
split. For those pools only, explicit `--wind-id-scope split` namespaces the
generated UID by split while preserving the raw source ID and waveform hashes.
The release bridge selects this policy for the recorded SC-Wind-Noise-Generator
source. Identical train/validation waveform hashes still fail the leakage audit.
Recorded speech and noise retain global source identities.
