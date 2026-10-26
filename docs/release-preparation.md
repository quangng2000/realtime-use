# Prepare a downloaded source release

`python -m realtime_use.prepare_release` connects a finalized source-audio release
to the existing paired-data loader. It never starts training, downloads corpora,
or downloads model weights. Run it in the pinned preparation environment described
in [data/README.md](../data/README.md), using a pristine official URGENT checkout.

The input root must contain `release/RELEASE.json`, its checksummed combined
speech/noise/wind train and validation indices, the RIR pool index, source archive
shards, and verified `sources/<corpus>/SOURCE_READY.json` markers. A partial release
is accepted only if every required role and split is present; missing sources are
retained in the output report. “All” means every speech record in that finalized
release, not all corpora listed in the paper. Finish source acquisition and release
finalization before starting this command; its inputs are frozen by hash.
Validation likewise contains all released validation speech. It is not the
Challenge's separately selected fixed evaluation subset.

Inspect the plan without writing output:

```bash
python -m realtime_use.prepare_release \
  --root /workspace-global/realtime-use/urgent2025-research-v1 \
  --source /opt/urgent-pristine \
  --output-dir /scratch/realtime-use/paired-research-v1
```

Add `--execute` to prepare audio. All released speech is selected by default, with
one degradation realization per utterance. `--chunk-size` defaults to 1024 and
bounds each official parameter-generation/rendering job. Repeating the identical
command resumes verified materialization and rendering; changed requests, input
hashes or existing conflicting files are rejected. Use a new output directory for
a changed recipe. The original source archives and download processes are untouched.

`--workers N` runs the official per-utterance renderer in parallel within each
chunk. `--materialize-workers N` extracts independent source/split groups in
parallel; each group has its own output directory. Both default to one. Worker
counts are execution settings recorded in `execution.json`, separate from the
frozen data recipe. They can change when resuming the same output. Chunk size,
seeds, source hashes, selection limits, and other recipe changes still require a
new output. Materialization uses spawned processes with one CPU thread per
worker and at most N groups in flight. An extraction failure stops those workers
and leaves verified files available for a subsequent resume.

Choose counts using a representative benchmark on the preparation host: extra
processes can saturate storage or compete with ongoing downloads. Validation
chunks still run before training chunks, and validation RIR selection remains
sequential across chunks. Parallel work preserves source order and metadata
selection; it does not run any model training.

Parallel preparation owns its process pool from the main thread. SIGTERM waits
for renderer cleanup (or terminates outstanding materializers) before releasing
the preparation lock. Restart only after the old process and its children exit.
Do not launch multiple release drivers against the same output directory.

Use a filesystem with POSIX locking and atomic rename for active preparation
outputs. Runpod documents that its object-backed Global Volumes lack those
guarantees and cannot attach to CPU pods; a successful filesystem call alone does
not establish that guarantee. Prefer local disk or a same-region network volume
for active preparation, retaining verified archives on persistent storage.
See [Runpod's Global Volume limitations](https://docs.runpod.io/storage/globalvolume/overview#limitations).
Checksummed resume verifies saved records and audio, but cannot make an
object-backed mount provide unsupported filesystem semantics.

For local preparation with persistent recovery, add
`--backup-dir /backup/realtime-use/paired-research-v1`. The backup directory must
be outside the active output tree, on storage that survives loss of the compute
instance. Its location is recorded in `execution.json`, outside the frozen
request, so it can change when resuming. The pipeline first checkpoints its
request and partitioned indices, then each closed materialization group and each
verified rendered chunk before proceeding by default. With `--overlap-backups`,
closed chunk checkpoints overlap the next local render. `--backup-workers 1`
(default) or `2` bounds concurrent checkpoint I/O; the queue is drained before
final audits and readiness. `--backup-compression gzip` optionally
uses lossless gzip level 1 for backup containers. Both options are execution
settings and preserve sampled audio and recipe identity. A backup failure stops preparation,
cleans up active materialization workers, and prevents publication of `READY.json`.

Checkpoints stream immutable tar archives directly to the backup destination,
flush them durably, verify their full contents by checksum, and publish a synced
receipt last. They do not
need a second local tar copy. Runtime timings and reused-file counts do not change
checkpoint identity. The final checkpoint includes manifests, audits,
`data-paths.yaml`, and the staged readiness marker; local readiness is published
only after that final backup succeeds. The backup destination may be a network
volume or another mounted persistent filesystem; storage capacity and mount
durability must be checked separately.

To recover after local output loss, restore to the **same absolute output path**:

```bash
python -m realtime_use.preparation_backup restore \
  --backup-dir /backup/realtime-use/paired-research-v1 \
  --output-root /scratch/realtime-use/paired-research-v1
```

Then rerun the original preparation command with `--backup-dir`. A partial backup
restores completed units without advertising readiness; remaining units resume.
The source release, original archive shards, and pinned upstream checkout are
external dependencies and are **not** included in these preparation checkpoints.
Retain them separately at their recorded paths. Do not reclaim local completed
groups or chunks while the release driver is running; this option provides
recovery checkpoints, not a rolling storage cache.

For a deliberately limited readiness check, add `--max-speech-per-split 16` and use
a separate output directory. The report labels this an `explicit_readiness_subset`;
it never implies that the complete corpus was prepared.

The bridge extracts each source/split once, verifies each selected archived audio
member's hash and decoded samples, and emits official SCP, speaker and transcript
files. Multichannel recordings explicitly retain their first channel as mono float
WAV, with the transformation recorded. Source-index hashes and archive sizes are
checked during planning; the materializer verifies the selected audio rather than
rehashing entire archive files an extra time.
An early audit checks source utterances, normalized speaker identities, and
materialized audio hashes across splits before rendering. Validation renders
before the larger training split so validation failures surface first.
The documented synthetic wind generator restarts recording indices per split.
Only those generated wind IDs receive an explicit split namespace. Other noise
IDs remain global, and waveform hashes are still compared across all noise and
wind splits, so renaming generated IDs cannot admit duplicated audio.

Official simulation YAML parameters remain unchanged except local inventory/output
paths, one realization per utterance, and chunk seeds: the official base seed plus
the chunk index. Chunking therefore changes the exact random sequence relative to
one monolithic official run. Training RIR reuse follows the upstream template.
Validation RIR reuse stays disabled, including across chunks: already selected
RIR IDs are excluded from later chunks. A new chunk fails if a selected rate has
no remaining eligible RIR. Within a chunk, exhaustion follows the official
generator's behavior of omitting reverberation; inspect realized coverage counts.
`--validation-reuse-rir` is an explicit recipe change and requires a
different output directory. RIRs may still be shared across train and validation,
as in the configured source pool; speech, speakers and noise undergo split audits.

This route is explicitly for **research with commercial clearance pending**.
It retains per-record and source license evidence and never manufactures a
`commercial_use: true` attestation. Anechoic targets use the pinned official RIR
onset estimator (first 10%-of-peak crossing); this is documented as an estimate,
not an audited physical direct-path measurement. It is not a claim of exact paper
dataset reproduction or commercial permission.

`progress.json` reports materialization group, rendering split/chunk, and completed
utterance counts. After every train and validation chunk is verified, the command
combines manifests with unique chunk-qualified IDs, audits cross-split identities
and hashes, then publishes:

- `manifests/train.jsonl` and `manifests/validation.jsonl`
- `split-audit.json`
- `data-paths.yaml`, containing the Hydra `data.train_manifest` and
  `data.validation_manifest` paths
- `READY.json`, the final readiness marker with counts, manifest hashes, usage
  limitations and `training_started: false`

On resume, readiness moves to `last-successful-ready.json` before revalidation or
repair. That file is historical evidence only. A new `READY.json` appears after
all checks pass again.

Use the paths from `data-paths.yaml` with `experiment=paper_flexible12` when training
is separately authorized. A readiness marker establishes data-pipeline readiness;
it does not validate CUDA/DDP training, metric assets, model quality or latency.
