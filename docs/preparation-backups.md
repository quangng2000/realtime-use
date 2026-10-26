# Preparation backups and recovery

Preparation runs on local POSIX storage. Optional backups are immutable archives
on a separate persistent destination, including an object-backed filesystem mount.
The backup protocol does not depend on destination file locks, atomic rename,
or a mutable “latest” pointer. The destination must support `fsync` for durable
archive and receipt publication. It never starts training.

`PreparationBackup(output_root, backup_dir).checkpoint(unit, paths, identity_paths)`
is called only after a preparation unit has closed all its writers and passed its
checks. The release driver uses the logical units `root`, `materialized/<group>`,
`chunks/<split>/<number>`, and `final`. Backup errors propagate to the driver;
work must not advance past a failed checkpoint.

Each checkpoint:

1. Binds the original absolute output path and the root `request.json` checksum.
2. Computes a semantic identity from the supplied identity files. Materialization
   uses its request and verified records; rendered chunks use their saved plan,
   manifest, render journal, and generated metadata. Timings and reused/extracted
   counters are archived but do not determine whether a checkpoint is new.
3. Streams the closed files into a new uniquely named tar under `objects/`, with
   no complete tar staged on the local disk. It flushes and syncs the archive,
   then reads it back and checks its byte count and SHA-256.
4. Writes a uniquely named JSON receipt under `commits/` last and checks that it
   is synced and can be read back intact. The receipt includes its own checksum. A partial
   archive or truncated receipt is never a completed checkpoint.

An unchanged unit reuses its receipt only after verifying the saved archive
again. Backup contents are not silently replaced. Completed archives and receipts
remain immutable; obsolete or interrupted objects may occupy additional storage.
No automatic deletion or retention policy is applied.

The final receipt binds the receipts of every preceding unit. Its archive may
read local `READY.pending.json`, storing it under the final name `READY.json`.
The release driver can therefore publish local readiness only after the final
backup succeeds. If a later checkpoint advances a unit, recovery excludes the old
final receipt and does not restore its stale readiness marker.

## Throughput options

`--overlap-backups` allows closed chunk checkpoints to run while the next chunk
renders on local disk. `--backup-workers 1` (default) or `2` limits concurrent checkpoints; the queue
is bounded and final readiness waits for every result. Materialization checkpoints remain synchronous; use `--materialize-workers 1`
to avoid competing remote source reads. Interrupted work can require repeating an uncommitted chunk;
completed receipts remain valid.

`--backup-compression gzip` uses standard-library lossless gzip level 1 to reduce
transfer and storage size. It changes the checkpoint container only; audio bytes
restore exactly. Default `none` writes uncompressed tar. Both formats may coexist
in one backup, and changing compression does not change the frozen recipe or
force already verified checkpoints to be rewritten. Benchmark throughput on the
actual destination before choosing these runtime options.

## Restore

Stop the preparation driver before restoring. Use the same environment/runtime
versions and recreate the exact original absolute output path:

```bash
python -m realtime_use.preparation_backup restore \
  --backup-dir /persistent/preparation-backups/research-v1 \
  --output-root /opt/realtime-use-data/paired-research-v1
```

The command selects the latest complete receipt for each logical unit. It rejects
mixed recipe identities, a different absolute output root, corrupted committed
receipts, and archive checksum/size mismatches. It verifies every selected archive
before extracting any unit, then checks the extracted semantic identity files
against their receipts. Missing or corrupted committed archives are errors;
recovery does not silently fall back to older data.

Extraction uses local staging and accepts only confined regular files and
directories. Absolute paths, traversal, symlinks, hard links, and special files are
rejected. Existing output files must match exactly; conflicts are reported without
overwriting those files. Validated files are published using exclusive local hard
links, with `READY.json` published last. A repeated restore is safe when existing
files match. A failed restore may leave ordinary verified files, but never newly
publishes readiness before the remaining files have passed their checks.

After restoring, rerun the same preparation request. The existing resume checks
verify source/runtime identities, materialized audio, saved render metadata,
journals, output hashes, and loader behavior. Concurrency settings may change;
recipe settings and absolute paths may not. Frozen source archives, their release
indices, the pinned simulator checkout, and the preparation environment are
independent dependencies and must also be available at their original paths.
They are not implicitly included in the paired-output checkpoint.

## Capacity and interruption limits

The current release pipeline retains every materialized source and rendered pair.
Provision local capacity for those files, metadata, active worker scratch, and
free-space headroom. Streaming backups do not require another full local tar.
Restore stages the selected files locally; restoring into an empty output uses
one copy of their payload, while restoring over a populated output temporarily
requires room for a second staged copy. Prefer a fresh, empty local destination
at the original absolute output path.

Previously committed units survive loss of the CPU machine when their persistent
backup remains available. An unexpected machine loss can require repeating the
active, not-yet-committed chunks or source materialization group. Without
overlap, at most one chunk is active. With overlap, up to `backup-workers` closed
chunks and one currently rendering chunk may need to be repeated; each has up to
the configured 1,024 utterances. A planned shutdown should wait until a checkpoint boundary
has completed. The original source archives remain independently preserved.
