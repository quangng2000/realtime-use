# Data preparation and native-rate batches

The training loader consumes permitted, aligned mono pairs. It does not download datasets, estimate alignment, downmix, or silently resample them. Keep real audio and generated data outside Git. The optional official preparation adapter runs separately from training.

Downloaded source archives are not paired manifests. See [source materialization](../docs/source-materialization.md) for verified extraction and [release preparation](../docs/release-preparation.md) for the resumable source-release → official simulator → train/validation manifest workflow. These commands perform data preparation only; they never launch training.

## Paired manifests

Use one JSON object per line. Audio paths resolve relative to the manifest unless absolute:

```json
{"id":"corpusA:utterance1","noisy":"noisy/a.wav","clean":"clean/a.wav","sample_rate":16000,"transcript":"Optional reference transcript","language":"en"}
{"id":"corpusB:utterance2","noisy":"noisy/b.wav","clean":"clean/b.wav","sample_rate":48000,"language":"de"}
```

Every pair must have equal sample counts, actual/declared rates that agree, finite samples, and one channel. Set `model.sample_rates` to the permitted native rates; an empty list retains the single `model.sample_rate` behavior for old configurations. The paper-oriented configuration uses 8000, 16000, 22050, 24000, 32000, 44100, and 48000 Hz. A rate absent from the configured set is rejected.

Training crops are specified in seconds and converted separately for each example's rate. Paired streams share the exact crop and padding. Each collated batch has one integer `sample_rate`. Rate-bucket sampling builds global batches before sharding, so all DDP ranks process the same rate and equal local batch sizes on each step. All rates are visited; rare-rate buckets can repeat examples to make the number divisible by world size. This is data-driven sampling, not a promise of uniform rate frequencies. Lightning advances the sampler epoch for reproducible reshuffling.

Validation marks distributed padding duplicates invalid and counts each original example once in a complete pass. Set `data.validation_full_utterance=true` to avoid training-length validation crops. Older configurations without that field retain their historical crop behavior. Set a sufficient validation-batch limit for a complete pass; limits apply per rank. Full utterances at high rates may require validation batch size/memory planning.

Optional `teacher` paths must match the pair exactly. Only nonzero regression `loss.kd` enables their loading. Adversarial training ignores them. Additional random input augmentation is rejected with cached teacher targets because it would make those targets stale.

## Official URGENT preparation adapter

`python -m realtime_use.prepare_data urgent` integrates the [Apache-2.0 URGENT 2025 simulator](https://github.com/urgent-challenge/urgent2025_challenge/tree/daf1730cc11bf450d05c2d9e1d8bb3afdd63c427), pinned to `daf1730cc11bf450d05c2d9e1d8bb3afdd63c427`. The adapter verifies the imported source bytes against that Git revision. It delegates additive noise, reverberation, clipping, bandwidth limitation, codecs, packet loss, and nonlinear wind mixing to upstream functions. It does not run the upstream dataset-download scripts.

Use a separate preparation environment. A supported dependency candidate is Python 3.11/3.12, matching PyTorch/torchaudio 2.7.1, ESPnet 202509, resampy 0.4.3, and compatible FFmpeg 6 executable/shared libraries. Install the project with its `preparation` extra after the matching PyTorch/torchaudio build. Torchaudio's `AudioEffector` must initialize its FFmpeg extension; installing an executable alone is insufficient. Newer torchaudio versions removed this API. No GPU is needed for preparation.

```bash
python -m pip install -e '.[preparation,test]'
ffmpeg -version
python -c 'from torchaudio.utils import ffmpeg_utils; print(ffmpeg_utils.get_versions())'
git clone https://github.com/urgent-challenge/urgent2025_challenge.git /absolute/path/to/urgent-source
git -C /absolute/path/to/urgent-source checkout daf1730cc11bf450d05c2d9e1d8bb3afdd63c427
REALTIME_USE_URGENT_SOURCE=/absolute/path/to/urgent-source python -m pytest -q tests/test_urgent_simulator.py
```

The optional preflight uses generated tiny signals, exercises all seven corruption families, and verifies delayed anechoic targets. It never trains a model or downloads weights. It skips when the source path is unset; setting the path makes missing FFmpeg/dependencies fail visibly. The data-only and adapter tests run without the external checkout. On the development host, the official noise/reverb/clipping/band-limit/packet-loss paths and source-function hashes were checked with synthetic signals. Full codec/wind execution remains a preparation-host check because FFmpeg is absent locally.

ESPnet imports unrelated text tokenizers when importing its audio silence detector; those tokenizers can automatically download NLP assets. The adapter redirects only that import to two unchanged, Apache-2.0 ESPnet 202509 utility functions in `_vendor/espnet_detection/`. Their source commit, hashes and license are bundled. All other official simulation functions remain external; importing the adapter does not fetch assets.

### Local source inventories

Start from the pinned upstream `conf/simulation_validation.yaml` or `conf/simulation_train.yaml`, replacing corpus paths with your permitted local inventories. Each audio SCP contains exactly three whitespace-separated fields: a namespaced ID, actual sample rate, and absolute audio filename. Shell pipelines and filenames containing whitespace are unsupported by the official format.

```text
mycorpus:utterance1 16000 /absolute/path/to/clean1.wav
```

Provide `speech_scps`, `noise_scps`, `rir_scps`, and `wind_noise_scps`, plus one `speech_utt2spk` per speech SCP. Speaker mappings contain `utterance_id speaker_id`; namespace speaker IDs consistently across every split. Optional `speech_text` maps IDs to transcripts. Wind IDs must begin with `wind_noise`, as required by upstream dispatch. Inputs must already be mono. The simulator may resample noise/RIRs at higher rates to the selected speech rate; speech retains its declared native rate. Supply adequate noise/RIR/wind coverage at each speech rate, or a higher rate. Precomputed permitted wind recordings are acceptable; the pinned repository also contains `simulation/simulate_wind_noise.py` for its own wind-source generation.

Keep positive weights for all four additional augmentation entries (`clipping`, `bandwidth_limitation`, `codec`, `packet_loss`) and nonzero wind/reverb probabilities. The pinned generator's `prob_reverberation` condition is inverted: when the value is strictly between zero and one, actual RIR-selection probability is **1 minus the configured value**. The adapter records this behavior rather than silently rewriting the official generator. A value of 0.5 is unaffected. Seven enabled families do not guarantee all appear in a finite random sample; inspect the generated coverage counts.

### Anechoic targets and provenance

The official simulator normally constructs early-reflected targets. This adapter instead replaces its target RIR with a unit impulse at an **explicitly audited direct-path delay**, in samples at the rendering rate. Convolving with that impulse shifts otherwise anechoic clean speech; noisy reverberation still uses the full RIR. Clean, noisy, and noise share the official final gain normalization. Tails are truncated to the original length, following upstream. Direct-path delay is not inferred from a peak silently: supply and audit it for each RIR/rate that the simulator can select.

A provenance JSON accompanies the simulation YAML:

```json
{
  "split":"train",
  "sources":[
    {"id":"speech-corpus-v1","role":"speech","scp":"speech.scp","license":"Your actual license","license_evidence":"License document URL or reviewed local record","commercial_use":true},
    {"id":"noise-corpus-v1","role":"noise","scp":"noise.scp","license":"Your actual license","license_evidence":"License evidence","commercial_use":true},
    {"id":"rir-corpus-v1","role":"rir","scp":"rir.scp","license":"Your actual license","license_evidence":"License evidence","commercial_use":true},
    {"id":"wind-corpus-v1","role":"wind","scp":"wind.scp","license":"Your actual license","license_evidence":"License evidence","commercial_use":true}
  ],
  "rir_direct_path_samples":{"rir-corpus:room1":{"16000":37,"48000":111}}
}
```

These permission fields are the dataset owner's attestation, not automatic license verification. The simulator's code license does not grant rights to its listed corpora. The adapter rejects absent permission records and explicitly noncommercial licenses. Keep split construction and licenses in the project provenance ledger.

Those permission checks describe the default `--usage commercial` route. For an existing research dataset whose commercial clearance is pending, explicit `--usage research` preserves its licenses, evidence, and `commercial_use` values without asserting clearance. The research output remains labeled accordingly; preparing it does not change the rights attached to the source recordings.

When a research RIR corpus has no audited arrival labels, explicit `--rir-alignment upstream-estimated-onset` uses the pinned official `get_rir_start_sample` on the exact mono/resampled RIR used for convolution. It records the 10%-of-peak onset estimate, rendering rate, source/rendered RIR hashes, and `physical_direct_path_audited: false`. The target remains a delayed anechoic signal, but the delay is an estimate rather than a verified physical arrival. The default `audited` mode still requires the supplied direct-path map.

Inspect a plan first; this validates local inventories but does not render audio:

```bash
python -m realtime_use.prepare_data urgent \
  --source /absolute/path/to/urgent-source \
  --config /absolute/path/to/simulation.yaml \
  --provenance /absolute/path/to/provenance.json \
  --output-dir /absolute/path/to/prepared-train
```

Add `--execute` to that command only when ready to prepare the local inputs. This runs parameter generation and then rendering, never training. Existing output directories are rejected. Results include `manifest.jsonl`, generated float WAV pairs, official `log/meta.tsv`, resolved simulation YAML, provenance/source hashes, package versions, and realized degradation counts. Rendering is sequential; independent split/shard jobs can be run in distinct output directories. The manifest is published only after all pairs pass decoding, finite-sample, rate and length checks.

After interruption, add `--resume` with the identical inputs and settings. The adapter verifies the saved plan and generated metadata, then reuses only completed pairs whose hashes and decoded audio still validate. Incomplete pairs are rendered again. The final manifest is published only after the full chunk passes. Source-release preparation splits large speech inventories into such chunks and audits the combined train/validation manifests before publishing its readiness record.

Two narrow upstream wind fixes are recorded in the adapter: parse the literal clipping flag rather than treating the string `"False"` as true, and pass a copy of speech into the upstream in-place gain routine so it cannot alter the clean target. Float WAV output avoids extra PCM16 target quantization. These adaptations mean this is an official-simulator integration with anechoic targets, not byte-for-byte Challenge audio reproduction.

### Split audit

```bash
python -m realtime_use.prepare_data audit-splits \
  /absolute/path/to/prepared-train/manifest.jsonl \
  /absolute/path/to/prepared-validation/manifest.jsonl \
  /absolute/path/to/prepared-test/manifest.jsonl
```

The audit rejects cross-split reuse of source utterance IDs, speaker IDs, speech-file hashes, noise IDs and noise-file hashes. Repeated variants within one split are allowed. RIR reuse is recorded but not automatically prohibited. Namespaced identifiers and byte hashes cannot detect every transcoded duplicate; independently review corpus split design and speaker identity. Run this audit before using the resulting manifests.

## Additional training-only corruptions

Optional `audiomentations` Gaussian SNR, clipping and zero-phase low-pass transforms remain available under `data.augmentation.enabled=true`. They are disabled by default and can further corrupt already-degraded inputs. For native-rate data, the low-pass upper bound is limited to each rate's Nyquist frequency; invalid lower bounds still fail. These three transforms are not substitutes for the official seven-family preparation path and do not alter validation audio.

## Synthetic smoke data

Synthetic tones/noise exist only for software checks. Top-level `smoke=true` selects the tiny legacy model; `data.smoke=true` changes the dataset only. Neither produces a useful pretrained model or establishes quality. No training has to run to validate data preparation and batching.
