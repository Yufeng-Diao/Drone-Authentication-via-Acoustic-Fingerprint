# Indexed audio, bounded extraction and shared feature caches (P3)

The maintained Ubuntu workflow uses an external recording index and immutable
NumPy arrays. It keeps P2's float64 scientific conventions while avoiding full
recording decoding, per-frame files and repeated extraction for every model.
See [NUMERICAL_VALIDATION.md](NUMERICAL_VALIDATION.md) for the feature definitions.

## Run with an explicit cache

```bash
uv run --locked drone-auth run \
  --config config/validation/open_set_trial_01.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --cache-root ../refactor_artifacts/cache/features \
  --output ../refactor_artifacts/runs/open-set-001
```

Without `--cache-root`, the runner uses `feature-cache` beside the output
folder. Use a dedicated directory: this application owns its cache files and
reserved `.building-*` staging directories. Cache, run outputs and raw audio
must be separate; caches and outputs are rejected inside source projects.
Runs retain the existing three result/model/prediction files. `--save-features`
still adds the optional P2-compatible NPZ validation snapshot, which duplicates
selected features intentionally and is unnecessary for ordinary cached runs.

To validate the complete published inventory and build the recording index:

```bash
uv run --locked drone-auth index \
  --manifest config/paper/recordings.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --cache-root ../refactor_artifacts/cache/features
```

An inventory build performs one directory walk and fails on missing or extra
WAVs. Each recording is hashed and its power measured with bounded buffers.
Normal experiment runs select through the manifest and perform no directory
walk. They lazily index selected recordings if a full inventory was not built.

## Input verification and its tradeoff

The index stores relative recording paths, the verified SHA-256, PCM payload
offset, frame count, full-recording signal power and a filesystem signature:
device, inode, byte size, nanosecond modification time and change time. The
frozen recording/subset manifests continue to provide identity, pairing, dates,
distances and interval selection. A modified filesystem signature triggers
rehashing and WAV format validation; a changed checksum fails explicitly.
An unchanged signature permits reuse without rereading the WAV. This is an
explicit performance policy for local immutable research datasets, not a claim
that a stat check is a fresh cryptographic content verification.

Pass `--verify-inputs` to force SHA-256 verification even when the signature is
unchanged. File signatures are checked again after extraction. The index has a
content digest to detect accidental metadata corruption. That digest is not a
signature against malicious local modification. Source audio remains read-only.
The independent P0 full-dataset validator remains available for comprehensive
inventory audits.

PCM16 squares are summed in bounded int64 blocks with a Python integer total,
then divided once by the complete recording length. The sum is exact for PCM16;
its floating-point result agrees with the P2 float64 reference within the stated
1e-12 numerical tolerance. Room-noise training can read only its declared
interval after verification instead of converting the entire WAV to float64.

## Identity and label metadata

The recording manifest joins by `identity` to
[`identities.json`](../config/paper/identities.json), transcribed from Section 3
and Table 1 (printed page 661). Bodies A through H occur three times: identities
1–8 with their original a–h propeller sets, 9–16 with the shared spare set x1–x4,
and 17–24 with the shared spare set y1–y4. Thus 24 identities represent eight
physical bodies in different combinations. Room noise has null identity;
classifier label 0 additionally represents background training and rejected
attacks, so it must never replace the original identity in provenance. Keeping
this relation separate avoids changing the frozen audio manifest and its hashes.

## Cache layout and reuse

```text
features/
  .lock
  recordings-<audio-root-key>.json
  <base-key>/
    features.npy
    metadata.json
  <selected-key>/
    features.npy
    metadata.json
```

Train and test splits have separate entries. For a new configuration this is
normally four entry directories: base train/test and selected train/test, each
with two files, plus one recording index and one advisory lock. Files aggregate
all selected intervals; there are no per-recording or per-frame cache files.
More requested configurations naturally create more entries. The cache is
persistent and has no automatic eviction; remove the dedicated cache directory
when it is no longer needed and no run is using it. Raw WAVs are never cached as
additional audio files.

Base entries retain unnormalized MFCCs, including coefficient 0. Their keys bind
recording content and selection, exact segment boundaries and label provenance,
frame/filter settings, noise seed/SNR when applicable, numerical implementation
hashes and NumPy/SciPy/scikit-learn versions. Selected entries additionally bind
derivative blocks, retained coefficients and normalization. Estimator choice is
excluded. Different models load the same read-only selected arrays, and another
coefficient fraction or block count can reuse the raw MFCC arrays. Partition
changes that alter selected intervals or labels intentionally use other keys.
Training keys exclude test SNR, so noisy evaluations reuse clean training data.

Every cache hit verifies metadata, shape, dtype, byte size and the array's
SHA-256 before opening a read-only memory map. Newly published arrays have
already been hashed by the writer under the same lock, so their immediate
first opening does not repeat that full-file read. Valid hits do not rewrite cache
files. Corrupt entries fail with their path; remove that specific entry and run
again to rebuild it. Do not repair metadata to make a damaged array appear valid.
Removing the entire cache is also safe when no processes are using it.

## Bounded processing without changing the experiment

`--batch-frames` accepts 1 through 256, default 32. Each batch reads only its audio
window and retains the previous window overlap without rereading it. It retains one preceding sample for pre-emphasis, preserves
the original frame origin and hop rounding, and pads only the final frame of a
continuous segment. Feature selection runs in 256-row blocks with a two-frame
context on each side, sufficient for two recursive N=1 derivatives. Real
segment boundaries still use edge replication. No context crosses a split.
Changing the extraction batch size is an execution choice, not a feature/cache
identity change; tests compare multiple sizes against the whole-segment P2
reference with absolute and relative tolerance 1e-12.

The noisy reader consumes PCG64 draws in original sample order, including draws
for skipped intervals, and retains overlap between adjacent read requests.
This preserves the P2 noise sample at every position. Power still refers to the
complete recording; empirical SNR is accumulated over the complete noise
stream. No full noisy recording is retained or written. A requested SNR remains
one task at a time; full sweep orchestration and clean-model reuse are P4.

The runner reports row counts and the array-byte estimate before allocation and
checks available space before creating each cache array. Default 32-frame batches
bound feature extraction scratch space; very long frames need more memory per
batch, so reduce `--batch-frames` on smaller machines. Arrays are float64 and
loaded via read-only mmap. **Estimator fitting can still copy inputs or allocate
substantial working memory.** P3 does not impose a general memory cap on sklearn.

Numerical work runs on the main thread with a one-thread BLAS/OpenMP limit using
[threadpoolctl](https://github.com/joblib/threadpoolctl). RF and KNN also explicitly
use one worker. The existing heartbeat thread only reports elapsed time. These
execution controls avoid nested parallelism; classifier formulas, seeds and
iteration budgets remain unchanged. Full eight-model convergence testing is P4.

## Atomic writes and interrupted work

Writers hold a single Ubuntu `flock` lock for cache preparation. Another process
waits, then validates and reuses the published entry. One writer constructs an
array and metadata in a staging directory, flushes them once, and atomically
renames the directory after validation. There is no fsync per frame. A process
killed during writing leaves only staging data, which is removed while holding
the lock on the next preparation. It cannot count as a cache hit. Completed base
entries can survive a later interrupted selection step and be reused.

Run completion JSON and recording-index updates also use temporary files and
atomic replacement. `result.json` is published after model/prediction writes;
checksums detect missing or damaged artifacts before reuse. Failed runs retain
their failure records. `--resume` still reuses only matching complete tasks; it
does not checkpoint or continue an interrupted estimator fit. P4 handles task
restart orchestration. Cache and output durability assumes normal local Ubuntu
filesystem semantics; remote/network filesystems have not been tested.

## Reproduce the resource comparison

Use the same locked numerical stack for both revisions, with P2 installed in an
independent environment. Keep the P2 interpreter's virtualenv path intact; do not
resolve its executable symlink to the shared base interpreter.

```bash
uv run --locked python -I -B tools/benchmark_pipeline.py \
  --baseline-python ../refactor_artifacts/environments/p3-baseline314/bin/python \
  --candidate-python ../refactor_artifacts/environments/p3-py314/bin/python \
  --config config/validation/open_set_trial_01.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_evidence/pipeline-benchmark-001 \
  --artifacts ../refactor_artifacts/runs/pipeline-benchmark-001
```

The tool runs one warm-up plus three measured repetitions for P2, P3 with empty
application caches, and P3 with populated application caches. It records exact
commands, exit codes, RSS traces, process peak RSS, CPU/wall time, logical reads
and writes, process storage I/O, system swap counters, cache file hashes/mtimes,
and output bytes/file counts. Runtime artifacts remain separate from logs.
A valid populated cache must be unchanged. Every run retains feature snapshots
for independent comparison; correctness must be checked before interpreting
performance. The machine's OS page cache is uncontrolled: none of these runs is
a physical cold-disk benchmark. System swap activity is not attributable to one
process, and process storage counters do not measure SSD wear or diagnose past
hardware failure. First-time cache construction adds writes and can be slower;
report that cost alongside repeated-use savings.

The direct code path is `data.py` (PCM/index facts), `features.py` (numerics),
`noise.py` (deterministic noise), `pipeline.py` (cache and shared arrays), then
`workflow.py` (model and results). `storage.py` contains only external-path and
atomic-JSON helpers. No database, new array format or decoded-audio cache is used.
