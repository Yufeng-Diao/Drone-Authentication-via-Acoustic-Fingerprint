# Full experiment and release status

The full public-data sweep and local P6 acceptance are complete on Ubuntu using
the validated scientific implementation at revision `62f6d2e`. All 8,440 tasks
completed, including 4,096 fits and 4,344 evaluations reusing clean models.
See [the measured results and scientific differences](REPRODUCTION_RESULTS.md).
This is a complete corrected-protocol rerun, not exact recovery of every
historical score. The maintained workflow is published on `main`, which remains
the default branch. The original version is preserved on the public
[`legacy/acsac2022` branch](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/legacy/acsac2022).

Both Ubuntu interpreter jobs passed [remote CI](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/actions/runs/39155417609)
on revision `7554b95dd72014ce0cdf231ae6975bc093b0c34d`, covering regression tests,
model-default auditing, installed-package synthetic execution and paper-plan
expansion. CI does not download the full audio dataset or repeat the full sweep.
The initial remote attempt exposed an invalid job-level `runner.temp` reference;
the workflow now sets external runtime paths after the runner starts. This fix
changes CI setup only, not the scientific implementation or published Docker image.

Use [the live monitor](LIVE_MONITOR.md) to observe a sweep locally or through SSH.
A full run can outlive an interactive session. Keep the trainer detached, preserve
its exact command, and use a fixed non-editable environment and frozen input
configuration throughout the run. Do not update packages or source inside that
environment while experiments are running. Keep outputs, logs and evidence
outside the checkout. A monitor restart does not resume a failed trainer.

## Release completion criteria

- All 8,440 planned tasks have verified current completions on the full selected
  dataset, including all ten open-set partitions for every model.
- Saved-result export verifies checksums and predictions and produces Figures
  2–5, Tables 2–5 and diagnostics, with all missing coverage explicitly reported.
- Printed paper values and new measurements are compared in percentage points;
  corrected definitions, convergence warnings and runtime changes remain visible.
  Table 5 macro and pooled rejection are reported separately.
- Runtime, peak memory, storage and file counts are measured. Process I/O does
  not establish physical SSD wear or explain historical hardware failure.
- Reproducibility checks, researcher-facing review and the final release review
  are complete. Local checks and remote CI are reported separately.

The local criteria above have been checked, including complete artifact hashes,
eight full-data classifier repeats and controlled open-set label-order diagnostics.
Remote validation and branch publication complement those local results; their
scopes remain distinct. Future changes should be developed on a topic branch,
checked in CI, and merged into `main` with commit history preserved.

## Maintained and historical code

The supported execution path is `src/drone_fingerprint/`, with the attributed
MFCC core in `toolbox/mfcc_base.py` and `toolbox/mfcc_sigproc.py`. Researchers can
follow the module map and examples in [the development guide](DEVELOPMENT.md).

Legacy experiment scripts, runners, PKL generators, unused toolbox wrappers,
YAML configurations, the Conda environment, tutorial screenshots and empty
output placeholders have been removed from the maintained tree. They remain
available in the frozen original branch. No compatibility with those entry points
is implied by modern CLI tests. Use the [historical guide](LEGACY_WORKFLOW.md)
when investigating the old implementation. The MFCC core, classifier definitions,
paper manifests and dependency lock are unchanged by this source-tree cleanup.

The paper's alternating within-recording train/test split is retained. Results
do not establish recording-held-out, unseen-room or adversarial replay robustness.
Original plots have no preserved source CSV; curve comparisons must identify any
visual interpretation rather than invent pointwise reference measurements.


## Preserving the original public implementation

The public [`legacy/acsac2022` branch](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/legacy/acsac2022)
points to the pre-refactor public commit
[`88d0098b189b8cb48da3ec05ab81f7890b36d6b5`](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5).
It preserves the original public tree and history, not the private development
repository. Keep this branch frozen and make maintenance changes on `main` through
a reviewed topic branch. The permanent commit link identifies the snapshot even
if branch settings change. Historical runtime reconstruction is not implied.

The original branch was published and its remote SHA verified before the
maintained branch was advanced to `main`. The transition used fast-forward
integration, preserving all original and refactoring commits. Do not squash or
rebase the scientific history: earlier validation reports reference those commits.

The paper's old `Eidos1970` repository URL currently redirects to the repository
under `Yufeng-Diao` and displays its default branch. Use the current repository
URL in new documentation. The default branch remains `main`; explicit links to
`legacy/acsac2022` show the original version.

## Local worktrees

The original public checkout stays on `legacy/acsac2022`, retaining its original
files. The maintained checkout uses `main`. They share Git history and remote
configuration; changing a branch's files does not require a second repository.

## Report links and generated evidence

The README links to version-controlled English reports so those links work in
fresh clones. Generated HTML comparisons, figures, logs and large experiment
artifacts remain outside source checkouts. They have no public hosting URL yet.
If publishing these bundles separately as release assets or a report site, add
links only after the public destination exists and has been checked. Do not put
local workspace paths in the public README or commit runtime evidence to make
those paths appear to work.


## Docker Hub release — 18 October 2026

The validated Linux/amd64 image is publicly available as
[`eidos1970/drone-acoustic-fingerprint:2026.10.18`](https://hub.docker.com/r/eidos1970/drone-acoustic-fingerprint).
Its source revision is `7aa6c2a468796bb01a9a9b0b21a2e8fef6e6778e`.
The [container guide](CONTAINER.md#image-availability-and-validation) records its
immutable digest and checks. Anonymous registry access returned the validated
image; an execution check uses the digest directly. Docker Hub publication is
separate from GitHub branch publication and remote CI. No full-paper
container training or additional architecture validation is implied.
