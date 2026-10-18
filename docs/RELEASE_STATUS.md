# Full experiment and release status

The full public-data sweep and local P6 acceptance are complete on Ubuntu using
the validated scientific implementation at revision `62f6d2e`. All 8,440 tasks
completed, including 4,096 fits and 4,344 evaluations reusing clean models.
See [the measured results and scientific differences](REPRODUCTION_RESULTS.md).
This is a complete corrected-protocol rerun, not exact recovery of every
historical score. Remote CI, push and publication have not been performed.

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
Remaining publication actions are remote CI, push and release review on the host
repository when authorized. Local tests are not represented as remote CI results.

## Maintained and historical code

The supported execution path is `src/drone_fingerprint/`, with the attributed
MFCC core in `toolbox/mfcc_base.py` and `toolbox/mfcc_sigproc.py`. Researchers can
follow the module map and examples in [the development guide](DEVELOPMENT.md).

`experiment/`, `runners/` and the remaining historical toolbox wrappers are kept
for comparison with the original release. Their original imports, dependencies
and generated-directory conventions are not the modern workflow. No compatibility
with those entry points is implied by modern CLI tests. Use the explicitly labelled
[historical guide](LEGACY_WORKFLOW.md) only when investigating the old implementation.

The paper's alternating within-recording train/test split is retained. Results
do not establish recording-held-out, unseen-room or adversarial replay robustness.
Original plots have no preserved source CSV; curve comparisons must identify any
visual interpretation rather than invent pointwise reference measurements.


## Preserving the original public implementation

The local `legacy/acsac2022` branch points to the pre-refactor public commit
[`88d0098b189b8cb48da3ec05ab81f7890b36d6b5`](https://github.com/Yufeng-Diao/Drone-Authentication-via-Acoustic-Fingerprint/tree/88d0098b189b8cb48da3ec05ab81f7890b36d6b5).
It preserves the original public tree and history, not the private development
repository. Keep this branch frozen and make future maintenance changes on the
maintained line. The permanent commit link remains usable independently of branch
publication. No claim is made that the historical runtime has been reconstructed.

On 14 October 2026, a read-only remote check confirmed that `origin/main` still
points to this original commit and that the legacy branch is not yet published.
The maintained work remains on `refactor/reproducible-pipeline`; `main` has not
been moved. Before publishing the transition:

1. Publish `legacy/acsac2022` and verify its remote commit matches the pinned hash.
2. Refresh the remote state, review the maintained branch and run remote CI.
   Resolve any newer upstream changes without rewriting the historical snapshot.
3. Merge the maintained branch into `main` with history preserved. Avoid a squash
   or rebase merge: scientific records reference intermediate validation commits.
   A fast-forward is also suitable if the reviewed `main` remains an ancestor.
4. Update the README's publication status and add the public legacy-branch link.
   Record the maintained release commit so users can cite their actual version.

These are publication steps, not actions already performed. The two source
worktrees share Git history, so preserving the legacy branch does not require a
new repository or copying code. The current original worktree has `main` checked
out; a later update to that worktree will display the maintained code after the
merge. Use the pinned commit or the legacy branch to inspect the original version.

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
separate from the pending GitHub branch publication and remote CI. No full-paper
container training or additional architecture validation is implied.
