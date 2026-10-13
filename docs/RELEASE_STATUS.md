# Full experiment and release status

The full public-data sweep is running on Ubuntu using the validated implementation
at revision `62f6d2e`. This is an ongoing corrected rerun, not a completed paper
reproduction. The configured grid contains 4,096 fits and 4,344 evaluations reusing
clean models. Missing and failed tasks remain visible; task count is not an ETA.

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

These criteria are pending until the complete results and final review exist.
The existing regression suite, real-data subset checks, browser monitoring and
container smoke checks validate the workflow; they do not replace the full run.

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
