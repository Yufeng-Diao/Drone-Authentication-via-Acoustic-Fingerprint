# Live training monitor over localhost or SSH

The `monitor` command serves a read-only webpage for one `paper` sweep. It uses
Python's standard library and bundled HTML/CSS/JavaScript; no database, Node.js,
frontend build, CDN, telemetry service or extra Python dependency is required.
It runs independently of training and polls every three seconds. Closing the
browser or stopping the monitor leaves the training process alone.

This complements the terminal heartbeat and final [figure gallery](FIGURES.md).
It does not launch, stop or resume experiments. A single `run` command has no
sweep ledger; use `paper` for live monitoring.

## Start the page

Use the locked environment from [the quickstart](QUICKSTART.md), from the checkout
root. Choose the exact same run and log roots for the monitor and trainer:

```bash
uv run --locked drone-auth monitor \
  --run ../refactor_artifacts/runs/monitor-example \
  --log-root ../refactor_evidence/monitor-example/attempts \
  --port 8765
```

The monitor may start before training. It displays a waiting state without
creating the run directory. Open the complete printed URL, including its `#token`
fragment, in a browser on the same machine. Each server start generates a new
access token. Keep the URL private; a restart requires opening the new URL.
The page remembers the token only in that browser tab's session storage.

In a separate terminal, start a bounded example. `paper_slow_fit.json` uses
20 ms frames and the two SVM classifiers on the validation subset; it can take
considerably longer than the small GNB smoke example. It has no wall-time cutoff.

```bash
mkdir -p ../refactor_evidence/monitor-example
nohup uv run --locked drone-auth paper \
  --config config/validation/paper_slow_fit.json \
  --audio-root ../refactor_artifacts/data/acsac2022/extracted \
  --output ../refactor_artifacts/runs/monitor-example \
  --cache-root ../refactor_artifacts/cache/features \
  --log-root ../refactor_evidence/monitor-example/attempts \
  > ../refactor_evidence/monitor-example/trainer.log 2>&1 &
```

`nohup` keeps this noninteractive trainer separate from an SSH terminal hangup.
For interactive control, use a persistent terminal multiplexer if installed.
Starting the monitor alone does not detach a trainer launched elsewhere. If the
monitor also needs to survive disconnection, launch it with `nohup`, redirect its
output to an external log, and use `umask 077` before creating that log because
it contains the access URL. The ordinary foreground monitor exits on Ctrl-C.

## View a remote training host

Run the monitor on the training host. On your own computer, open another terminal
and keep this SSH connection open:

```bash
ssh -N -L 8765:127.0.0.1:8765 kelvin@YOUR_TRAINING_HOST
```

Then open `http://127.0.0.1:8765/#TOKEN_FROM_THE_MONITOR` in your local browser.
Replace the SSH host and token with the actual values. The monitor always binds
`127.0.0.1`; no public bind option or inbound firewall opening is needed. A local
forward such as `8876:127.0.0.1:8765` also works: use port 8876 in your browser
while keeping the token unchanged. VS Code's Ports view can forward the same
remote port instead of a separate SSH command.

The server permits only its fixed page/assets and authenticated read-only JSON
endpoints. It has no directory listing, artifact download or command-execution
endpoint. It rejects cross-origin requests and non-loopback Host names. Tokens
are not a substitute for SSH or host account security; do not publish this
small local server through a public reverse proxy.

## Read the display correctly

- **Task completion:** completed configurations / selected configurations. Fits
  have very different costs, so this is not percent of runtime or a finish ETA.
- **Current phase:** feature preparation, initialization, training, evaluation or
  serialization, using the existing `progress.json`. The phase timer measures
  wall time since that phase began. Pinning an older task does not change the
  live phase or process cards.
- **CPU and memory:** Linux process counters from the runner holding the existing
  sweep lock. CPU percent is sampled CPU time divided by elapsed time, where
  100% means one core. RSS is resident memory in MiB. Process CPU seconds and
  storage I/O counters are cumulative for that process. I/O is not SSD wear.
- **Activity history:** at most six minutes in the current browser session;
  reload clears it. Missing/disconnected samples are not filled in. Background
  browser tabs may throttle polling, so always read the last-snapshot timestamp.
- **Coverage and tasks:** selected presets/model counts, filters, searchable
  task rows and pages of 50. A task click pins its latest attempt. Follow current
  task returns to automatic tracking. Up to 50 attempt-history entries and the
  latest 32 KiB of the current attempt log are displayed; full logs stay on disk.
- **Failure and disconnects:** failed/interrupted states and errors remain visible.
  A disconnected page labels old data as stale. If the ledger says running but
  no local runner lock is visible, the page says **Runner not observed**, never
  silently claims that the model is still fitting. Investigate before resuming.

The monitor reads plan/ledger metadata, progress and bounded log tails. It does
not read WAVs, feature arrays, predictions or serialized models, and it does not
write to run/cache/audio/log directories. Large plans are parsed once and the
ledger is reparsed only when its file identity/mtime/size changes. The process
probe reads `/proc/locks` without acquiring the training lock. Completion here
reflects the ledger; [result export](FIGURES.md) performs artifact verification.

Run the monitor on the same Ubuntu host and local filesystem as training. This
allows Linux lock-owner/PID observation. Other hosts, restricted PID namespaces,
network filesystems or process-permission restrictions can hide liveness or
counters; an unavailable observation is not proof that the trainer crashed.

## P6 and container use

For P6, inspect `config/paper/full_sweep.json --dry-run` with the `paper` command,
then use a fresh full-run directory and point the monitor at that directory and
its attempt log root. Monitoring does not change any scientific configuration,
model limit or resume rule. This addition changes executable source identity;
old P4/P5 sweeps remain viewable, but their source-bound training resume should
not be forced across revisions. Start P6 from the final validated revision.

For Docker training, the [Compose workflow](CONTAINER.md) runs the monitor from
the same image as training; no host Python installation is required. Its monitor
service uses host networking and host PID visibility on Ubuntu, preserving the
loopback-only URL and Linux lock/PID observation. Run/log directories are mounted
read-only. No Docker socket or privileged container is needed. This topology
shares the host process view and is intended for a trusted research host; isolated
PID namespaces or restricted process permissions cannot provide the same liveness
observation. See the container guide for the tested platform and limitations.

The native host monitor remains supported as an alternative. Point it at the
host run/log bind directories; log basenames are resolved beneath the explicit
host log root. A fully isolated bridge-network monitor is not the supplied
configuration. Do not add a public bind address to work around forwarding.
