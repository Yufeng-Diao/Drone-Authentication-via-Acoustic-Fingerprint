# Docker workflow on Ubuntu

Use Docker for running the released experiments without installing Python on the
host. Keep the [locked native environment](QUICKSTART.md) for source development
and interactive debugging. Both entry points use the same code and `uv.lock`.
The supported container platform is **Ubuntu Docker Engine, Linux/amd64**.
Install Docker Engine and the Compose plugin using the
[official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
Docker Desktop is not required.

## Image availability and validation

A Docker Hub release is being prepared; no public image reference is advertised
until push and pull verification succeed. Build locally using the next section.
The image contains the CLI, frozen experiment configurations, the paper-comparison
helper, Poppler and portable Liberation fonts. It excludes datasets, PDFs,
credentials, private source, models, caches and experiment logs.

The Python base is pinned by OCI digest; Python packages use the existing lock.
Poppler and fonts are installed from Debian's repositories at build time. Cite the
**released image digest** for the complete binary environment; rebuilding the
Dockerfile later may pick up system-package updates. The image records its source
revision in `org.opencontainers.image.revision`.

Earlier P5 checks passed synthetic execution, one real-data GNB fit, three noise
evaluations and ordinary plotting, with matching native predictions and metrics.
On 18 October 2026, the expanded Compose workflow passed synthetic execution,
eight real-data classifier fits with exact native prediction/metric agreement,
completed-task reuse, live CPU/RSS/log observation, monitor restart/token rotation,
ordinary plotting and all 8,352 points of the full paper-style comparison. An
incomplete gallery was correctly rejected. Training and plot outputs stayed in
external bind directories; no host Python was needed by the container workflow. A native
full-paper run is documented in [the results report](REPRODUCTION_RESULTS.md);
this does not imply a full-paper container rerun.

## Prepare paths and the image

Run from the checkout root. Use absolute paths in the external env file. The
example contains no credentials and must never contain a Docker Hub password or
token. Paths containing spaces can be quoted in that file.

```bash
mkdir -p ../refactor_artifacts/deployment
cp docker.env.example ../refactor_artifacts/deployment/docker.env
export DRONE_ENV="$(realpath ../refactor_artifacts/deployment/docker.env)"
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
```

Edit `DRONE_ENV` with these settings:

| Variable | Meaning |
| --- | --- |
| `DRONE_IMAGE` | Local build tag, or published Docker Hub tag/digest |
| `AUDIO_DIR` | Existing extracted dataset containing `1to8`, `9to16`, `17to24`, `noise` |
| `RUNS_DIR` | Parent directory for saved sweeps and their trained models |
| `CACHE_DIR` | Shared binary feature cache |
| `LOGS_DIR` | Parent directory for separate attempt logs |
| `FIGURES_DIR` | Parent directory for ordinary saved-result galleries |
| `PAPER_DIR` | Directory containing the optional author-provided `ACSAC_Yufeng.pdf` |
| `COMPARISONS_DIR` | Parent directory for paper-style comparison reports |
| `PAPER_CONFIG` | Configuration path inside the image; starts with the validation subset |
| `RUN_NAME` | Sweep directory name under `RUNS_DIR` and `LOGS_DIR` |
| `GALLERY_NAME` | New gallery directory name under `FIGURES_DIR` |
| `COMPARISON_NAME` | New comparison directory name under `COMPARISONS_DIR` |
| `MONITOR_PORT` | Host loopback port, normally `8765` |

Create the six writable/optional parent directories using `mkdir -p` with the
absolute paths you entered: runs, cache, logs, figures, papers and comparisons.
Do not create a new empty audio directory in place of the downloaded dataset.
Keep audio, runs, cache, logs, figures and comparisons disjoint and outside source
checkouts. Compose deliberately refuses nonexistent bind sources instead of
silently creating root-owned directories. Use directory names without `..` or `/`
for `RUN_NAME`, `GALLERY_NAME` and `COMPARISON_NAME`.

If using a published image, set `DRONE_IMAGE` to its exact reference and run:

```bash
docker compose --env-file "$DRONE_ENV" pull
```

For a local build, leave `DRONE_IMAGE=drone-auth:local`:

```bash
docker build --platform linux/amd64 \
  --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t drone-auth:local .
```

Build from a committed tree for publication; use a clearly labelled working-tree
revision for local edits. No Compose service builds or pushes images implicitly.
Re-export `DRONE_ENV`, `LOCAL_UID` and `LOCAL_GID` in each new terminal.

## Check, start and watch an experiment

First validate configuration, check the installed CLI, and inspect the selected
plan. The default config is an eight-model validation subset, not a full rerun.

```bash
docker compose --env-file "$DRONE_ENV" config --quiet
docker compose --env-file "$DRONE_ENV" run --rm train --help
docker compose --env-file "$DRONE_ENV" run --rm train paper \
  --config /opt/drone/config/validation/paper_smoke.json --dry-run
docker compose --env-file "$DRONE_ENV" up -d monitor train
docker compose --env-file "$DRONE_ENV" logs monitor
```

Open the complete printed `http://127.0.0.1:8765/#TOKEN` URL. The monitor starts
independently and can wait for a not-yet-created sweep. Detached training continues
after SSH disconnects. For remote access, forward port 8765 in VS Code or run on
your local computer:

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@TRAINING_HOST
```

A fresh monitor process generates a fresh token. Keep its URL and container log
private. No public HTTP listener is enabled. The monitor uses host networking to
preserve the existing loopback binding and host PID visibility to observe the
trainer's file lock, CPU and memory. This is why there is no `ports:` mapping.
Only the monitor shares those host namespaces; training and plotting have no
network. It runs as your UID with dropped capabilities and read-only run/log
mounts, without `privileged` or Docker socket access. Host PID visibility is
broader than an isolated PID namespace; use this topology on a trusted Ubuntu
research host. Rootless Docker/user-namespace remapping is not covered by this
validation; restricted `/proc` permissions can hide process metrics.

```bash
docker compose --env-file "$DRONE_ENV" logs -f train
docker compose --env-file "$DRONE_ENV" ps -a
```

Use the live webpage for phase, task status, CPU/RSS and per-attempt logs. Detailed
logs persist at `LOGS_DIR/RUN_NAME/`; Docker's own console logs are size-limited.
A completed trainer should exit with code 0. The monitor remains running after
completion, until explicitly stopped. The page shows task completion, not a fit
percentage or a runtime estimate; some initialization/fitting phases are slow.

## Full paper run and custom configurations

After the smoke run, change these entries in the external env file:

```dotenv
PAPER_CONFIG=/opt/drone/config/paper/full_sweep.json
RUN_NAME=full-paper-001
GALLERY_NAME=full-paper-001
```

Inspect the full plan, then use the same `up -d monitor train` command:

```bash
docker compose --env-file "$DRONE_ENV" run --rm train paper \
  --config /opt/drone/config/paper/full_sweep.json --dry-run
docker compose --env-file "$DRONE_ENV" up -d monitor train
```

This runs 4,096 fits and 4,344 evaluations. The [README result mapping](../README.md#reproduce-the-paper-experiments)
explains each preset and the measured native storage/runtime. The supplied image
has all standard configurations. For an edited configuration, mount the entire
configuration directory read-only so that its relative manifest paths still work:

```bash
docker compose --env-file "$DRONE_ENV" run --rm \
  -v "$PWD/config:/study-config:ro" train paper \
  --config /study-config/paper/my_study.json \
  --audio-root /audio --output /runs/my-study-001 \
  --cache-root /cache --log-root /logs/my-study-001 --resume
```

Create `my_study.json` following the [README configuration example](../README.md#configure-an-experiment).
Set `RUN_NAME=my-study-001` before starting its monitor. Do not start the default
`train` service simultaneously with this one-off command. Use `-d` before `train`
if the one-off command must survive SSH disconnection. An image is a fixed code
snapshot; editing host Python files does not change its installed implementation.
Rebuild a new image for code changes and use a new sweep name.

## Interrupt and resume

```bash
docker compose --env-file "$DRONE_ENV" stop train
docker compose --env-file "$DRONE_ENV" up -d train
```

Compose sends SIGINT and allows two minutes before forcing termination. Native
fits may delay interruption; unfinished tasks restart from the beginning. The
training service always requests `--resume`, which accepts a fresh output or
validates and reuses matching completed tasks. It does not overwrite incompatible
results or migrate a sweep to different code, settings or dependencies. A normal
`restart train` after completion checks and reuses the completed sweep.

Use a new `RUN_NAME` for a different configuration. Keep the same image digest,
settings and source data when resuming. To stop the monitor and remove service
containers after use:

```bash
docker compose --env-file "$DRONE_ENV" down
```

All experiment data remains in the host bind directories. No generated data is
stored only in a disposable container layer. Changing env settings reconfigures
services in the same Compose project; for simultaneous experiments, use separate
`--project-name` values, run names and monitor ports.

## Plot saved results

After training completes:

```bash
docker compose --env-file "$DRONE_ENV" run --rm plot
```

Open `FIGURES_DIR/GALLERY_NAME/index.html`. Exports include Figures 2–5 and Tables
2–5 as applicable, PNG/SVG/PDF figures, Markdown/LaTeX/JSON tables, and source
metadata. Subset or incomplete results remain visibly labelled. Plotting mounts
runs read-only and requires neither the audio nor cache mounts. Keep the entire
sweep, including clean-model dependencies, when moving it.

For another export choose a fresh `GALLERY_NAME`; an existing gallery is never
overwritten. Multiple compatible runs can be supplied by overriding the command:

```bash
docker compose --env-file "$DRONE_ENV" run --rm plot plot \
  --paper-dir /opt/drone/config/paper --runs /runs/study-a /runs/study-b \
  --output /figures/combined-001
```

## Paper-style comparison

Set `GALLERY_NAME` to a completed full-data, full-grid gallery and put the matching
`ACSAC_Yufeng.pdf` in `PAPER_DIR`. Choose a fresh `COMPARISON_NAME`, then run:

```bash
docker compose --env-file "$DRONE_ENV" run --rm compare
```

Open `COMPARISONS_DIR/COMPARISON_NAME/index.html`. This exports the historical-style
curves, individual panels and side-by-side paper comparisons. All 8,352 figure
points are required; the smoke gallery is intentionally rejected. The exact PDF
checksum is checked, and the PDF is never packaged or published in the image.
Poppler and fonts are already installed inside the container. See
[figure interpretation](FIGURES.md) for the scientific and visual scope.

## Publish a validated image to Docker Hub

Authenticate interactively with `docker login` on your own terminal. Docker Hub
supports a browser/device-code flow; do not put passwords or access tokens in the
env file, Git, chat, shell command arguments or evidence logs. See
[Docker's login documentation](https://docs.docker.com/reference/cli/docker/login/).

After testing the exact candidate, tag and push that image rather than rebuilding
it between validation and publication. Replace the example namespace and release
with your own Docker Hub repository and a new, immutable release name:

```bash
docker tag drone-auth:local YOUR_NAMESPACE/drone-acoustic-fingerprint:RELEASE
docker push YOUR_NAMESPACE/drone-acoustic-fingerprint:RELEASE
docker pull YOUR_NAMESPACE/drone-acoustic-fingerprint:RELEASE
docker image inspect YOUR_NAMESPACE/drone-acoustic-fingerprint:RELEASE \
  --format '{{json .RepoDigests}}'
```

Record the returned repository digest, image ID, source revision, build log,
installed package inventory and validation results outside the checkout. Verify
the pulled image ID matches the validated candidate and run its `--help` or
synthetic check. Then publish the exact reference in README and this guide. Use
`namespace/repository@sha256:...` in `DRONE_IMAGE` for a fixed scientific baseline.
Do not silently replace an existing release tag. A `latest` tag is optional and
must not be the sole identifier for a reported experiment.

The supported image is CPU-only, Linux/amd64. Neither a fixed image nor a matching
lock guarantees bitwise model equality across all CPUs. Refreshing the base or
lock requires new recorded validation; additional platforms require their own
checks. Publishing the image does not push or merge the Git branch.
