# CPU container for an Ubuntu host

**Validation status:** build and runtime smoke checks passed on an Ubuntu 24.04
Linux/amd64 host with Docker Engine 29.9.0 on 2026-10-13. The previously deferred
P5 container gate is closed. The unchanged Dockerfile passed a synthetic run,
one real-data GNB fit, three noise evaluations and saved-result plotting.
Predictions, labels, sample intervals and metrics match the native P5 subset
results exactly for all four real-data tasks. This is a bounded smoke check,
not full-paper or all-model container reproduction.

The image pins Python 3.14.8's Debian Bookworm slim base by OCI digest, installs
uv 0.13.0 and uses `uv sync --locked --no-dev --no-editable`. Ubuntu is the host
target; the container userland is Debian. No GPU, audio device or network service
is required at runtime. The build needs registry/package access. The context
allowlist excludes datasets, caches, results and logs. The default user is
unprivileged; match the host UID/GID for writable bind mounts.

Run from the checkout on a Linux/amd64 host:

```bash
docker build --platform linux/amd64 -t drone-auth:local .
mkdir -p ../refactor_artifacts/container-runs ../refactor_artifacts/container-cache
mkdir -p ../refactor_evidence/container-logs ../refactor_evidence/container-figures
docker run --rm --network none --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(realpath ../refactor_artifacts/container-runs),dst=/runs" \
  drone-auth:local synthetic --output /runs/synthetic-001
```

Check exit status zero, `/runs/synthetic-001` completion and prediction artifacts.
Then check public-data execution and plotting with separate writable roots.
This selection runs one clean Table 2 GNB fit and three Figure 5 evaluations
at -8, 0 and 15 dB, reusing the clean model:

```bash
docker run --rm --network none --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(realpath ../refactor_artifacts/data/acsac2022/extracted),dst=/audio,readonly" \
  --mount "type=bind,src=$(realpath ../refactor_artifacts/container-runs),dst=/runs" \
  --mount "type=bind,src=$(realpath ../refactor_artifacts/container-cache),dst=/cache" \
  --mount "type=bind,src=$(realpath ../refactor_evidence/container-logs),dst=/logs" \
  drone-auth:local paper --config /opt/drone/config/validation/paper_representative.json \
  --preset figure5 --model GNB --audio-root /audio --output /runs/paper-001 \
  --cache-root /cache --log-root /logs/paper-001
docker run --rm --network none --read-only --tmpfs /tmp \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$(realpath ../refactor_artifacts/container-runs/paper-001),dst=/saved-sweep,readonly" \
  --mount "type=bind,src=$(realpath ../refactor_evidence/container-figures),dst=/figures" \
  drone-auth:local plot --paper-dir /opt/drone/config/paper \
  --runs /saved-sweep --output /figures/paper-001
```

Preserve build output, image digest, command exits and the gallery as external
evidence. Use new output names on repetition. Figures must say incomplete
preview. Raw audio is mounted read-only; plotting needs only the saved sweep.
The changed plot mount path also checks that the complete sweep is portable.
New sweeps use relative clean-model references and can move together. Historical
P4 sweeps contain absolute paths; preserve those paths when mounting them.

The image fixes dependencies and base identity; it does not promise bitwise
identical model files across CPUs. No multi-architecture, GPU or Windows
container behavior has been tested. Refreshing the base or lock requires a new
native/container smoke run and recorded numerical comparisons.
