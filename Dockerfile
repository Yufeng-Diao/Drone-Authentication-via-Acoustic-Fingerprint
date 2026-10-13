# CPU-only Python runtime; digest resolved from Docker Hub on 2026-10-13.
FROM python:3.14.8-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH="" \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/drone/venv UV_CACHE_DIR=/tmp/uv-cache
RUN python -m pip install --no-cache-dir uv==0.13.0
WORKDIR /opt/drone/source
COPY pyproject.toml uv.lock .python-version README.md LICENSE LICENSE_sklearn_QDA ./
COPY src/ src/
COPY toolbox/ toolbox/
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python3.14 \
    && rm -rf /tmp/uv-cache
COPY config/ /opt/drone/config/
ENV PATH="/opt/drone/venv/bin:$PATH"
USER 10001:10001
WORKDIR /work
ENTRYPOINT ["drone-auth"]
CMD ["--help"]
