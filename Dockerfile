## MassKit - Multi-stage Docker image
##
## Build:   docker build -t masskit .
## Run CLI: docker run --rm -v $(pwd)/data:/data masskit info /data/sample.mzML
## Python:  docker run -it --rm masskit python

# ── Stage 1: build C++ core ──────────────────────────────────────────
FROM debian:bookworm-slim AS cpp-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpugixml-dev \
    zlib1g-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY core/ ./core/
RUN cmake -B core/build -S core -DCMAKE_BUILD_TYPE=Release \
    && cmake --build core/build --config Release -j$(nproc)

# ── Stage 2: Python runtime ──────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="MassKit"
LABEL org.opencontainers.image.description="LC-MS Data Analysis Toolkit"
LABEL org.opencontainers.image.source="https://github.com/lcms-toolkit/lcms-toolkit"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpugixml1v5 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r masskit && useradd -r -g masskit -m masskit

WORKDIR /app

# Copy built C++ artifacts (optional - Python pkg works without)
COPY --from=cpp-builder /build/core/build/liblcms_core* /usr/local/lib/

# Install Python package
COPY python/setup.py python/README* /app/python/
COPY python/masskit /app/python/masskit
COPY README.md /app/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /app/python \
    && pip install --no-cache-dir matplotlib pandas

USER masskit
WORKDIR /home/masskit

# Default command shows CLI help
ENTRYPOINT ["masskit"]
CMD ["--help"]
