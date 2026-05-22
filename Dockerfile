# syntax=docker/dockerfile:1.7

# =============================================================================
# Grimoire - Multi-GPU llama.cpp + DFlash inference server
# =============================================================================

ARG CUDA_BASE=nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG CUDA_RUNTIME=nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG GRIMOIRE_LLAMA_CPP_REPO_URL=https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant.git
ARG GRIMOIRE_LLAMA_CPP_REF=feature/turboquant-kv-cache
ARG GRIMOIRE_LLAMA_CPP_PINNED_SHA=0a635dcd92ba66c75fccfef91c3e106f4668f367
# Bump to force rebuild of the build stage (e.g. after upstream force-push)
ARG CACHE_BUST=10

# =============================================================================
# Build stage: Compile llama.cpp with CUDA + turbo4 cache + patches
# =============================================================================

FROM ${CUDA_BASE} AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ccache \
        git \
        ninja-build \
        pkg-config \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG GRIMOIRE_LLAMA_CPP_REPO_URL
ARG GRIMOIRE_LLAMA_CPP_REF
ARG GRIMOIRE_LLAMA_CPP_PINNED_SHA
ARG CACHE_BUST
ARG GRIMOIRE_CMAKE_CUDA_ARCHITECTURES=86;89

ENV CCACHE_DIR=/root/.ccache \
    CCACHE_COMPRESS=1 \
    CCACHE_MAXSIZE=5G

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/.cache/llama-cpp-src \
    --mount=type=cache,target=/app/.cache/llama-cpp-build \
    set -eux; \
    # If CACHE_BUST changed, invalidate the built marker so cmake re-runs
    cache_bust_file=/app/.cache/llama-cpp-build/.cache_bust; \
    if [ -f "$cache_bust_file" ]; then \
        old_bust=$(cat "$cache_bust_file"); \
        if [ "$old_bust" != "$CACHE_BUST" ]; then \
            echo "CACHE_BUST changed: $old_bust -> $CACHE_BUST, forcing rebuild"; \
            rm -f /app/.cache/llama-cpp-build/.built; \
        fi; \
    fi; \
    echo "$CACHE_BUST" > "$cache_bust_file"; \
    if [ ! -d /app/.cache/llama-cpp-src/repo/.git ]; then \
        rm -rf /app/.cache/llama-cpp-src/repo; \
        git clone --depth 1 --branch "$GRIMOIRE_LLAMA_CPP_REF" --single-branch "$GRIMOIRE_LLAMA_CPP_REPO_URL" /app/.cache/llama-cpp-src/repo; \
    else \
        old_ref=$(git -C /app/.cache/llama-cpp-src/repo rev-parse HEAD); \
        git -C /app/.cache/llama-cpp-src/repo remote set-url origin "$GRIMOIRE_LLAMA_CPP_REPO_URL"; \
        git -C /app/.cache/llama-cpp-src/repo fetch --depth 1 origin "$GRIMOIRE_LLAMA_CPP_REF"; \
        new_ref=$(git -C /app/.cache/llama-cpp-src/repo rev-parse FETCH_HEAD); \
    if [ "$old_ref" != "$new_ref" ]; then \
        git -C /app/.cache/llama-cpp-src/repo reset --hard FETCH_HEAD; \
        rm -f /app/.cache/llama-cpp-build/.built; \
    fi; \
fi; \
    current_sha=$(git -C /app/.cache/llama-cpp-src/repo rev-parse HEAD); \
    if [ "$current_sha" != "$GRIMOIRE_LLAMA_CPP_PINNED_SHA" ]; then \
        echo "ERROR: cloned SHA $current_sha != pinned $GRIMOIRE_LLAMA_CPP_PINNED_SHA"; \
        exit 1; \
    fi; \
    if [ ! -f /app/.cache/llama-cpp-build/.built ]; then \
        rm -f /app/.cache/llama-cpp-build/CMakeCache.txt; \
        cmake -S /app/.cache/llama-cpp-src/repo -B /app/.cache/llama-cpp-build \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FA=ON \
            -DGGML_NATIVE=OFF \
            -DGGML_BUILD_EXAMPLES=OFF \
            -DGGML_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_SERVER=ON \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_TOOLS_INSTALL=ON \
            "-DCMAKE_CUDA_ARCHITECTURES=${GRIMOIRE_CMAKE_CUDA_ARCHITECTURES}" \
            -DCMAKE_INSTALL_PREFIX=/opt/grimoire-llama-cpp \
            -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
            -DCMAKE_BUILD_TYPE=Release; \
        cmake --build /app/.cache/llama-cpp-build --target install --parallel $(nproc); \
        touch /app/.cache/llama-cpp-build/.built; \
    fi



# =============================================================================
# DFlash build stage: Compile the DFlash speculative decoding daemon
# =============================================================================

FROM ${CUDA_BASE} AS pflash-build

WORKDIR /app

COPY src/grimoire/pflash/ /app/pflash/

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake build-essential && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/.cache/pflash-build \
    set -eux; \
    cmake -B /app/.cache/pflash-build/build -S /app/pflash \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DPFLASH_FA_ALL_QUANTS=ON \
        -DPFLASH_ENABLE_BSA=ON; \
    cmake --build /app/.cache/pflash-build/build \
        --target pflash_daemon --parallel "$(nproc)"; \
    mkdir -p /opt/pflash; \
    cp /app/.cache/pflash-build/build/pflash_daemon /opt/pflash/pflash_daemon; \
    ccache --clear -q 2>/dev/null || rm -rf /root/.ccache/* 2>/dev/null; \
    find /app/.cache/pflash-build/build -name "libggml*.so*" -exec cp {} /opt/pflash/ \; 2>/dev/null || true; \
    ls -la /opt/pflash/

# Compile the park/unpark LD_PRELOAD shim
COPY src/grimoire/dflash/pflash_shim.c /app/pflash_shim.c
RUN gcc -shared -o /opt/pflash/pflash_shim.so -fPIC -I/usr/local/cuda/include \
    /app/pflash_shim.c -lcuda -ldl -Wall -Wextra 2>&1


# =============================================================================
# WebUI stage: Build the forked llama.cpp SvelteKit chat UI
# =============================================================================

FROM node:20-bookworm-slim AS webui

# Bump to force rebuild of the webui (e.g. after submodule update)
ARG WEBUI_BUST=1

WORKDIR /src/webui

COPY webui/ /src/webui/

RUN echo "webui-bust=${WEBUI_BUST}" && \
    VITE_PUBLIC_APP_NAME=chat.lost.plus npm ci && npm run build

RUN mkdir -p /opt/grimoire-webui && cp -r /src/webui/build/. /opt/grimoire-webui/


# =============================================================================
# Runtime stage: Lean CUDA runtime + Python + gateway
# =============================================================================

FROM ${CUDA_RUNTIME} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRIMOIRE_MODELS_DIR=/models \
    GRIMOIRE_REGISTRY_PATH=/var/lib/grimoire/models.json \
    GRIMOIRE_REGISTRY_SEED_PATH=/etc/grimoire/models.json \
    LD_LIBRARY_PATH=/opt/grimoire-llama-cpp/lib:/opt/grimoire-llama-cpp/lib64 \
    PATH=/opt/grimoire-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        python3.11 \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled llama-server
COPY --from=build /opt/grimoire-llama-cpp /opt/grimoire-llama-cpp

# Copy compiled pflash daemon
COPY --from=pflash-build /opt/pflash /opt/pflash

# Purge legacy directory name from older images
RUN rm -rf /opt/model-a-llama-cpp

# Copy built llama.cpp webui
COPY --from=webui /opt/grimoire-webui /opt/grimoire-webui

# Copy jinja chat templates (for huihui-gemma variant)
COPY templates/ /templates/

# Create registry and state directories
RUN mkdir -p /etc/grimoire /var/lib/grimoire
COPY etc/models.json /etc/grimoire/models.json

# Tokenizer files are mounted at runtime via /models volume (see compose)
# Tokenizers mounted at runtime via /models volume
# which resolves to /models/tokenizers/qwen3.6-27B via MODELS_DIR

# Install Python dependencies
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.11 -m venv /opt/grimoire-venv \
    && /opt/grimoire-venv/bin/pip install --upgrade pip \
    && /opt/grimoire-venv/bin/pip install .

# Expose gateway port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:9001/health

# Default entrypoint
ENTRYPOINT ["python", "-m", "grimoire.entrypoint"]
