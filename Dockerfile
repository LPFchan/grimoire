# syntax=docker/dockerfile:1.7

# =============================================================================
# Grimoire - Multi-GPU llama.cpp + DFlash inference server
# =============================================================================

ARG CUDA_BASE=nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG CUDA_RUNTIME=nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG GRIMOIRE_LLAMA_CPP_REPO_URL=https://github.com/TheTom/llama-cpp-turboquant.git
ARG GRIMOIRE_LLAMA_CPP_REF=feature-turboquant-kv-cache-b9079-69d8e4b
# Bump to force rebuild of the build stage (e.g. after upstream force-push)
ARG CACHE_BUST=1

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
ARG GRIMOIRE_CMAKE_CUDA_ARCHITECTURES=86;89

ENV CCACHE_DIR=/root/.ccache \
    CCACHE_COMPRESS=1 \
    CCACHE_MAXSIZE=5G

# Copy only non-webui patches for the build stage
RUN mkdir -p /app/patches
COPY patches/prefill-thinking-fix.patch /app/patches/

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/.cache/llama-cpp-src \
    --mount=type=cache,target=/app/.cache/llama-cpp-build \
    set -eux; \
    if [ ! -d /app/.cache/llama-cpp-src/repo/.git ]; then \
        rm -rf /app/.cache/llama-cpp-src/repo; \
        git clone --depth 1 --branch "$GRIMOIRE_LLAMA_CPP_REF" --single-branch "$GRIMOIRE_LLAMA_CPP_REPO_URL" /app/.cache/llama-cpp-src/repo; \
    else \
        old_ref=$(git -C /app/.cache/llama-cpp-src/repo rev-parse HEAD); \
        git -C /app/.cache/llama-cpp-src/repo remote set-url origin "$GRIMOIRE_LLAMA_CPP_REPO_URL"; \
        git -C /app/.cache/llama-cpp-src/repo fetch --depth 1 origin "$GRIMOIRE_LLAMA_CPP_REF"; \
        new_ref=$(git -C /app/.cache/llama-cpp-src/repo rev-parse FETCH_HEAD); \
        if [ "$old_ref" != "$new_ref" ] || [ ! -f /app/.cache/llama-cpp-build/.patched ]; then \
            git -C /app/.cache/llama-cpp-src/repo reset --hard FETCH_HEAD; \
            git -C /app/.cache/llama-cpp-src/repo clean -fdx; \
            for patch in /app/patches/*.patch; do \
                [ -f "$patch" ] || continue; \
                echo "Applying $patch"; \
                git -C /app/.cache/llama-cpp-src/repo apply "$patch"; \
            done; \
            rm -f /app/.cache/llama-cpp-build/.built /app/.cache/llama-cpp-build/.patched; \
            touch /app/.cache/llama-cpp-build/.patched; \
        fi; \
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

FROM ${CUDA_BASE} AS dflash-build

WORKDIR /app

COPY dflash/ /app/dflash-hub

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates cmake build-essential && rm -rf /var/lib/apt/lists/*

# test_dflash is the daemon entrypoint upstream ships under the test tree;
# pflash_daemon doesn't exist as a standalone target, so we build and rename
# test_dflash to /opt/dflash/dflash at install.
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/.cache/dflash-build \
    set -eux; \
    cd /app/dflash-hub/dflash; \
    rm -f .git && git init && git add -A && git -c user.name=build -c user.email=build commit -qm snapshot; \
    git submodule update --init --recursive; \
    cmake -B /app/.cache/dflash-build/build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DDFLASH27B_TESTS=ON \
        -DDFLASH27B_FA_ALL_QUANTS=ON \
        -DDFLASH27B_ENABLE_BSA=ON; \
    cmake --build /app/.cache/dflash-build/build \
        --target test_dflash --parallel "$(nproc)"; \
    mkdir -p /opt/dflash; \
    cp /app/.cache/dflash-build/build/test_dflash /opt/dflash/dflash; \
    cp -r /app/.cache/dflash-build/build/lib/* /opt/dflash/ 2>/dev/null || true


# =============================================================================
# WebUI stage: Build the stock llama.cpp SvelteKit chat UI
# =============================================================================

FROM node:20-bookworm-slim AS webui

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG GRIMOIRE_LLAMA_CPP_REPO_URL
ARG GRIMOIRE_LLAMA_CPP_REF

WORKDIR /src

# Webui-only patches (e.g. grimoire-webui-history.patch which swaps the upstream
# Dexie/IndexedDB DatabaseService for fetch calls to grimoire's /history endpoints).
COPY patches/ /src/patches/

# Dashboard page is owned as a standalone file, not a patch.
COPY dashboard/ /src/dashboard/

RUN --mount=type=cache,target=/cache/webui-src \
    --mount=type=cache,target=/root/.npm \
    set -eux; \
    if [ ! -d /cache/webui-src/repo/.git ]; then \
        rm -rf /cache/webui-src/repo; \
        git clone --depth 1 --branch "$GRIMOIRE_LLAMA_CPP_REF" --single-branch "$GRIMOIRE_LLAMA_CPP_REPO_URL" /cache/webui-src/repo; \
    else \
        old_ref=$(git -C /cache/webui-src/repo rev-parse HEAD); \
        git -C /cache/webui-src/repo remote set-url origin "$GRIMOIRE_LLAMA_CPP_REPO_URL"; \
        git -C /cache/webui-src/repo fetch --depth 1 origin "$GRIMOIRE_LLAMA_CPP_REF"; \
        new_ref=$(git -C /cache/webui-src/repo rev-parse FETCH_HEAD); \
        if [ "$old_ref" != "$new_ref" ] || [ ! -f /cache/webui-src/.patched ]; then \
            git -C /cache/webui-src/repo reset --hard FETCH_HEAD; \
            git -C /cache/webui-src/repo clean -fdx -- tools/server/webui tools/server/public; \
            for patch in /src/patches/grimoire-webui-*.patch; do \
                [ -f "$patch" ] || continue; \
                echo "Applying webui patch: $patch"; \
                git -C /cache/webui-src/repo apply "$patch"; \
            done; \
            rm -f /cache/webui-src/.built; \
            touch /cache/webui-src/.patched; \
        fi; \
    fi; \
    cp -r /cache/webui-src/repo/tools /src/tools; \
    mkdir -p /src/tools/server/webui/src/routes/dashboard; \
    cp /src/dashboard/* /src/tools/server/webui/src/routes/dashboard/; \
    cd /src/tools/server/webui; \
    npm ci; \
    npm run build; \
    mkdir -p /opt/grimoire-webui; \
    cp -r /src/tools/server/public/. /opt/grimoire-webui/


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

# Copy compiled dflash daemon
COPY --from=dflash-build /opt/dflash /opt/dflash

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
# No COPY needed — the dflash model config points to 'tokenizers/qwen3.6-27B'
# which resolves to /models/tokenizers/qwen3.6-27B via MODELS_DIR

# Install Python dependencies
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN python3.11 -m venv /opt/grimoire-venv \
    && /opt/grimoire-venv/bin/pip install --upgrade pip \
    && /opt/grimoire-venv/bin/pip install .

# Expose gateway port
EXPOSE 9001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:9001/health

# Default entrypoint
ENTRYPOINT ["python", "-m", "grimoire.entrypoint"]
