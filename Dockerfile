# syntax=docker/dockerfile:1.7

# =============================================================================
# Grimoire - Multi-GPU llama.cpp inference server
# =============================================================================

ARG CUDA_BASE=nvidia/cuda:12.8.1-devel-ubuntu22.04
ARG CUDA_RUNTIME=nvidia/cuda:12.8.1-runtime-ubuntu22.04
ARG GRIMOIRE_LLAMA_CPP_REPO_URL=https://github.com/TheTom/llama-cpp-turboquant.git
ARG GRIMOIRE_LLAMA_CPP_REF=feature-turboquant-kv-cache-b9079-69d8e4b

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

# Copy patches
COPY patches/ /app/patches/

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/.cache/llama-cpp-src \
    --mount=type=cache,target=/app/.cache/llama-cpp-build \
    set -eux; \
    if [ ! -d /app/.cache/llama-cpp-src/repo/.git ]; then \
        rm -rf /app/.cache/llama-cpp-src/repo; \
        git clone --depth 1 --branch "$GRIMOIRE_LLAMA_CPP_REF" --single-branch "$GRIMOIRE_LLAMA_CPP_REPO_URL" /app/.cache/llama-cpp-src/repo; \
    fi; \
    git -C /app/.cache/llama-cpp-src/repo remote set-url origin "$GRIMOIRE_LLAMA_CPP_REPO_URL"; \
    git -C /app/.cache/llama-cpp-src/repo fetch --depth 1 origin "$GRIMOIRE_LLAMA_CPP_REF"; \
    git -C /app/.cache/llama-cpp-src/repo reset --hard FETCH_HEAD; \
    git -C /app/.cache/llama-cpp-src/repo clean -fdx; \
    # Apply non-webui patches (webui patches are applied in the webui stage)
    for patch in /app/patches/*.patch; do \
        case "$(basename "$patch")" in grimoire-webui-*) continue ;; esac; \
        echo "Applying $patch"; \
        git -C /app/.cache/llama-cpp-src/repo apply "$patch"; \
    done; \
    cmake -S /app/.cache/llama-cpp-src/repo -B /app/.cache/llama-cpp-build \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_BUILD_EXAMPLES=OFF \
        -DGGML_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_TOOLS_INSTALL=ON \
        "-DCMAKE_CUDA_ARCHITECTURES=${GRIMOIRE_CMAKE_CUDA_ARCHITECTURES}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/.cache/llama-cpp-build --target llama-server --parallel $(nproc) --verbose \
    && cmake --install /app/.cache/llama-cpp-build --prefix /opt/model-a-llama-cpp


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

RUN --mount=type=cache,target=/cache/webui-src \
    --mount=type=cache,target=/root/.npm \
    set -eux; \
    if [ ! -d /cache/webui-src/repo/.git ]; then \
        rm -rf /cache/webui-src/repo; \
        git clone --depth 1 --branch "$GRIMOIRE_LLAMA_CPP_REF" --single-branch "$GRIMOIRE_LLAMA_CPP_REPO_URL" /cache/webui-src/repo; \
    fi; \
    git -C /cache/webui-src/repo remote set-url origin "$GRIMOIRE_LLAMA_CPP_REPO_URL"; \
    git -C /cache/webui-src/repo fetch --depth 1 origin "$GRIMOIRE_LLAMA_CPP_REF"; \
    git -C /cache/webui-src/repo reset --hard FETCH_HEAD; \
    git -C /cache/webui-src/repo clean -fdx -- tools/server/webui tools/server/public; \
    for patch in /src/patches/grimoire-webui-*.patch; do \
        [ -f "$patch" ] || continue; \
        echo "Applying webui patch: $patch"; \
        git -C /cache/webui-src/repo apply "$patch"; \
    done; \
    cp -r /cache/webui-src/repo/tools /src/tools; \
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
    LD_LIBRARY_PATH=/opt/model-a-llama-cpp/lib:/opt/model-a-llama-cpp/lib64 \
    PATH=/opt/grimoire-venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
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
COPY --from=build /opt/model-a-llama-cpp /opt/model-a-llama-cpp

# Copy built llama.cpp webui
COPY --from=webui /opt/grimoire-webui /opt/grimoire-webui

# Copy jinja chat templates (for huihui/super gemma variants)
COPY templates/ /templates/

# Create registry and state directories
RUN mkdir -p /etc/grimoire /var/lib/grimoire
COPY etc/models.json /etc/grimoire/models.json

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
    CMD curl -f http://localhost:9001/health || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "grimoire.entrypoint"]
