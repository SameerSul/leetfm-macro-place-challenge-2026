#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_DIR="${DREAMPLACE_SOURCE_DIR:-$ROOT/dreamplace_src}"
BUILD_ROOT="${DREAMPLACE_BUILD_ROOT:-$ROOT/dreamplace_build}"
INSTALL_DIR="$BUILD_ROOT/install"
BUILD_DIR="$BUILD_ROOT/cmake-build"
PYTHON_ENV="$BUILD_ROOT/dpenv"
MAMBA_ROOT="$BUILD_ROOT/mamba"
TOOLCHAIN="$MAMBA_ROOT/envs/dptool"
UPSTREAM="https://github.com/limbo018/DREAMPlace.git"
UPSTREAM_COMMIT="37214b40fe3837cc7d392c7d6092ccd6ff04a02c"
LOCAL_PATCH_COMMIT="b1d1f4bdae57c78f9eb4b4c3a3974adc6f44ccb6"
ACTION="${1:-all}"

ensure_source() {
    mkdir -p "$SOURCE_DIR"
    if ! git -C "$SOURCE_DIR" rev-parse --git-dir >/dev/null 2>&1; then
        git -C "$SOURCE_DIR" init
        git -C "$SOURCE_DIR" remote add origin "$UPSTREAM"
    fi
    if ! git -C "$SOURCE_DIR" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
        git -C "$SOURCE_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
    fi

    local head
    head="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ -z "$head" ]]; then
        git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_COMMIT"
        head="$UPSTREAM_COMMIT"
    fi
    if [[ "$head" != "$UPSTREAM_COMMIT" && "$head" != "$LOCAL_PATCH_COMMIT" ]]; then
        echo "Unexpected DREAMPlace revision: $head" >&2
        echo "Expected $UPSTREAM_COMMIT (upstream) or $LOCAL_PATCH_COMMIT (local patched tree)." >&2
        exit 2
    fi

    if ! grep -q "CUB 2.x (CUDA 12.1)" \
        "$SOURCE_DIR/dreamplace/ops/utility/src/utils_cub.cuh"; then
        git -C "$SOURCE_DIR" apply "$ROOT/scripts/dreamplace/cuda12-cub.patch"
    fi
    git -C "$SOURCE_DIR" submodule update --init --recursive
}

ensure_environments() {
    mkdir -p "$BUILD_ROOT"
    if [[ ! -x "$TOOLCHAIN/bin/cmake" ]]; then
        local micromamba
        micromamba="${MICROMAMBA:-$(command -v micromamba || true)}"
        if [[ -z "$micromamba" ]]; then
            echo "micromamba is required to create the pinned DREAMPlace toolchain." >&2
            exit 2
        fi
        "$micromamba" --root-prefix "$MAMBA_ROOT" create -y \
            -f "$ROOT/scripts/dreamplace/environment.yml"
    fi

    if [[ ! -x "$PYTHON_ENV/bin/python" ]]; then
        command -v uv >/dev/null 2>&1 || {
            echo "uv is required to create the pinned DREAMPlace Python environment." >&2
            exit 2
        }
        uv venv --python 3.10 "$PYTHON_ENV"
        uv pip install --python "$PYTHON_ENV/bin/python" \
            --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
        uv pip install --python "$PYTHON_ENV/bin/python" \
            -r "$ROOT/scripts/dreamplace/requirements.txt"
    fi
}

build_dreamplace() {
    local python torch_cmake cuda_arch
    python="$PYTHON_ENV/bin/python"
    torch_cmake="$("$python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cuda_arch="${DREAMPLACE_CUDA_ARCH:-8.9}"
    mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

    CC="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-cc" \
    CXX="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-c++" \
    CUDAHOSTCXX="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-c++" \
    CUDACXX="$TOOLCHAIN/bin/nvcc" \
    CUDA_HOME="$TOOLCHAIN" \
    "$TOOLCHAIN/bin/cmake" -S "$SOURCE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$torch_cmake;$TOOLCHAIN" \
        -DPYTHON_EXECUTABLE="$python" \
        -DCMAKE_CXX_ABI=0 \
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
    "$TOOLCHAIN/bin/cmake" --build "$BUILD_DIR" --parallel "${DREAMPLACE_BUILD_JOBS:-3}"
    "$TOOLCHAIN/bin/cmake" --install "$BUILD_DIR"
    "$python" "$ROOT/scripts/patch_dreamplace_install.py"
}

run_preflight() {
    "${PYTHON:-python3}" "$ROOT/scripts/dreamplace/preflight.py"
}

case "$ACTION" in
    source) ensure_source ;;
    env) ensure_environments ;;
    build) ensure_source; ensure_environments; build_dreamplace ;;
    preflight) run_preflight ;;
    all) ensure_source; ensure_environments; build_dreamplace; run_preflight ;;
    *)
        echo "Usage: $0 [source|env|build|preflight|all]" >&2
        exit 2
        ;;
esac
