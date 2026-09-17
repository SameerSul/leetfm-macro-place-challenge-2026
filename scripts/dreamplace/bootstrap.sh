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
LEGACY_LOCAL_PATCH_COMMIT="b1d1f4bdae57c78f9eb4b4c3a3974adc6f44ccb6"
LOCAL_PATCH_COMMIT="4c64c3f49eca86ccf5d5a050c92e030352cc8d62"
PRIOR_LOCAL_PATCH_COMMIT="495ad92aaaae72f9255ce394afde9d66aa4851e2"
CURRENT_LOCAL_PATCH_COMMIT="7558a3915c924616f7c61204cf3e41a409eed3f8"
HETEROSTA_SHA256="51b2c757993ee4b8eeaf0fb217c8ba144ca57aa6827d59f21bcba56e38419f77"
ACTION="${1:-all}"

source_patches() {
    local patch entries rel _added _deleted
    for patch in cuda12-cub.patch runtime-fixes.patch; do
        entries="$(git -C "$SOURCE_DIR" apply --numstat "$ROOT/scripts/dreamplace/$patch")"
        while read -r _added _deleted rel; do
            if git -C "$SOURCE_DIR" apply --reverse --check --include="$rel" \
                "$ROOT/scripts/dreamplace/$patch" 2>/dev/null; then
                continue
            fi
            if [[ "${1:-apply}" == check ]]; then
                echo "DREAMPlace source patch is missing or stale: $patch ($rel)" >&2
                exit 2
            fi
            git -C "$SOURCE_DIR" apply --check --include="$rel" "$ROOT/scripts/dreamplace/$patch"
            git -C "$SOURCE_DIR" apply --include="$rel" "$ROOT/scripts/dreamplace/$patch"
        done <<< "$entries"
    done
}

ensure_source() {
    mkdir -p "$SOURCE_DIR"
    # A directory inside this repository must not inherit the parent Git tree.
    if [[ ! -e "$SOURCE_DIR/.git" ]]; then
        git -C "$SOURCE_DIR" init
        git -C "$SOURCE_DIR" remote add origin "$UPSTREAM"
    fi
    if ! git -C "$SOURCE_DIR" cat-file -e "$UPSTREAM_COMMIT^{commit}" 2>/dev/null; then
        git -C "$SOURCE_DIR" fetch --depth 1 origin "$UPSTREAM_COMMIT"
    fi

    local head
    head="$(git -C "$SOURCE_DIR" rev-parse --verify HEAD 2>/dev/null || true)"
    if [[ -z "$head" ]]; then
        git -C "$SOURCE_DIR" checkout --detach "$UPSTREAM_COMMIT"
        head="$UPSTREAM_COMMIT"
    fi
    if [[ "$head" != "$UPSTREAM_COMMIT" \
        && "$head" != "$LEGACY_LOCAL_PATCH_COMMIT" \
        && "$head" != "$LOCAL_PATCH_COMMIT" \
        && "$head" != "$PRIOR_LOCAL_PATCH_COMMIT" \
        && "$head" != "$CURRENT_LOCAL_PATCH_COMMIT" ]]; then
        echo "Unexpected DREAMPlace revision: $head" >&2
        echo "Expected $UPSTREAM_COMMIT (upstream), $LEGACY_LOCAL_PATCH_COMMIT (legacy local)," >&2
        echo "or a recorded local patched tree through $CURRENT_LOCAL_PATCH_COMMIT." >&2
        exit 2
    fi

    source_patches
    "${PYTHON:-python3}" "$ROOT/scripts/dreamplace/apply_visualizer_patch.py" \
        "$SOURCE_DIR/dreamplace/NonLinearPlace.py"
    git -C "$SOURCE_DIR" submodule update --init --recursive
}

ensure_environments() {
    mkdir -p "$BUILD_ROOT"
    if [[ ! -x "$TOOLCHAIN/bin/cmake" || ! -x "$TOOLCHAIN/bin/nvcc" \
        || ! -x "$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" ]]; then
        local micromamba
        micromamba="${MICROMAMBA:-$(command -v micromamba || true)}"
        if [[ -z "$micromamba" ]]; then
            echo "micromamba is required to create the pinned DREAMPlace toolchain." >&2
            exit 2
        fi
        "$micromamba" --root-prefix "$MAMBA_ROOT" create -y -p "$TOOLCHAIN" \
            -f "$ROOT/scripts/dreamplace/environment-linux-64.lock"
    fi
    "${PYTHON:-python3}" - "$ROOT" "$BUILD_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.dreamplace.preflight import verify_toolchain
verify_toolchain(Path(sys.argv[2]))
PY

    command -v uv >/dev/null 2>&1 || {
        echo "uv is required to install the pinned DREAMPlace Python environment." >&2
        exit 2
    }
    if [[ ! -x "$PYTHON_ENV/bin/python" ]]; then
        uv venv --python 3.10.20 "$PYTHON_ENV"
    fi
    # Reruns also complete interrupted installs. Every transitive dependency is pinned.
    uv pip install --python "$PYTHON_ENV/bin/python" --no-deps \
        --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1+cu121
    uv pip install --python "$PYTHON_ENV/bin/python" --no-deps \
        -r "$ROOT/scripts/dreamplace/requirements.txt"
    uv pip check --python "$PYTHON_ENV/bin/python"
}

build_dreamplace() {
    local python torch_cmake cuda_arch native_libs
    python="$PYTHON_ENV/bin/python"
    torch_cmake="$("$python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    native_libs="$("$python" -c 'from pathlib import Path; import torch; p = Path(torch.__file__).parent; print(":".join(map(str, [p / "lib", *sorted((p.parent / "nvidia").glob("*/lib"))])))')"
    cuda_arch="${DREAMPLACE_CUDA_ARCH:-8.9}"
    mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

    PATH="$TOOLCHAIN/bin:$PATH" \
    CC="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" \
    CXX="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" \
    CUDAHOSTCXX="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" \
    CUDACXX="$TOOLCHAIN/bin/nvcc" \
    CUDA_HOME="$TOOLCHAIN" \
    "$TOOLCHAIN/bin/cmake" -S "$SOURCE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MAKE_PROGRAM="$TOOLCHAIN/bin/make" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$native_libs:$TOOLCHAIN/lib" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,$native_libs:$TOOLCHAIN/lib" \
        -DCMAKE_DISABLE_FIND_PACKAGE_GUROBI=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_CPLEX=ON \
        -DCMAKE_DISABLE_FIND_PACKAGE_LPSOLVE=ON \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DCMAKE_PREFIX_PATH="$torch_cmake;$TOOLCHAIN" \
        -DPython_EXECUTABLE="$python" \
        -DPYTHON_EXECUTABLE="$python" \
        -DCMAKE_C_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" \
        -DCMAKE_CXX_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" \
        -DCMAKE_CUDA_COMPILER="$TOOLCHAIN/bin/nvcc" \
        -DCUDA_TOOLKIT_ROOT_DIR="$TOOLCHAIN" \
        -DCUDA_TOOLKIT_INCLUDE="$TOOLCHAIN/targets/x86_64-linux/include" \
        -DBOOST_ROOT="$TOOLCHAIN" -DBoost_NO_SYSTEM_PATHS=ON \
        -DCMAKE_CXX_ABI=0 \
        -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
    printf '%s  %s\n' "$HETEROSTA_SHA256" \
        "$BUILD_DIR/thirdparty/HeteroSTA/heterosta.v1.1.20251211.tar.gz" | sha256sum --check
    "$TOOLCHAIN/bin/cmake" --build "$BUILD_DIR" --parallel "${DREAMPLACE_BUILD_JOBS:-3}"
    "$TOOLCHAIN/bin/cmake" --install "$BUILD_DIR"
    "$python" "$ROOT/scripts/patch_dreamplace_install.py" --install-dir "$INSTALL_DIR"
    "$python" "$ROOT/scripts/dreamplace/apply_visualizer_patch.py" \
        "$INSTALL_DIR/dreamplace/NonLinearPlace.py"
}

run_preflight() {
    source_patches check
    "${PYTHON:-python3}" "$ROOT/scripts/dreamplace/apply_visualizer_patch.py" --check \
        "$SOURCE_DIR/dreamplace/NonLinearPlace.py" "$INSTALL_DIR/dreamplace/NonLinearPlace.py"
    "${PYTHON:-python3}" "$ROOT/scripts/patch_dreamplace_install.py" \
        --install-dir "$INSTALL_DIR" --check
    "${PYTHON:-python3}" "$ROOT/scripts/dreamplace/preflight.py" \
        --build-root "$BUILD_ROOT" --source-dir "$SOURCE_DIR"
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
