#!/usr/bin/env bash
# this_file: build.sh
#
# Full verification gate: Rust workspace (lib, CLI, extension), both Python
# distributions, and a smoke test in a throwaway virtual environment. Leaves the
# checkout byte-identical — a build that modifies tracked files is a failure.
#
# Nothing here needs model weights. Checkpoint-dependent tests skip themselves
# when `ref/` is empty (`cargo test`) or when `transformers` is absent
# (`tests/test_parity.py`).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LOG="$ROOT/build.log.txt"
TEMP_DIR=""
BUILD_COMPLETED=0

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}

finish() {
    local status=$?
    local message
    trap - EXIT
    if (( status == 0 && BUILD_COMPLETED == 1 )); then
        message='BUILD SUCCESS: the crate, the extension module, bananamendy and the WebAssembly build passed.'
    else
        (( status == 0 )) && status=1
        message="BUILD FAILURE: exit status $status; inspect $LOG"
    fi
    printf '\n%s\n' "$message" >>"$LOG" || status=1
    printf '\n%s\n' "$message"
    cleanup
    exit "$status"
}

handle_signal() {
    trap - INT TERM
    exit "$1"
}

require_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'Required tool is missing: %s\n' "$1" >&2
        return 1
    }
}

verify_layout() {
    local required
    for required in \
        Cargo.toml \
        crates/bananamendr/Cargo.toml \
        crates/bananamendr/src/lib.rs \
        crates/bananamendr/src/main.rs \
        crates/bananamendr-py/Cargo.toml \
        crates/bananamendr-py/pyproject.toml \
        crates/bananamendr-wasm/Cargo.toml \
        crates/bananamendr-wasm/src/lib.rs \
        bananamendy/pyproject.toml \
        bananamendy/tests \
        tests/wasm_parity.mjs \
        wasm.sh \
        scripts/release-manifests.py \
        scripts/check-python-artifacts.py; do
        [[ -e "$ROOT/$required" ]] || {
            printf 'Canonical project path is missing: %s\n' "$required" >&2
            return 1
        }
    done
}

record_tree() {
    local destination=$1
    git -C "$ROOT" status --porcelain=v1 --untracked-files=all >"$destination.status"
    git -C "$ROOT" diff --no-ext-diff --binary HEAD -- >"$destination.diff"
}

verify_tree_unchanged() {
    local before=$1 after=$2
    if ! cmp -s "$before.status" "$after.status" || ! cmp -s "$before.diff" "$after.diff"; then
        printf 'Build changed the checkout. Before/after status follows.\n' >&2
        diff -u "$before.status" "$after.status" >&2 || true
        return 1
    fi
}

python_gate() {
    local expected=$1
    local dist="$TEMP_DIR/python-dist"
    local venv="$TEMP_DIR/python-venv"
    local python="$venv/bin/python"

    mkdir -p "$dist/bananamendr" "$dist/bananamendy"
    (cd "$ROOT/crates/bananamendr-py" && uv build --no-sources --out-dir "$dist/bananamendr")
    (cd "$ROOT/bananamendy" && uv build --no-sources --out-dir "$dist/bananamendy")
    python3 "$ROOT/scripts/check-python-artifacts.py" "$dist/bananamendr" "$expected" \
        >"$dist/bananamendr.txt"
    python3 "$ROOT/scripts/check-python-artifacts.py" "$dist/bananamendy" "$expected" \
        >"$dist/bananamendy.txt"
    local rust_wheel python_wheel
    rust_wheel="$(sed -n '1p' "$dist/bananamendr.txt")"
    python_wheel="$(sed -n '1p' "$dist/bananamendy.txt")"

    uv venv --clear --python python3 "$venv"
    uv pip install --python "$python" "$rust_wheel" "$python_wheel" pytest httpx
    "$python" -m pytest "$ROOT/bananamendy/tests" -q
    # Parity against transformers only runs where the checkpoints and torch are
    # present; the file skips itself otherwise, and pytest reports "no tests
    # collected" (5) for a fully skipped suite. Anything else is a real failure.
    local parity_status=0
    "$python" -m pytest "$ROOT/tests" -q || parity_status=$?
    (( parity_status == 0 || parity_status == 5 )) || return "$parity_status"
    "$python" - "$expected" <<'PY'
import sys

import bananamendr
import bananamendy
from bananamendy.server import create_app

expected = sys.argv[1]
assert bananamendy.__version__ == expected, (bananamendy.__version__, expected)
assert hasattr(bananamendr, "Model"), dir(bananamendr)
assert set(bananamendy.REGISTRY) == {"nano", "mini", "pro"}, bananamendy.REGISTRY
routes = {route.path for route in create_app(bananamendy.Config()).routes}
for path in ("/health", "/v1/models", "/v1/chat/completions", "/v1/completions"):
    assert path in routes, (path, routes)
print(f"Python smoke passed: bananamendy {bananamendy.__version__}")
PY
    "$venv/bin/bananamendy" registry >/dev/null
}

# Builds the WebAssembly module, compares it with the Python module, and checks
# that docs/ holds the module for this version. `./wasm.sh --refresh` writes it.
wasm_gate() {
    if [[ "${BANANAMEND_SKIP_WASM:-0}" == "1" ]]; then
        printf 'WASM gate skipped, because BANANAMEND_SKIP_WASM=1.\n'
        return 0
    fi
    for tool in wasm-pack node rustup; do
        command -v "$tool" >/dev/null 2>&1 || {
            printf 'The WASM gate needs %s. Install it, or set BANANAMEND_SKIP_WASM=1.\n' \
                "$tool" >&2
            return 1
        }
    done
    "$ROOT/wasm.sh"
}

run_build() {
    local tool
    for tool in git cargo rustc uv uvx python3; do
        require_tool "$tool"
    done
    verify_layout
    record_tree "$TEMP_DIR/before"

    printf 'Repository: %s\n' "$ROOT"
    printf 'rustc: %s\n' "$(rustc --version)"
    printf 'cargo: %s\n' "$(cargo --version)"
    printf 'uv: %s\n' "$(uv --version)"
    printf 'python: %s\n' "$(python3 --version 2>&1)"

    local version
    version="$(python3 "$ROOT/scripts/release-manifests.py" check)"
    printf 'Synchronized version: %s\n' "$version"

    cargo fmt --manifest-path "$ROOT/Cargo.toml" --all -- --check
    cargo clippy --manifest-path "$ROOT/Cargo.toml" --all-targets --no-deps -- -D warnings
    cargo build --manifest-path "$ROOT/Cargo.toml" --workspace
    cargo build --manifest-path "$ROOT/Cargo.toml" -p bananamendr --examples
    cargo test --manifest-path "$ROOT/Cargo.toml" --workspace
    cargo test --manifest-path "$ROOT/Cargo.toml" -p bananamendr --doc
    # The engine must also build in the form that a browser uses: no files, one
    # thread, and the Rust library for regular expressions.
    cargo build --manifest-path "$ROOT/Cargo.toml" -p bananamendr \
        --no-default-features --features wasm
    python_gate "$version"
    wasm_gate

    record_tree "$TEMP_DIR/after"
    verify_tree_unchanged "$TEMP_DIR/before" "$TEMP_DIR/after"
}

trap finish EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
: >"$LOG"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bananamend-build.XXXXXX")"
run_build 2>&1 | tee -a "$LOG"
BUILD_COMPLETED=1
