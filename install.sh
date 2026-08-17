#!/usr/bin/env bash
# this_file: install.sh
#
# Installs the `bananamendr` CLI binary into a prefix, and the `bananamendr`
# extension module plus `bananamendy` into the interpreter uv considers the
# system Python. Wheels are built from this checkout and smoke-tested in a
# throwaway environment first. The checkout is left byte-identical.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PREFIX="${BANANAMEND_PREFIX:-${HOME:?HOME is required}/.local}"
TEMP_DIR=""
TARGET_PYTHON=""

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}
trap cleanup EXIT

usage() {
    printf 'Usage: %s [--prefix PATH] [--python PATH]\n' "$0"
}

while (( $# > 0 )); do
    case "$1" in
        --prefix)
            (( $# >= 2 )) || { printf '%s\n' '--prefix requires a path' >&2; exit 2; }
            PREFIX=$2
            shift 2
            ;;
        --python)
            (( $# >= 2 )) || { printf '%s\n' '--python requires a path' >&2; exit 2; }
            TARGET_PYTHON=$2
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for tool in cargo rustc uv python3 git; do
    command -v "$tool" >/dev/null 2>&1 || {
        printf 'Required tool is missing: %s\n' "$tool" >&2
        exit 1
    }
done

for required in crates/bananamendr/Cargo.toml crates/bananamendr-py/pyproject.toml \
    bananamendy/pyproject.toml scripts/check-python-artifacts.py; do
    [[ -e "$ROOT/$required" ]] || {
        printf 'Canonical project path is missing: %s\n' "$required" >&2
        exit 1
    }
done

mkdir -p "$PREFIX"
PREFIX="$(cd "$PREFIX" && pwd -P)"
[[ -w "$PREFIX" ]] || {
    printf 'Install prefix is not writable: %s\n' "$PREFIX" >&2
    exit 1
}

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bananamend-install.XXXXXX")"
DIST="$TEMP_DIR/dist"
VENV="$TEMP_DIR/smoke-venv"
CLI="$PREFIX/bin/bananamendr"
mkdir -p "$DIST/bananamendr" "$DIST/bananamendy" "$PREFIX/bin"

git -C "$ROOT" status --porcelain=v1 --untracked-files=all >"$TEMP_DIR/before.status"
git -C "$ROOT" diff --no-ext-diff --binary HEAD -- >"$TEMP_DIR/before.diff"

VERSION="$(python3 "$ROOT/scripts/release-manifests.py" check)"
if [[ -z "$TARGET_PYTHON" ]]; then
    TARGET_PYTHON="$(uv python find --system --no-managed-python)"
fi
printf 'Installing bananamend %s\n' "$VERSION"
printf 'rustc: %s\n' "$(rustc --version)"
printf 'uv: %s\n' "$(uv --version)"
printf 'target Python: %s\n' "$TARGET_PYTHON"

CARGO_TARGET_DIR="$TEMP_DIR/cargo-target" cargo install \
    --path "$ROOT/crates/bananamendr" \
    --root "$PREFIX" \
    --force
[[ -x "$CLI" ]] || {
    printf 'CLI installation did not create %s\n' "$CLI" >&2
    exit 1
}

(cd "$ROOT/crates/bananamendr-py" && uv build --no-sources --out-dir "$DIST/bananamendr")
(cd "$ROOT/bananamendy" && uv build --no-sources --out-dir "$DIST/bananamendy")
python3 "$ROOT/scripts/check-python-artifacts.py" "$DIST/bananamendr" "$VERSION" >"$TEMP_DIR/r.txt"
python3 "$ROOT/scripts/check-python-artifacts.py" "$DIST/bananamendy" "$VERSION" >"$TEMP_DIR/y.txt"
RUST_WHEEL="$(sed -n '1p' "$TEMP_DIR/r.txt")"
PY_WHEEL="$(sed -n '1p' "$TEMP_DIR/y.txt")"

uv venv --clear --python "$TARGET_PYTHON" "$VENV"
uv pip install --python "$VENV/bin/python" "$RUST_WHEEL" "$PY_WHEEL"
"$VENV/bin/python" - "$VERSION" <<'PY'
import sys

import bananamendr
import bananamendy

expected = sys.argv[1]
assert bananamendy.__version__ == expected, (bananamendy.__version__, expected)
assert hasattr(bananamendr, "Model")
print(f"Wheel smoke passed: bananamendy {bananamendy.__version__}")
PY

uv pip install --system --python "$TARGET_PYTHON" --reinstall "$RUST_WHEEL" "$PY_WHEEL"
"$CLI" --version
"$TARGET_PYTHON" -c 'import bananamendr, bananamendy; print("installed bananamendy", bananamendy.__version__)'

git -C "$ROOT" status --porcelain=v1 --untracked-files=all >"$TEMP_DIR/after.status"
git -C "$ROOT" diff --no-ext-diff --binary HEAD -- >"$TEMP_DIR/after.diff"
if ! cmp -s "$TEMP_DIR/before.status" "$TEMP_DIR/after.status" || \
    ! cmp -s "$TEMP_DIR/before.diff" "$TEMP_DIR/after.diff"; then
    printf 'Install changed the checkout. Before/after status follows.\n' >&2
    diff -u "$TEMP_DIR/before.status" "$TEMP_DIR/after.status" >&2 || true
    exit 1
fi

printf '\nInstallation complete: bananamend %s\n' "$VERSION"
printf 'CLI: %s\n' "$CLI"
printf 'Python: %s (bananamendy on PATH next to it)\n' "$TARGET_PYTHON"
printf 'Add the CLI for this shell with: export PATH="%s/bin:$%s"\n' "$PREFIX" PATH
