#!/usr/bin/env bash
# this_file: publish.sh
#
# Release bananamendr (crates.io + PyPI) and bananamendy (PyPI) from a clean,
# pushed main. Dry run by default; `--real` performs the irreversible uploads.
#
# Order of operations, and why:
#   1. read-only preflight   — refuse anything ambiguous before touching state
#   2. checkpoint            — commit the working tree so the release is a ref
#   3. pre-mutation gates    — full build + dry-run uploads at the CURRENT version
#   4. prepare target        — sync manifests to the predicted version, rebuild
#   5. gitnextver            — one commit + tag + push (it only tags a dirty tree,
#                              which step 4 guarantees)
#   6. verify refs           — local tag, remote branch and remote tag must agree
#   7. upload                — crate first, then PyPI (bananamendr before bananamendy)
#
# The predicted version mirrors gitnextver: highest v-tag plus one patch, or
# 1.0.0 for a repository with no tags.

set -euo pipefail
export GIT_PAGER=cat
export PAGER=cat

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
MODE="dry-run"
PHASE="startup"
TEMP_DIR=""
CHECKPOINT_MESSAGE="chore: checkpoint before release"
SNAPSHOT_COMMIT=""
REMOTE_BASE=""
TARGET_VERSION=""
TARGET_TAG=""
CURRENT_VERSION=""
RELEASE_COMMIT=""
R_WHEEL=""
R_SDIST=""
Y_WHEEL=""
Y_SDIST=""
COMPLETED=()
MANIFEST_ALLOWLIST=(
    "Cargo.toml"
    "bananamendy/pyproject.toml"
    "bananamendy/src/bananamendy/__init__.py"
)

usage() {
    printf 'Usage: %s [--real]\n' "$0"
}

while (( $# > 0 )); do
    case "$1" in
        --real) MODE="real"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

cleanup() {
    if [[ -n "$TEMP_DIR" && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}

finish() {
    local status=$?
    trap - EXIT
    if (( status != 0 )); then
        printf '\nRELEASE FAILURE in phase %s (exit %d).\n' "$PHASE" "$status" >&2
        if (( ${#COMPLETED[@]} > 0 )); then
            printf 'Already published and NOT revocable:\n' >&2
            printf '  %s\n' "${COMPLETED[@]}" >&2
        fi
    fi
    cleanup
    exit "$status"
}
trap finish EXIT

require_tool() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'Required tool is missing: %s\n' "$1" >&2
        return 1
    }
}

remote_tag_commit() {
    local tag=$1 refs peeled
    refs="$(git -C "$ROOT" ls-remote --tags origin "refs/tags/$tag" "refs/tags/$tag^{}")"
    peeled="$(printf '%s\n' "$refs" | awk '$2 ~ /\^\{\}$/ {print $1}')"
    if [[ -n "$peeled" ]]; then
        printf '%s\n' "$peeled"
    else
        printf '%s\n' "$refs" | awk '$2 !~ /\^\{\}$/ {print $1}'
    fi
}

read_only_preflight() {
    PHASE="read-only-preflight"
    local tool
    for tool in git cargo rustc uv uvx python3; do
        require_tool "$tool"
    done
    if [[ "$MODE" == "real" ]]; then
        require_tool curl
        [[ -n "${CARGO_REGISTRY_TOKEN:-}" ]] || {
            printf 'CARGO_REGISTRY_TOKEN is required for a real release.\n' >&2
            return 1
        }
        [[ -n "${UV_PUBLISH_TOKEN:-}" ]] || {
            printf 'UV_PUBLISH_TOKEN is required for a real release.\n' >&2
            return 1
        }
    fi

    [[ -x "$ROOT/build.sh" ]] || {
        printf 'build.sh is missing or not executable.\n' >&2
        return 1
    }
    [[ -z "$(git -C "$ROOT" ls-files -u)" ]] || {
        printf 'Publish cannot checkpoint a tree with unresolved conflicts.\n' >&2
        return 1
    }

    local branch upstream head repository_state
    branch="$(git -C "$ROOT" symbolic-ref --quiet --short HEAD)" || {
        printf 'Publish requires an attached branch.\n' >&2
        return 1
    }
    [[ "$branch" == "main" ]] || {
        printf 'Publish requires branch main; found %s.\n' "$branch" >&2
        return 1
    }
    upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}')" || {
        printf 'Publish requires an upstream branch.\n' >&2
        return 1
    }
    [[ "$upstream" == "origin/main" ]] || {
        printf 'Publish requires upstream origin/main; found %s.\n' "$upstream" >&2
        return 1
    }
    head="$(git -C "$ROOT" rev-parse HEAD)"
    REMOTE_BASE="$(git -C "$ROOT" ls-remote --exit-code origin refs/heads/main | awk 'NR == 1 {print $1}')"
    [[ -n "$REMOTE_BASE" ]] || {
        printf 'Live origin/main is missing.\n' >&2
        return 1
    }
    if [[ "$head" == "$REMOTE_BASE" ]]; then
        repository_state="equal"
    elif git -C "$ROOT" cat-file -e "$REMOTE_BASE^{commit}" 2>/dev/null && \
         git -C "$ROOT" merge-base --is-ancestor "$REMOTE_BASE" "$head"; then
        repository_state="local-ahead"
    else
        printf 'Live origin/main is ahead of or diverged from local HEAD. local=%s remote=%s\n' \
            "$head" "$REMOTE_BASE" >&2
        return 1
    fi

    local tags_file="$TEMP_DIR/tags.txt"
    git -C "$ROOT" tag --list 'v*' >"$tags_file"
    git -C "$ROOT" ls-remote --tags origin \
        | awk '$2 !~ /\^\{\}$/ {sub("refs/tags/", "", $2); print $2}' >>"$tags_file"
    TARGET_VERSION="$(python3 "$ROOT/scripts/release-manifests.py" next-patch <"$tags_file")"
    TARGET_TAG="v$TARGET_VERSION"
    CURRENT_VERSION="$(python3 "$ROOT/scripts/release-manifests.py" check)"

    # When earlier releases exist, the committed manifests must match the latest
    # tag; the first release has nothing to match against.
    if [[ -s "$tags_file" ]]; then
        local major minor patch latest_tag local_latest remote_latest
        IFS=. read -r major minor patch <<<"$TARGET_VERSION"
        latest_tag="v$major.$minor.$((patch - 1))"
        local_latest="$(git -C "$ROOT" rev-parse "$latest_tag^{commit}")" || {
            printf 'Latest canonical tag is missing locally: %s\n' "$latest_tag" >&2
            return 1
        }
        remote_latest="$(remote_tag_commit "$latest_tag")"
        [[ -n "$remote_latest" && "$local_latest" == "$remote_latest" ]] || {
            printf 'Latest local and remote tag commits differ for %s.\n' "$latest_tag" >&2
            return 1
        }
        [[ "$CURRENT_VERSION" == "${latest_tag#v}" ]] || {
            printf 'Manifest version %s does not equal latest release tag %s.\n' \
                "$CURRENT_VERSION" "$latest_tag" >&2
            return 1
        }
    fi

    if git -C "$ROOT" show-ref --verify --quiet "refs/tags/$TARGET_TAG" || \
        [[ -n "$(remote_tag_commit "$TARGET_TAG")" ]]; then
        printf 'Predicted release tag already exists: %s\n' "$TARGET_TAG" >&2
        return 1
    fi

    printf 'Preflight passed: current=%s predicted=%s branch=%s repository=%s remote-base=%s\n' \
        "$CURRENT_VERSION" "$TARGET_VERSION" "$branch" "$repository_state" "$REMOTE_BASE"
}

checkpoint_changes() {
    PHASE="checkpoint"
    git -C "$ROOT" add -A -- .
    git -C "$ROOT" diff --cached --check
    if git -C "$ROOT" diff --cached --quiet; then
        printf 'Checkpoint not needed: working tree already clean.\n'
    else
        git -C "$ROOT" commit -m "$CHECKPOINT_MESSAGE"
    fi
    [[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
        printf 'Checkpoint did not produce a clean tree.\n' >&2
        return 1
    }
    SNAPSHOT_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
    printf 'Checkpoint commit: %s\n' "$SNAPSHOT_COMMIT"
}

# Build both distributions for $1 into a labelled directory and set the
# R_/Y_ wheel and sdist paths.
python_package_gate() {
    local expected=$1 label=$2
    local dist="$TEMP_DIR/dist-$label"
    local venv="$TEMP_DIR/venv-$label"
    mkdir -p "$dist/bananamendr" "$dist/bananamendy"
    (cd "$ROOT/crates/bananamendr-py" && uv build --no-sources --out-dir "$dist/bananamendr")
    (cd "$ROOT/bananamendy" && uv build --no-sources --out-dir "$dist/bananamendy")
    python3 "$ROOT/scripts/check-python-artifacts.py" "$dist/bananamendr" "$expected" >"$dist/r.txt"
    python3 "$ROOT/scripts/check-python-artifacts.py" "$dist/bananamendy" "$expected" >"$dist/y.txt"
    R_WHEEL="$(sed -n '1p' "$dist/r.txt")"
    R_SDIST="$(sed -n '2p' "$dist/r.txt")"
    Y_WHEEL="$(sed -n '1p' "$dist/y.txt")"
    Y_SDIST="$(sed -n '2p' "$dist/y.txt")"

    uv venv --clear --python python3 "$venv"
    uv pip install --python "$venv/bin/python" "$R_WHEEL" "$Y_WHEEL"
    "$venv/bin/python" - "$expected" <<'PY'
import sys

import bananamendr
import bananamendy

expected = sys.argv[1]
assert bananamendy.__version__ == expected, (bananamendy.__version__, expected)
assert hasattr(bananamendr, "Model"), dir(bananamendr)
print(f"Python package gate passed: bananamendy {bananamendy.__version__}")
PY
}

pre_mutation_gates() {
    PHASE="pre-mutation-gates"
    "$ROOT/build.sh"
    # A dry run inspects an uncommitted tree on purpose; a real release has
    # already been checkpointed, so cargo must see a clean checkout there.
    local dirty=()
    [[ "$MODE" == "dry-run" ]] && dirty=(--allow-dirty)
    cargo publish --manifest-path "$ROOT/Cargo.toml" --dry-run -p bananamendr "${dirty[@]+"${dirty[@]}"}"
    python_package_gate "$CURRENT_VERSION" current
    uv publish --dry-run "$R_WHEEL" "$R_SDIST"
    uv publish --dry-run "$Y_WHEEL" "$Y_SDIST"
}

assert_manifest_diff() {
    local actual="$TEMP_DIR/actual-manifests.txt"
    local expected="$TEMP_DIR/expected-manifests.txt"
    git -C "$ROOT" diff --name-only -- >"$actual"
    printf '%s\n' "${MANIFEST_ALLOWLIST[@]}" >"$expected"
    LC_ALL=C sort -o "$actual" "$actual"
    LC_ALL=C sort -o "$expected" "$expected"
    if ! cmp -s "$actual" "$expected"; then
        printf 'Release preparation changed files outside the manifest allowlist.\n' >&2
        diff -u "$expected" "$actual" >&2 || true
        return 1
    fi
    [[ -z "$(git -C "$ROOT" diff --cached --name-only)" ]] || {
        printf 'Release preparation unexpectedly staged files.\n' >&2
        return 1
    }
}

prepare_target() {
    PHASE="prepare-target"
    python3 "$ROOT/scripts/release-manifests.py" sync "$TARGET_VERSION" >/dev/null
    assert_manifest_diff
    PHASE="target-build"
    "$ROOT/build.sh"
    assert_manifest_diff
}

verify_release_boundary() {
    PHASE="release-boundary"
    local live_remote head
    live_remote="$(git -C "$ROOT" ls-remote --exit-code origin refs/heads/main | awk 'NR == 1 {print $1}')"
    [[ "$live_remote" == "$REMOTE_BASE" ]] || {
        printf 'Live origin/main changed after preflight. before=%s now=%s\n' \
            "$REMOTE_BASE" "$live_remote" >&2
        return 1
    }
    head="$(git -C "$ROOT" rev-parse HEAD)"
    [[ -n "$SNAPSHOT_COMMIT" && "$head" == "$SNAPSHOT_COMMIT" ]] || {
        printf 'HEAD changed after checkpoint. snapshot=%s now=%s\n' "$SNAPSHOT_COMMIT" "$head" >&2
        return 1
    }
    if git -C "$ROOT" show-ref --verify --quiet "refs/tags/$TARGET_TAG" || \
        [[ -n "$(remote_tag_commit "$TARGET_TAG")" ]]; then
        printf 'Predicted release tag appeared after preflight: %s\n' "$TARGET_TAG" >&2
        return 1
    fi
    assert_manifest_diff
    printf 'Release boundary verified: snapshot=%s target=%s\n' "$SNAPSHOT_COMMIT" "$TARGET_TAG"
}

verify_release_refs() {
    PHASE="verify-refs"
    [[ -z "$(git -C "$ROOT" status --porcelain=v1 --untracked-files=all)" ]] || {
        printf 'Release command left a dirty tree.\n' >&2
        return 1
    }
    RELEASE_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
    local local_tag branch_remote tag_remote new_tags
    local_tag="$(git -C "$ROOT" rev-parse "$TARGET_TAG^{commit}")" || {
        printf 'Expected local release tag is missing: %s\n' "$TARGET_TAG" >&2
        return 1
    }
    new_tags="$(comm -13 "$TEMP_DIR/tags-before.txt" <(git -C "$ROOT" tag --list 'v*' | LC_ALL=C sort))"
    [[ "$new_tags" == "$TARGET_TAG" ]] || {
        printf 'Release command created unexpected tags: %s\n' "$new_tags" >&2
        return 1
    }
    branch_remote="$(git -C "$ROOT" ls-remote --exit-code origin refs/heads/main | awk 'NR == 1 {print $1}')"
    tag_remote="$(remote_tag_commit "$TARGET_TAG")"
    [[ "$local_tag" == "$RELEASE_COMMIT" && \
       "$branch_remote" == "$RELEASE_COMMIT" && \
       "$tag_remote" == "$RELEASE_COMMIT" ]] || {
        printf 'Release ref mismatch: HEAD=%s local-tag=%s remote-branch=%s remote-tag=%s\n' \
            "$RELEASE_COMMIT" "$local_tag" "$branch_remote" "$tag_remote" >&2
        return 1
    }
    python3 "$ROOT/scripts/release-manifests.py" check "$TARGET_VERSION" >/dev/null

    local hatch_version
    hatch_version="$(cd "$ROOT/crates/bananamendr-py" && uvx --with hatch-vcs hatchling version)"
    [[ "$hatch_version" == "$TARGET_VERSION" ]] || {
        printf 'Tag-derived Hatch version differs: expected=%s hatch=%s\n' \
            "$TARGET_VERSION" "$hatch_version" >&2
        return 1
    }
    printf 'Verified release refs and tag-derived version: %s at %s\n' "$TARGET_TAG" "$RELEASE_COMMIT"
}

wait_for_crate() {
    local package=$1 version=$2
    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if curl --fail --silent --show-error \
            --header 'User-Agent: bananamend-release-check' \
            "https://crates.io/api/v1/crates/$package/$version" >/dev/null; then
            return 0
        fi
        sleep 5
    done
    printf 'Timed out waiting for crates.io visibility: %s@%s\n' "$package" "$version" >&2
    return 1
}

wait_for_python() {
    local package=$1 version=$2
    local deadline=$((SECONDS + 180))
    while (( SECONDS < deadline )); do
        if curl --fail --silent --show-error \
            --header 'User-Agent: bananamend-release-check' \
            "https://pypi.org/pypi/$package/$version/json" >/dev/null; then
            return 0
        fi
        sleep 5
    done
    printf 'Timed out waiting for PyPI visibility: %s@%s\n' "$package" "$version" >&2
    return 1
}

publish_all() {
    PHASE="upload-crate-bananamendr"
    cargo publish --manifest-path "$ROOT/Cargo.toml" --dry-run -p bananamendr
    # Cargo/libcurl HTTP/2 uploads can fail mid-stream for registry writes.
    CARGO_HTTP_MULTIPLEXING=false cargo publish --manifest-path "$ROOT/Cargo.toml" -p bananamendr
    PHASE="registry-crate-bananamendr"
    wait_for_crate bananamendr "$TARGET_VERSION"
    COMPLETED+=("crates.io bananamendr@$TARGET_VERSION")

    PHASE="python-final-gate"
    python_package_gate "$TARGET_VERSION" release
    uv publish --dry-run "$R_WHEEL" "$R_SDIST"
    uv publish --dry-run "$Y_WHEEL" "$Y_SDIST"

    # bananamendr first: bananamendy pins bananamendr=={version} and must be
    # installable the moment it appears.
    PHASE="upload-python-bananamendr"
    uv publish "$R_WHEEL" "$R_SDIST"
    wait_for_python bananamendr "$TARGET_VERSION"
    COMPLETED+=("PyPI bananamendr@$TARGET_VERSION")

    PHASE="upload-python-bananamendy"
    uv publish "$Y_WHEEL" "$Y_SDIST"
    wait_for_python bananamendy "$TARGET_VERSION"
    COMPLETED+=("PyPI bananamendy@$TARGET_VERSION")
}

TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/bananamend-release.XXXXXX")"
read_only_preflight

if [[ "$MODE" == "dry-run" ]]; then
    PHASE="dry-run-gates"
    changes="$(git -C "$ROOT" status --short --untracked-files=all)"
    if [[ -n "$changes" ]]; then
        printf 'Would checkpoint these non-ignored changes:\n%s\n' "$changes"
    else
        printf 'No checkpoint commit would be needed.\n'
    fi
    pre_mutation_gates
    PHASE="dry-run-complete"
    printf '\nDRY RUN SUCCESS\n'
    printf 'Predicted next release: %s\n' "$TARGET_TAG"
    printf 'Real mode syncs %d manifests, rebuilds, tags once, then uploads bananamendr to crates.io and bananamendr + bananamendy to PyPI.\n' \
        "${#MANIFEST_ALLOWLIST[@]}"
    exit 0
fi

checkpoint_changes
pre_mutation_gates
prepare_target
verify_release_boundary
git -C "$ROOT" tag --list 'v*' | LC_ALL=C sort >"$TEMP_DIR/tags-before.txt"
PHASE="create-release"
(cd "$ROOT" && uvx gitnextver@1.0.1 --directory . --verbose)
verify_release_refs
publish_all
PHASE="complete"

printf '\nRELEASE SUCCESS: %s at %s\n' "$TARGET_TAG" "$RELEASE_COMMIT"
printf 'Published:\n'
printf '  %s\n' "${COMPLETED[@]}"
