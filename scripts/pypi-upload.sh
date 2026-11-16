#!/usr/bin/env bash
# this_file: scripts/pypi-upload.sh
#
# Upload locally built wheels/sdists to PyPI without a token:
#   scripts/pypi-upload.sh dist/*.whl dist/*.tar.gz
#   scripts/pypi-upload.sh --dry-run target/wheels/*.whl
# Attaches the files to a draft GitHub release (no git tag), dispatches
# .github/workflows/pypi-upload.yml, which uploads them through Trusted
# Publishing, and waits for the run. Exits non-zero if the upload fails.

set -euo pipefail

dry=""
if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi
(( $# )) || { echo "usage: $0 [--dry-run] FILE.whl|FILE.tar.gz ..." >&2; exit 2; }
for f in "$@"; do
  [[ -f "$f" && "$f" =~ \.(whl|tar\.gz)$ ]] || { echo "error: not a wheel or sdist: $f" >&2; exit 2; }
done

if [[ -n "$dry" ]]; then
  uvx twine check "$@"
  echo "[dry-run] would upload $# file(s) to PyPI through pypi-upload.yml"
  exit 0
fi

REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
RELEASE="pypi-staging-$(date -u +%Y%m%d%H%M%S)-$$"

gh release create "$RELEASE" "$@" -R "$REPO" --draft --title "$RELEASE" \
  --notes "Staging for PyPI; deleted by pypi-upload.yml after upload."
gh workflow run pypi-upload.yml -R "$REPO" -f release="$RELEASE"

run_id=""
for _ in $(seq 1 30); do
  run_id="$(gh run list -R "$REPO" --workflow pypi-upload.yml -L 10 --json databaseId,displayTitle \
    -q ".[] | select(.displayTitle == \"PyPI upload $RELEASE\") | .databaseId" | head -n1)"
  [[ -n "$run_id" ]] && break
  sleep 2
done
[[ -n "$run_id" ]] || { echo "error: no pypi-upload.yml run appeared for $RELEASE" >&2; exit 1; }

if ! gh run watch "$run_id" -R "$REPO" --exit-status; then
  echo "error: PyPI upload failed; draft $RELEASE kept. Retry: gh workflow run pypi-upload.yml -R $REPO -f release=$RELEASE" >&2
  exit 1
fi
echo "Uploaded to PyPI: $*"
