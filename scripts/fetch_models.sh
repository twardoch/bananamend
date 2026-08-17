#!/usr/bin/env bash
# this_file: scripts/fetch_models.sh
#
# Clones the BananaMind-2 chat checkpoints (and BananaMindOS, for reference)
# into ref/. Needs git with git-lfs. Total download is about 1.4 GB.
#
#   scripts/fetch_models.sh              # all three checkpoints + BananaMindOS
#   scripts/fetch_models.sh nano         # just the smallest one (40 MB)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/ref"
mkdir -p "$dest"

declare -A repos=(
  [nano]="https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat"
  [mini]="https://huggingface.co/BananaMind/BananaMind-2-Mini-Chat"
  [pro]="https://huggingface.co/BananaMind/BananaMind-2-Pro-Preview-Chat"
  [os]="https://github.com/BananaMind/BananaMindOS"
)

wanted=("$@")
if [ ${#wanted[@]} -eq 0 ]; then
  wanted=(nano mini pro os)
fi

for key in "${wanted[@]}"; do
  url="${repos[$key]:-}"
  if [ -z "$url" ]; then
    echo "unknown target '$key' (expected: nano mini pro os)" >&2
    exit 2
  fi
  name="$(basename "$url")"
  if [ -d "$dest/$name" ]; then
    echo "$name: already present, skipping"
    continue
  fi
  echo "$name: cloning"
  git clone --depth 1 "$url" "$dest/$name"
done

echo "checkpoints in $dest:"
ls -1 "$dest"
