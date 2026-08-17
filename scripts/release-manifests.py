#!/usr/bin/env python3
# this_file: scripts/release-manifests.py
"""Validate and synchronize the release-critical version graph.

One version number governs every manifest in this repository; the git tag is
the source of truth and this script is the only thing allowed to move the
number in tracked files.

    release-manifests.py check [expected]   # print the agreed version, or fail
    release-manifests.py sync <version>     # rewrite every manifest
    release-manifests.py next-patch         # read tags on stdin, print next
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")

# (path, regex with a single capture group for the version, replacement template)
SITES: tuple[tuple[str, str, str], ...] = (
    (
        "Cargo.toml",
        r'(?m)^version = "([^"]+)"$',
        'version = "{version}"',
    ),
    (
        "Cargo.toml",
        r'(?m)^bananamendr = \{ path = "crates/bananamendr", version = "=([^"]+)" \}$',
        'bananamendr = {{ path = "crates/bananamendr", version = "={version}" }}',
    ),
    (
        "bananamendy/pyproject.toml",
        r'(?m)^version = "([^"]+)"$',
        'version = "{version}"',
    ),
    (
        "crates/bananamendr-wasm/Cargo.toml",
        r'(?m)^bananamendr = \{ path = "../bananamendr", version = "=([^"]+)", default-features = false, features = \["wasm"\] \}$',
        'bananamendr = {{ path = "../bananamendr", version = "={version}", default-features = false, features = ["wasm"] }}',
    ),
    (
        "bananamendy/pyproject.toml",
        r'(?m)^  "bananamendr==([^"]+)",$',
        '  "bananamendr=={version}",',
    ),
    (
        "bananamendy/src/bananamendy/__init__.py",
        r'(?m)^__version__ = "([^"]+)"$',
        '__version__ = "{version}"',
    ),
)


def read(path: str) -> str:
    target = ROOT / path
    if not target.is_file():
        raise SystemExit(f"Manifest is missing: {path}")
    return target.read_text(encoding="utf-8")


def observed() -> dict[tuple[str, str], str]:
    """Map every version site to the version it currently declares."""
    found: dict[tuple[str, str], str] = {}
    for path, pattern, _ in SITES:
        matches = re.findall(pattern, read(path))
        if len(matches) != 1:
            raise SystemExit(
                f"Expected exactly one version match in {path} for {pattern!r}; "
                f"found {len(matches)}"
            )
        found[(path, pattern)] = matches[0]
    return found


def check(expected: str | None) -> str:
    found = observed()
    versions = set(found.values())
    if len(versions) != 1:
        detail = "\n".join(f"  {path}: {value}" for (path, _), value in found.items())
        raise SystemExit(f"Manifest versions disagree:\n{detail}")
    version = versions.pop()
    if not SEMVER.match(version):
        raise SystemExit(f"Manifest version is not semver: {version}")
    if expected is not None and version != expected:
        raise SystemExit(f"Manifest version {version} does not equal expected {expected}")
    return version


def sync(version: str) -> None:
    if not SEMVER.match(version):
        raise SystemExit(f"Refusing to sync a non-semver version: {version}")
    for path, pattern, template in SITES:
        target = ROOT / path
        text = read(path)
        updated, count = re.subn(pattern, template.format(version=version), text)
        if count != 1:
            raise SystemExit(f"Version rewrite in {path} touched {count} sites")
        if updated != text:
            target.write_text(updated, encoding="utf-8")


def next_patch(tag_lines: list[str]) -> str:
    """Next version from v-prefixed tags on stdin; 1.0.0 when there are none.

    This mirrors `gitnextver`, which is what actually creates the tag: highest
    existing v-tag plus one patch, or v1.0.0 for a repository with no tags.
    """
    versions: list[tuple[int, int, int]] = []
    for line in tag_lines:
        candidate = line.strip()
        if not candidate:
            continue
        match = TAG.match(candidate)
        if not match:
            raise SystemExit(f"Malformed release tag blocks prediction: {candidate}")
        versions.append(tuple(int(part) for part in match.groups()))  # type: ignore[arg-type]
    if not versions:
        return "1.0.0"
    major, minor, patch = max(versions)
    return f"{major}.{minor}.{patch + 1}"


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    command, *rest = argv
    if command == "check":
        print(check(rest[0] if rest else None))
    elif command == "sync":
        if len(rest) != 1:
            raise SystemExit("sync takes exactly one version argument")
        sync(rest[0])
        print(check(rest[0]))
    elif command == "next-patch":
        print(next_patch(sys.stdin.read().splitlines()))
    else:
        raise SystemExit(f"Unknown command: {command}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
