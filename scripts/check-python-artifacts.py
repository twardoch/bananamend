#!/usr/bin/env python3
# this_file: scripts/check-python-artifacts.py
"""Verify wheel and sdist metadata for one canonical Python release version."""

from __future__ import annotations

import email.parser
import sys
import tarfile
import zipfile
from pathlib import Path


class ArtifactError(RuntimeError):
    """Report an invalid or ambiguous Python distribution set."""


def metadata_version(payload: bytes, source: Path) -> str:
    """Read a Version field from core metadata."""
    metadata = email.parser.BytesParser().parsebytes(payload)
    version = metadata.get("Version")
    if not version:
        raise ArtifactError(f"{source}: metadata has no Version field")
    return version


def inspect(dist: Path, expected: str) -> tuple[Path, Path]:
    """Require exactly one wheel and sdist whose metadata equals expected."""
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ArtifactError(
            f"{dist}: expected one wheel and one sdist, found {len(wheels)} and {len(sdists)}"
        )
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ArtifactError(f"{wheel}: expected one wheel METADATA file")
        wheel_version = metadata_version(archive.read(names[0]), wheel)
    with tarfile.open(sdist, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.name.endswith("/PKG-INFO")]
        if len(members) != 1:
            raise ArtifactError(f"{sdist}: expected one sdist PKG-INFO file")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ArtifactError(f"{sdist}: cannot read PKG-INFO")
        sdist_version = metadata_version(stream.read(), sdist)
    if wheel_version != expected or sdist_version != expected:
        raise ArtifactError(
            f"artifact versions differ: expected={expected}, wheel={wheel_version}, "
            f"sdist={sdist_version}"
        )
    return wheel, sdist


def main(argv: list[str]) -> int:
    """Run artifact inspection and print the two admitted paths."""
    if len(argv) != 3:
        print("usage: check-python-artifacts.py DIST_DIR EXPECTED_VERSION", file=sys.stderr)
        return 2
    try:
        wheel, sdist = inspect(Path(argv[1]).resolve(), argv[2])
    except (ArtifactError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"Python artifact error: {error}", file=sys.stderr)
        return 1
    print(wheel)
    print(sdist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
