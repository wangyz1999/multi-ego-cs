"""Crash-safe IO helpers.

Collection runs are long and get interrupted (SLURM time limits, a Windows
machine rebooting mid-record). Every write here is atomic: content lands in a
temporary file on the same filesystem and is then renamed into place, so a
reader never observes a half-written JSON file and a resumed run can trust
whatever is already on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serialise `payload` to `path` atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record. Line-buffered + fsync so an interrupted run keeps it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_jsonl_atomic(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream records, tolerating a truncated final line from a hard kill."""
    p = Path(path)
    if not p.exists():
        return
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A partial trailing line is expected after an abrupt stop.
                import logging

                logging.getLogger("mecs.io").warning(
                    "Skipping malformed line %d in %s", lineno, p
                )


def du_bytes(path: str | Path) -> int:
    """Recursive apparent size in bytes. Returns 0 for a missing path."""
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    total = 0
    for dirpath, _dirnames, filenames in os.walk(p):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
    return total


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PiB"


def materialize(src: Path, dst: Path, mode: str = "auto") -> str:
    """Place `src` at `dst` as cheaply as the filesystem allows.

    Modes:
      ``hardlink``  same-filesystem only; fails over to the next option
      ``symlink``   a pointer, costs one inode and no bytes
      ``copy``      a real, self-contained duplicate
      ``auto``      hardlink, then symlink, then copy  (default)

    Why ``auto`` prefers a symlink over a copy: a release tree is usually
    assembled on scratch from sources on a different filesystem, so a hard link
    is impossible and a copy duplicates hundreds of gigabytes for no benefit -
    the upload stage just opens the path, and a symlink resolves transparently.

    Use ``copy`` when the release must be self-contained: tarred up, moved to
    another machine, or kept after the source is deleted.

    Returns what actually happened, so callers can report it.
    """
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"

    order = {
        "auto": ("hardlink", "symlink", "copy"),
        "hardlink": ("hardlink", "symlink", "copy"),
        "symlink": ("symlink", "copy"),
        "copy": ("copy",),
    }.get(mode, ("hardlink", "symlink", "copy"))

    for attempt in order:
        try:
            if attempt == "hardlink":
                # Resolve first: hard-linking a symlink would link its target
                # under a name that outlives the intermediate link.
                os.link(src.resolve(), dst)
                return "hardlink"
            if attempt == "symlink":
                dst.symlink_to(src.resolve())
                return "symlink"
            shutil.copy2(src, dst)
            return "copy"
        except OSError:
            continue
    raise OSError(f"Could not materialize {src} at {dst} (mode={mode})")


def link_or_copy(src: Path, dst: Path, prefer_link: bool = True) -> str:
    """Backwards-compatible shim for :func:`materialize`."""
    return materialize(src, dst, mode="auto" if prefer_link else "copy")
