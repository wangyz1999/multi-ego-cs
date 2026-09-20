"""FACEIT match-id handling.

A FACEIT CS2 match id looks like::

    1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60

and the demo served for it is named with a ``-<match>-<map>`` suffix::

    1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60-1-1.dem

Both forms turn up in the wild - the API returns the bare id, filenames and
directory names carry the suffix - and mixing them silently produces empty
joins. Everything downstream of stage 02 uses the *suffixed* form as the
canonical ``match_id``, because that is what the demo, the recordings and the
published dataset are all keyed on.
"""

from __future__ import annotations

import re

# 1-<uuid4>, optionally followed by -<n>-<n>
_BARE = r"1-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
BARE_RE = re.compile(rf"^{_BARE}$")
SUFFIXED_RE = re.compile(rf"^{_BARE}-\d+-\d+$")
ANY_RE = re.compile(rf"({_BARE}(?:-\d+-\d+)?)")

DEFAULT_SUFFIX = "-1-1"


def is_bare(match_id: str) -> bool:
    return bool(BARE_RE.match(match_id))


def is_suffixed(match_id: str) -> bool:
    return bool(SUFFIXED_RE.match(match_id))


def bare_id(match_id: str) -> str:
    """Strip a trailing ``-<n>-<n>`` suffix, if present."""
    m = re.match(rf"^({_BARE})(?:-\d+-\d+)?$", match_id)
    if not m:
        raise ValueError(f"Not a FACEIT match id: {match_id!r}")
    return m.group(1)


def canonical_id(match_id: str, suffix: str = DEFAULT_SUFFIX) -> str:
    """Return the suffixed form used for filenames and dataset keys."""
    if is_suffixed(match_id):
        return match_id
    if is_bare(match_id):
        return f"{match_id}{suffix}"
    raise ValueError(f"Not a FACEIT match id: {match_id!r}")


def extract_ids(text: str, unique: bool = True) -> list[str]:
    """Pull every match id out of arbitrary text (a file listing, a CSV, …)."""
    found = ANY_RE.findall(text)
    if not unique:
        return found
    seen: dict[str, None] = {}
    for f in found:
        seen.setdefault(f, None)
    return list(seen)


def match_id_from_path(name: str) -> str | None:
    """Best-effort id recovery from a filename or directory name."""
    ids = extract_ids(name)
    return ids[0] if ids else None


def steamid_from_path(name: str) -> str | None:
    """A 17-digit Steam64 id, as used for per-player filenames."""
    m = re.search(r"\b(7656\d{13})\b", name)
    return m.group(1) if m else None
