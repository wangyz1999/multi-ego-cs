"""Match-id handling. Mixing bare and suffixed ids silently produces empty joins."""

import pytest

from multi_ego_cs.util import ids

BARE = "1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60"
SUFFIXED = f"{BARE}-1-1"


def test_recognises_both_forms():
    assert ids.is_bare(BARE) and not ids.is_suffixed(BARE)
    assert ids.is_suffixed(SUFFIXED) and not ids.is_bare(SUFFIXED)


def test_round_trip():
    assert ids.bare_id(SUFFIXED) == BARE
    assert ids.bare_id(BARE) == BARE
    assert ids.canonical_id(BARE) == SUFFIXED
    assert ids.canonical_id(SUFFIXED) == SUFFIXED


def test_rejects_non_ids():
    for bad in ("", "nope", "2-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60"):
        with pytest.raises(ValueError):
            ids.bare_id(bad)


def test_extract_from_paths_and_text():
    assert ids.match_id_from_path(f"{SUFFIXED}.dem") == SUFFIXED
    assert ids.match_id_from_path(f"match={SUFFIXED}") == SUFFIXED
    assert ids.match_id_from_path("no-ids-here") is None


def test_extract_deduplicates_and_preserves_order():
    text = f"{SUFFIXED}\n{BARE}\n{SUFFIXED}\n"
    found = ids.extract_ids(text)
    assert found == [SUFFIXED, BARE]


def test_steamid_extraction():
    assert ids.steamid_from_path("76561198064353169.mp4") == "76561198064353169"
    assert ids.steamid_from_path("round_5.mp4") is None
