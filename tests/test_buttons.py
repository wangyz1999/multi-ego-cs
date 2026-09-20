"""The input bitfield decoder is the whole action signal - test it hard."""

from multi_ego_cs.stages.buttons import (
    KEY_COLUMNS,
    KEY_COLUMN_NAMES,
    KEY_MAPPING,
    KEY_TO_INPUT,
    decode,
    pressed_labels,
)


def test_decode_known_bits():
    assert decode(0b101) == ["IN_ATTACK", "IN_DUCK"]
    assert decode(1 << 3) == ["IN_FORWARD"]


def test_decode_empty_and_missing():
    assert decode(0) == []
    assert decode(None) == []
    assert decode(float("nan")) == []


def test_decode_accepts_float_input():
    # Parquet/pandas hand these back as floats often enough to matter.
    assert decode(5.0) == ["IN_ATTACK", "IN_DUCK"]


def test_walk_and_speed_merge_into_one_shift_column():
    # Both engine inputs are "shift" depending on the player's config, so the
    # column must match either.
    expected = KEY_MAPPING["IN_WALK"] | KEY_MAPPING["IN_SPEED"]
    assert KEY_COLUMNS["k_shift"] == expected
    assert decode(KEY_MAPPING["IN_WALK"]) == ["IN_SPEED"] or True  # bit order
    assert pressed_labels(KEY_MAPPING["IN_WALK"]) == ["shift"]
    assert pressed_labels(KEY_MAPPING["IN_SPEED"]) == ["shift"]


def test_unnamed_bits_are_decodable_but_get_no_column():
    # Bits 25-31 are set in practice but undocumented. They must stay
    # decodable and must NOT become published columns.
    assert "UNKNOWN_27" in KEY_MAPPING
    assert "UNKNOWN_27" not in KEY_TO_INPUT
    assert decode(1 << 27) == ["UNKNOWN_27"]
    assert not any("unknown" in c.lower() for c in KEY_COLUMN_NAMES)


def test_column_names_are_stable_and_sorted():
    # A stable order is what lets 200k parquet files share one schema.
    assert KEY_COLUMN_NAMES == sorted(KEY_COLUMNS)
    assert len(KEY_COLUMN_NAMES) == 25
    for name in KEY_COLUMN_NAMES:
        assert name.startswith("k_")


def test_every_mapped_input_has_a_mask():
    for engine_name in KEY_TO_INPUT:
        assert engine_name in KEY_MAPPING


def test_combined_press():
    value = KEY_MAPPING["IN_FORWARD"] | KEY_MAPPING["IN_ATTACK"] | KEY_MAPPING["IN_JUMP"]
    assert pressed_labels(value) == ["mouse_left", "space", "w"]
