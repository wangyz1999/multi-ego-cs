"""CS2 ``buttons`` bitfield decoding.

The demo's per-tick ``buttons`` property is a 64-bit bitfield where each bit is
one input being held. Decoding it is what turns a positional trajectory into an
*action* trajectory: which keys the player was pressing, frame by frame.

Two views of the same data are emitted:

* one ``k_<label>`` Int8 column per logical input (``k_w``, ``k_mouse_left``, …)
* optionally, the raw integer in ``buttons``, so a consumer can decode bits
  this table does not name

Some engine inputs collapse onto one physical key - ``IN_WALK`` and
``IN_SPEED`` are both "shift" depending on the player's config - so their masks
are OR-ed into a single column rather than emitted twice.

Bit positions are taken from the CS2 engine's ``IN_*`` constants. Bits 25-31
are unnamed but do get set in practice, so they are kept in ``KEY_MAPPING``
(for raw decoding) while being deliberately absent from ``KEY_TO_INPUT``: an
unnamed bit has no meaningful column name and inventing one would be a guess
baked into the published schema.
"""

from __future__ import annotations

KEY_MAPPING: dict[str, int] = {
    "IN_ATTACK": 1 << 0,
    "IN_JUMP": 1 << 1,
    "IN_DUCK": 1 << 2,
    "IN_FORWARD": 1 << 3,
    "IN_BACK": 1 << 4,
    "IN_USE": 1 << 5,
    "IN_CANCEL": 1 << 6,
    "IN_TURNLEFT": 1 << 7,
    "IN_TURNRIGHT": 1 << 8,
    "IN_MOVELEFT": 1 << 9,
    "IN_MOVERIGHT": 1 << 10,
    "IN_ATTACK2": 1 << 11,
    "IN_RELOAD": 1 << 13,
    "IN_ALT1": 1 << 14,
    "IN_ALT2": 1 << 15,
    "IN_SPEED": 1 << 16,
    "IN_WALK": 1 << 17,
    "IN_ZOOM": 1 << 18,
    "IN_WEAPON1": 1 << 19,
    "IN_WEAPON2": 1 << 20,
    "IN_BULLRUSH": 1 << 21,
    "IN_GRENADE1": 1 << 22,
    "IN_GRENADE2": 1 << 23,
    "IN_ATTACK3": 1 << 24,
    "UNKNOWN_25": 1 << 25,
    "UNKNOWN_26": 1 << 26,
    "UNKNOWN_27": 1 << 27,
    "UNKNOWN_28": 1 << 28,
    "UNKNOWN_29": 1 << 29,
    "UNKNOWN_30": 1 << 30,
    "UNKNOWN_31": 1 << 31,
    "IN_SCORE": 1 << 33,
    "IN_INSPECT": 1 << 35,
}

# Engine name -> published column label (without the `k_` prefix).
KEY_TO_INPUT: dict[str, str] = {
    "IN_ATTACK": "mouse_left",
    "IN_ATTACK2": "mouse_right",
    "IN_ATTACK3": "mouse_middle",
    "IN_JUMP": "space",
    "IN_DUCK": "ctrl",
    "IN_FORWARD": "w",
    "IN_BACK": "s",
    "IN_MOVELEFT": "a",
    "IN_MOVERIGHT": "d",
    "IN_USE": "e",
    "IN_RELOAD": "r",
    "IN_WALK": "shift",
    "IN_SPEED": "shift",
    "IN_ZOOM": "zoom",
    "IN_SCORE": "tab",
    "IN_INSPECT": "f",
    "IN_GRENADE1": "grenade1",
    "IN_GRENADE2": "grenade2",
    "IN_CANCEL": "cancel",
    "IN_TURNLEFT": "turn_left",
    "IN_TURNRIGHT": "turn_right",
    "IN_ALT1": "alt1",
    "IN_ALT2": "alt2",
    "IN_WEAPON1": "weapon1",
    "IN_WEAPON2": "weapon2",
    "IN_BULLRUSH": "bullrush",
}


def _build_key_columns() -> dict[str, int]:
    """``k_<label>`` -> OR-ed bitmask of every engine input mapped to it."""
    columns: dict[str, int] = {}
    for engine_name, label in KEY_TO_INPUT.items():
        col = f"k_{label}"
        columns[col] = columns.get(col, 0) | KEY_MAPPING[engine_name]
    return columns


KEY_COLUMNS: dict[str, int] = _build_key_columns()

# Stable, sorted order so every parquet file in the dataset shares a schema.
KEY_COLUMN_NAMES: list[str] = sorted(KEY_COLUMNS)

_BIT_TO_NAME: list[tuple[int, str]] = sorted(
    ((mask, name) for name, mask in KEY_MAPPING.items()), key=lambda kv: kv[0]
)


def decode(buttons: int | float | None) -> list[str]:
    """Return the engine input names set in a ``buttons`` value.

    >>> decode(0b101)
    ['IN_ATTACK', 'IN_DUCK']
    """
    if buttons is None:
        return []
    try:
        if buttons != buttons:  # NaN
            return []
    except TypeError:
        return []
    value = int(buttons)
    if value == 0:
        return []
    return [name for mask, name in _BIT_TO_NAME if value & mask]


def pressed_labels(buttons: int | float | None) -> list[str]:
    """Return the published column labels active in a ``buttons`` value."""
    names = set(decode(buttons))
    return sorted({KEY_TO_INPUT[n] for n in names if n in KEY_TO_INPUT})
