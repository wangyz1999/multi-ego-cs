"""Stage 05 - tick-level state/action trajectories.

One parquet per (match, round, player), covering exactly the alive window that
stage 03 computed and stage 04 recorded, at the demo's tick rate (64 Hz):

    state_action/match=<id>/round=<n>/<steamid>.parquet

Columns
-------
``tick``, ``tick_norm``, ``game_sec``   time, raw / zeroed / seconds
``x``, ``y``, ``z``, ``health``          world state
``pitch``, ``yaw``                       absolute view angles
``delta_pitch``, ``delta_yaw``           per-tick view change (wrap-corrected)
``usercmd_mouse_dx/dy``                  raw mouse counts from the usercmd
``k_*``                                  one Int8 column per logical input
``buttons``                              raw bitfield (optional)

The ``delta_yaw`` wrap correction matters: yaw is stored in (-180, 180], so a
player panning through due-south produces a raw diff near ±360 that would read
as an impossibly fast flick. Values outside ±180 are folded back into range.

Every file is written with the same column order and dtypes (``SCHEMA``), so
the whole tree loads as one table. Without that, parquet readers infer a schema
per file and a union across 200k files fails on the first dtype disagreement.
"""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import Any

from ..config import Config, resolve_workers
from ..util.io_utils import read_json
from ..util.logging_setup import get_logger
from .buttons import KEY_COLUMNS, KEY_COLUMN_NAMES, decode

log = get_logger("actions")

# Player properties requested from the demo parser. Asking for only what we
# need is the single biggest lever on parse time and peak memory.
PLAYER_PROPS = ["buttons", "usercmd_mouse_dx", "usercmd_mouse_dy", "pitch", "yaw", "health"]

# Canonical published column order.
BASE_COLUMNS = [
    "tick",
    "tick_norm",
    "game_sec",
    "x",
    "y",
    "z",
    "health",
    "pitch",
    "yaw",
    "delta_pitch",
    "delta_yaw",
    "usercmd_mouse_dx",
    "usercmd_mouse_dy",
]


def schema(keep_raw_buttons: bool = True) -> dict[str, Any]:
    """Canonical {column: polars dtype} for every emitted parquet."""
    import polars as pl

    out: dict[str, Any] = {
        "tick": pl.Int32,
        "tick_norm": pl.Int32,
        "game_sec": pl.Float32,
        "x": pl.Float32,
        "y": pl.Float32,
        "z": pl.Float32,
        "health": pl.Float32,
        "pitch": pl.Float32,
        "yaw": pl.Float32,
        "delta_pitch": pl.Float32,
        "delta_yaw": pl.Float32,
        "usercmd_mouse_dx": pl.Float32,
        "usercmd_mouse_dy": pl.Float32,
    }
    for col in KEY_COLUMN_NAMES:
        out[col] = pl.Int8
    if keep_raw_buttons:
        out["buttons"] = pl.List(pl.Utf8)
    return out


def column_order(keep_raw_buttons: bool = True) -> list[str]:
    cols = [*BASE_COLUMNS, *KEY_COLUMN_NAMES]
    if keep_raw_buttons:
        cols.append("buttons")
    return cols


def _round_is_complete(
    round_dir: Path, steamids: list[str], expected_columns: list[str]
) -> bool:
    """True when every player's parquet exists with the current schema.

    Checking columns (not just existence) means a schema change invalidates
    stale files instead of silently leaving a mixed-schema tree behind.
    """
    import polars as pl

    for sid in steamids:
        p = round_dir / f"{sid}.parquet"
        if not p.exists():
            return False
        try:
            have = pl.read_parquet_schema(p)
        except Exception:
            return False
        if list(have.keys()) != expected_columns:
            return False
    return True


def _process_match(args: tuple[str, str, str, str, dict[str, Any]]) -> tuple[str, str, str]:
    """Parse one demo and emit every (round, player) parquet. Runs in a subprocess."""
    match_id, demo_str, metadata_str, out_root_str, opts = args
    import polars as pl
    from awpy import Demo

    try:
        meta = read_json(metadata_str)
        alive = meta.get("player_alive_times") or {}
        if not alive:
            return match_id, "skipped", "no player_alive_times in metadata"

        keep_raw = bool(opts["keep_raw_buttons"])
        cols = column_order(keep_raw)
        dtypes = schema(keep_raw)
        out_root = Path(out_root_str)
        wanted_rounds = set(opts.get("rounds") or [])

        # Resume check before the expensive parse.
        todo_rounds = []
        for round_num, entries in alive.items():
            if wanted_rounds and int(round_num) not in wanted_rounds:
                continue
            rdir = out_root / f"match={match_id}" / f"round={round_num}"
            sids = [e["steamid"] for e in entries]
            if not _round_is_complete(rdir, sids, cols):
                todo_rounds.append(round_num)
        if not todo_rounds:
            return match_id, "skipped", "already complete"

        dem = Demo(demo_str, tickrate=int(opts["tickrate"]))
        dem.parse(player_props=PLAYER_PROPS)
        ticks = dem.ticks.rename({"X": "x", "Y": "y", "Z": "z"})

        tickrate = float(opts["tickrate"])
        written = 0

        for round_num in todo_rounds:
            entries = alive[round_num]
            rdir = out_root / f"match={match_id}" / f"round={round_num}"
            rdir.mkdir(parents=True, exist_ok=True)

            for entry in entries:
                sid = entry["steamid"]
                start = int(entry["alive_start_tick"])
                # alive_end_tick is the death tick itself; the last tick the
                # player was actually alive is the one before it.
                end = int(entry["alive_end_tick"]) - 1
                if end < start:
                    continue

                sub = ticks.filter(
                    (pl.col("steamid").cast(pl.Int64) == int(sid))
                    & (pl.col("tick").cast(pl.Int64) >= start)
                    & (pl.col("tick").cast(pl.Int64) <= end)
                ).sort("tick")

                if sub.is_empty():
                    continue

                raw_dyaw = pl.col("yaw").diff().fill_null(0.0)
                sub = sub.with_columns(
                    (pl.col("tick") - start).alias("tick_norm"),
                    ((pl.col("tick") - start) / tickrate).round(6).alias("game_sec"),
                    pl.when(raw_dyaw > 180.0)
                    .then(raw_dyaw - 360.0)
                    .when(raw_dyaw < -180.0)
                    .then(raw_dyaw + 360.0)
                    .otherwise(raw_dyaw)
                    .round(4)
                    .alias("delta_yaw"),
                    pl.col("pitch").diff().fill_null(0.0).round(4).alias("delta_pitch"),
                )

                sub = sub.with_columns(
                    [
                        ((pl.col("buttons").cast(pl.Int64).fill_null(0) & mask) != 0)
                        .cast(pl.Int8)
                        .alias(col)
                        for col, mask in KEY_COLUMNS.items()
                    ]
                )

                if keep_raw:
                    sub = sub.with_columns(
                        pl.col("buttons")
                        .map_elements(decode, return_dtype=pl.List(pl.Utf8))
                        .alias("buttons")
                    )

                # Any requested column the demo did not supply becomes null of
                # the right dtype, so the schema stays identical everywhere.
                missing = [c for c in cols if c not in sub.columns]
                if missing:
                    sub = sub.with_columns(
                        [pl.lit(None).cast(dtypes[c]).alias(c) for c in missing]
                    )

                sub = sub.select(cols).with_columns(
                    [pl.col(c).cast(dtypes[c]) for c in cols]
                )

                tmp = rdir / f".{sid}.parquet.tmp"
                sub.write_parquet(
                    tmp,
                    compression=opts["compression"],
                    compression_level=int(opts["compression_level"]),
                )
                tmp.replace(rdir / f"{sid}.parquet")
                written += 1

        return match_id, "ok", f"{written} trajectories over {len(todo_rounds)} rounds"

    except Exception as exc:
        return match_id, "failed", f"{type(exc).__name__}: {exc}"


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()
    acfg = cfg.actions

    candidates = []
    for match_id in layout.match_ids_with_demos():
        if match_ids and match_id not in set(match_ids):
            continue
        if not layout.metadata_file(match_id).exists():
            log.warning("No metadata for %s - run `mecs metadata` first; skipping.", match_id)
            continue
        candidates.append(match_id)

    if not candidates:
        log.error("Nothing to do: no (demo + metadata) pairs found under %s", layout.root)
        return {"ok": 0, "total": 0}

    if force:
        import shutil

        for match_id in candidates:
            shutil.rmtree(layout.state_action / f"match={match_id}", ignore_errors=True)

    workers = min(resolve_workers(acfg.workers, acfg.cpu_fraction), len(candidates))
    log.info("Extracting actions for %d matches with %d worker(s)", len(candidates), workers)

    opts = {
        "tickrate": acfg.tickrate,
        "compression": acfg.compression,
        "compression_level": acfg.compression_level,
        "keep_raw_buttons": acfg.keep_raw_buttons,
        "rounds": acfg.rounds,
    }
    payload = [
        (
            m,
            str(layout.demo_file(m)),
            str(layout.metadata_file(m)),
            str(layout.state_action),
            opts,
        )
        for m in candidates
    ]

    tally: dict[str, int] = {}
    failures: list[str] = []

    if workers == 1:
        results: Any = (_process_match(a) for a in payload)
        pool = None
    else:
        pool = cf.ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_process_match, payload)

    for i, (match_id, status, detail) in enumerate(results, start=1):
        tally[status] = tally.get(status, 0) + 1
        if status == "failed":
            failures.append(f"{match_id}: {detail}")
            log.warning("[%d/%d] FAILED %s  %s", i, len(candidates), match_id[:24], detail)
        else:
            log.info("[%d/%d] %-7s %s  %s", i, len(candidates), status, match_id[:24], detail)

    if pool is not None:
        pool.shutdown()

    log.info("Actions summary: %s", tally)
    if failures:
        log.warning("First failures:\n  %s", "\n  ".join(failures[:5]))
    return {"tally": tally, "failures": failures, "total": len(candidates)}
