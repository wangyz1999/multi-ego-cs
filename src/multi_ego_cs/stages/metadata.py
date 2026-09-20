"""Stage 03 - derive per-match structure from a demo.

Produces one JSON per match holding everything the later stages need without
re-parsing the demo:

* ``players``      - the 10 players in *demo order*, with normalised team ids
* ``rounds``       - tick boundaries, winner, reason, bomb plant/site
* ``kills``        - full kill feed with positions
* ``player_alive_times`` - per round, per player: the tick window in which the
  player was alive. **This is the contract with stage 04**: the recorder
  spectates each player for exactly this window, so one video clip == one
  player's life in one round.
* ``statistics``   - convenience aggregates

Why alive windows and not whole rounds: a dead player's camera follows whoever
killed them, which would silently contaminate an egocentric dataset with
another player's point of view.

Validation is *reported, not fatal*. At 1000 matches you will meet demos with
9 players, a missing side assignment, or a corrupt tick stream; one bad demo
must not abort the run. Problems are recorded under ``validation`` and the
match is flagged ``usable: false`` so stage 07 can exclude it.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config, resolve_workers
from ..util.io_utils import read_json, write_json_atomic
from ..util.logging_setup import get_logger

log = get_logger("metadata")

# CS2 team_number as it appears in the demo -> our 0/1 convention.
_TEAM_REMAP = {2: 0, 3: 1}

EXPECTED_PLAYERS = 10
EXPECTED_PER_SIDE = 5


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:  # NaN
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    return str(value)


def _as_steamid(value: Any) -> str | None:
    """Steam64 ids arrive as float64 in some columns; round-trip via int."""
    i = _as_int(value)
    return str(i) if i is not None else None


def _extract_players(demo_path: Path, problems: list[str]) -> list[dict[str, Any]]:
    """Player list in demo order, which is the order `spec_player N` follows."""
    from demoparser2 import DemoParser

    parser = DemoParser(str(demo_path))
    df = parser.parse_player_info()

    players: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sid = _as_steamid(row.get("steamid"))
        # Bots and placeholder entries have short ids.
        if not sid or len(sid) < 15:
            continue
        raw_team = _as_int(row.get("team_number"))
        players.append(
            {
                "steamid": sid,
                "name": _as_str(row.get("name")),
                "team_number": _TEAM_REMAP.get(raw_team, raw_team),
            }
        )

    if len(players) != EXPECTED_PLAYERS:
        problems.append(f"expected {EXPECTED_PLAYERS} players, found {len(players)}")
    for p in players:
        if p["team_number"] not in (0, 1):
            problems.append(f"player {p['name']} has team_number {p['team_number']}")
    return players


def _round_sides(dem: Any, problems: list[str]):
    """Per (round, player) side assignment, used to map T/CT onto team ids."""
    import polars as pl

    table = (
        dem.ticks.group_by(["round_num", "steamid"])
        .agg(
            [
                pl.col("side").filter(pl.col("side").is_not_null()).first().alias("side"),
                pl.col("name").filter(pl.col("name").is_not_null()).first().alias("name"),
            ]
        )
        .unique()
        .sort(["round_num", "steamid"])
    )

    nulls = table.filter(pl.col("side").is_null())
    if len(nulls):
        problems.append(f"{len(nulls)} (round, player) rows have no side assignment")

    counts = (
        table.filter(pl.col("side").is_not_null())
        .group_by(["round_num", "side"])
        .agg(pl.len().alias("n"))
    )
    bad = counts.filter(pl.col("n") != EXPECTED_PER_SIDE)
    if len(bad):
        problems.append(f"{len(bad)} (round, side) groups do not have {EXPECTED_PER_SIDE} players")

    return table


def _team_numbers_for_round(
    round_num: int, round_sides: Any, steamid_to_team: dict[str, int]
) -> tuple[int | None, int | None]:
    import polars as pl

    rows = round_sides.filter(pl.col("round_num") == round_num)
    t_ids: set[int] = set()
    ct_ids: set[int] = set()
    for row in rows.iter_rows(named=True):
        sid = _as_steamid(row.get("steamid"))
        side = (row.get("side") or "").upper()
        if sid is None or sid not in steamid_to_team:
            continue
        if side == "T":
            t_ids.add(steamid_to_team[sid])
        elif side == "CT":
            ct_ids.add(steamid_to_team[sid])
    t = next(iter(t_ids)) if len(t_ids) == 1 else None
    ct = next(iter(ct_ids)) if len(ct_ids) == 1 else None
    return t, ct


def _alive_windows(
    players: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    kills: list[dict[str, Any]],
    problems: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """For each round, the tick window each player was alive.

    Deaths are matched on player *name* rather than steamid: the kill feed
    stores steamids as float64, and Steam64 ids exceed float64's exact integer
    range, so a small number of ids round to a neighbouring value. Names are
    stable within a single match.
    """
    # round -> victim name -> earliest death tick
    deaths: dict[int, dict[str, int]] = {}
    for kill in kills:
        rnum = kill["round_number"]
        victim = kill["victim_name"]
        tick = kill["tick"]
        # weapon "world" marks environmental/invalid kills that do not
        # correspond to a real death event in the tick stream.
        if rnum is None or victim is None or tick is None or kill.get("weapon") == "world":
            continue
        per_round = deaths.setdefault(rnum, {})
        if victim not in per_round or tick < per_round[victim]:
            per_round[victim] = tick

    out: dict[str, list[dict[str, Any]]] = {}
    for rinfo in rounds:
        rnum = rinfo["round_number"]
        # The round goes live at freeze_end, not at `start` - the freeze time
        # before it is buy phase, where nobody can move.
        start = rinfo["freeze_end_tick"]
        end = rinfo["end_tick"]
        if rnum is None or start is None or end is None:
            problems.append(f"round {rnum} has incomplete tick boundaries")
            continue

        entries: list[dict[str, Any]] = []
        for player in players:
            alive_end = end
            died = False
            death_tick = deaths.get(rnum, {}).get(player["name"])
            if death_tick is not None:
                alive_end = death_tick
                died = True

            duration = alive_end - start
            if duration <= 0:
                problems.append(
                    f"round {rnum} player {player['name']}: non-positive alive duration {duration}"
                )
                duration = max(0, duration)

            entries.append(
                {
                    "steamid": player["steamid"],
                    "player_name": player["name"],
                    "team_number": player["team_number"],
                    "alive_start_tick": start,
                    "alive_end_tick": alive_end,
                    "alive_duration_ticks": duration,
                    "died_in_round": died,
                }
            )
        out[str(rnum)] = entries
    return out


def _statistics(
    rounds: list[dict[str, Any]], kills: list[dict[str, Any]], players: list[dict[str, Any]]
) -> dict[str, Any]:
    per_kills: Counter[str] = Counter()
    per_deaths: Counter[str] = Counter()
    for k in kills:
        if k["attacker_name"]:
            per_kills[k["attacker_name"]] += 1
        if k["victim_name"]:
            per_deaths[k["victim_name"]] += 1

    total = len(kills)
    hs = sum(1 for k in kills if k.get("headshot"))
    plants = sum(1 for r in rounds if r.get("bomb_plant_tick") is not None)

    return {
        "total_rounds": len(rounds),
        "total_kills": total,
        "total_players": len(players),
        "headshot_kills": hs,
        "headshot_percentage": round(hs / total * 100, 2) if total else 0.0,
        "round_end_reasons": dict(Counter(r["reason"] for r in rounds if r["reason"])),
        "round_winners": dict(Counter(r["winner"] for r in rounds if r["winner"])),
        "most_used_weapons": dict(Counter(k["weapon"] for k in kills if k["weapon"]).most_common(10)),
        "player_statistics": {
            p["steamid"]: {
                "name": p["name"],
                "team_number": p["team_number"],
                "kills": per_kills.get(p["name"], 0),
                "deaths": per_deaths.get(p["name"], 0),
                "kd_ratio": round(per_kills.get(p["name"], 0) / max(per_deaths.get(p["name"], 0), 1), 3),
            }
            for p in players
        },
        "bomb_statistics": {
            "total_plants": plants,
            "exploded": sum(1 for r in rounds if r.get("reason") == "bomb_exploded"),
            "defused": sum(1 for r in rounds if r.get("reason") == "bomb_defused"),
        },
    }


def extract(demo_path: Path, match_id: str) -> dict[str, Any]:
    """Parse one demo into the stage-03 metadata document."""
    from awpy import Demo

    problems: list[str] = []
    players = _extract_players(demo_path, problems)

    dem = Demo(str(demo_path))
    dem.parse()

    rounds: list[dict[str, Any]] = []
    for row in dem.rounds.to_pandas().to_dict("records"):
        rounds.append(
            {
                "round_number": _as_int(row.get("round_num")),
                "start_tick": _as_int(row.get("start")),
                "freeze_end_tick": _as_int(row.get("freeze_end")),
                "end_tick": _as_int(row.get("end")),
                "official_end_tick": _as_int(row.get("official_end")),
                "winner": _as_str(row.get("winner")),
                "reason": _as_str(row.get("reason")),
                "bomb_plant_tick": _as_int(row.get("bomb_plant")),
                "bomb_site": _as_str(row.get("bomb_site")),
            }
        )

    steamid_to_team = {p["steamid"]: p["team_number"] for p in players}
    sides = _round_sides(dem, problems)
    for rinfo in rounds:
        rnum = rinfo["round_number"]
        if rnum is None:
            continue
        t, ct = _team_numbers_for_round(rnum, sides, steamid_to_team)
        rinfo["t_team_number"] = t
        rinfo["ct_team_number"] = ct
        if t is not None and ct is not None and t == ct:
            problems.append(f"round {rnum}: T and CT resolved to the same team id {t}")

    kills: list[dict[str, Any]] = []
    for row in dem.kills.to_pandas().to_dict("records"):
        kills.append(
            {
                "tick": _as_int(row.get("tick")),
                "round_number": _as_int(row.get("round_num")),
                "victim_steamid": _as_steamid(row.get("victim_steamid")),
                "victim_name": _as_str(row.get("victim_name")),
                "victim_side": _as_str(row.get("victim_side")),
                "attacker_steamid": _as_steamid(row.get("attacker_steamid")),
                "attacker_name": _as_str(row.get("attacker_name")),
                "attacker_side": _as_str(row.get("attacker_side")),
                "weapon": _as_str(row.get("weapon")),
                "headshot": bool(row.get("headshot")) if row.get("headshot") is not None else False,
                "victim_position": {
                    "x": _as_float(row.get("victim_X")),
                    "y": _as_float(row.get("victim_Y")),
                    "z": _as_float(row.get("victim_Z")),
                },
                "attacker_position": {
                    "x": _as_float(row.get("attacker_X")),
                    "y": _as_float(row.get("attacker_Y")),
                    "z": _as_float(row.get("attacker_Z")),
                },
            }
        )

    alive = _alive_windows(players, rounds, kills, problems)
    header = dem.header if isinstance(dem.header, dict) else dict(dem.header or {})

    return {
        "schema_version": 1,
        "match_id": match_id,
        "demo_file": demo_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "map_name": header.get("map_name"),
        "tickrate": header.get("tickrate"),
        "header": header,
        "players": players,
        "rounds": rounds,
        "kills": kills,
        "player_alive_times": alive,
        "statistics": _statistics(rounds, kills, players),
        "validation": {"problems": problems, "usable": not problems},
        "usable": not problems,
    }


def _worker(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    match_id, demo_str, out_str = args
    try:
        doc = extract(Path(demo_str), match_id)
        write_json_atomic(Path(out_str), doc)
        n_rounds = len(doc["rounds"])
        flag = "" if doc["usable"] else f" [{len(doc['validation']['problems'])} problems]"
        return match_id, True, f"{doc.get('map_name')} {n_rounds} rounds{flag}"
    except Exception as exc:
        return match_id, False, f"{type(exc).__name__}: {exc}"


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()
    available = layout.match_ids_with_demos()
    if match_ids:
        wanted = set(match_ids)
        available = [m for m in available if m in wanted]

    if not available:
        log.error("No demos in %s - run `mecs download` first.", layout.demo)
        return {"ok": 0, "total": 0}

    todo = [
        m for m in available if force or not layout.metadata_file(m).exists()
    ]
    log.info(
        "%d demos, %d already have metadata, %d to parse",
        len(available), len(available) - len(todo), len(todo),
    )
    if not todo:
        return {"ok": 0, "skipped": len(available), "total": len(available)}

    workers = min(resolve_workers(cfg.actions.workers, cfg.actions.cpu_fraction), len(todo))
    # Demo parsing is memory-hungry (a full tick table per demo). Cap workers
    # so a 101-match batch does not OOM a shared node.
    log.info("Parsing with %d worker process(es)", workers)

    payload = [
        (m, str(layout.demo_file(m)), str(layout.metadata_file(m))) for m in todo
    ]

    ok = failed = 0
    unusable: list[str] = []
    if workers == 1:
        results = (_worker(a) for a in payload)
    else:
        pool = cf.ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_worker, payload)

    for i, (match_id, success, detail) in enumerate(results, start=1):
        if success:
            ok += 1
            if "problems]" in detail:
                unusable.append(match_id)
            log.info("[%d/%d] ok     %s  %s", i, len(todo), match_id[:24], detail)
        else:
            failed += 1
            log.warning("[%d/%d] FAILED %s  %s", i, len(todo), match_id[:24], detail)

    if workers > 1:
        pool.shutdown()

    log.info("Metadata: %d ok, %d failed, %d flagged with validation problems",
             ok, failed, len(unusable))
    if unusable:
        log.warning("Flagged matches (excluded from release unless --include-unusable): %s",
                    ", ".join(m[:24] for m in unusable[:10]))
    return {"ok": ok, "failed": failed, "unusable": unusable, "total": len(todo)}


def load_metadata(cfg: Config, match_id: str) -> dict[str, Any] | None:
    p = cfg.layout.metadata_file(match_id)
    if not p.exists():
        return None
    try:
        return read_json(p)
    except Exception as exc:
        log.warning("Unreadable metadata for %s: %s", match_id, exc)
        return None
