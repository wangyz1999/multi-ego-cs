"""Stage 01 - decide which matches to collect.

FACEIT has no "give me N recent competitive matches" endpoint, so discovery
fans out over the regional leaderboard: walk top players, pull each player's
recent match history, deduplicate, then fetch each candidate's full payload to
read the map, the demo URL and the roster.

Output is ``<root>/matches.jsonl``, one record per accepted match::

    {"match_id": "1-…-1-1", "bare_id": "1-…", "map": "de_mirage",
     "demo_urls": [...], "finished_at": "2025-09-12T00:03:20Z",
     "players": [...], "discovered_via": "<player_id>"}

The file is append-only and deduplicated on read, so an interrupted run can be
restarted and will skip whatever it already found. Scaling from 100 to 1000
matches is a matter of raising ``discover.target_matches`` and letting it run
longer - see docs/COLLECTING_AT_SCALE.md.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..util import ids as idutil
from ..util.io_utils import append_jsonl, read_jsonl
from ..util.logging_setup import get_logger
from .faceit_api import (
    FaceitClient,
    FaceitError,
    read_demo_urls,
    read_finished_at,
    read_picked_map,
    read_players,
)

log = get_logger("discover")


def _load_existing(path) -> dict[str, dict[str, Any]]:
    """Existing accepted matches, keyed by canonical id (last write wins)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path):
        mid = rec.get("match_id")
        if mid:
            out[mid] = rec
    return out


def _accepts(
    payload: dict[str, Any],
    cfg: Config,
    picked_map: str | None,
) -> tuple[bool, str]:
    """Filter a candidate match. Returns (accepted, reason_if_rejected)."""
    dcfg = cfg.discover

    status = (payload.get("status") or payload.get("state") or "").upper()
    if status and status not in {"FINISHED", "COMPLETED"}:
        return False, f"status={status}"

    if not picked_map:
        return False, "map-undecided"
    if dcfg.maps and picked_map not in dcfg.maps:
        return False, f"map={picked_map}"

    if not read_demo_urls(payload):
        return False, "no-demo-url"

    # Round count lives in results when present; a forfeit shows 1-0.
    results = payload.get("results") or []
    if isinstance(results, list) and results and dcfg.min_rounds:
        factions = (results[0] or {}).get("factions") or {}
        scores = [int((v or {}).get("score", 0) or 0) for v in factions.values()]
        if scores and sum(scores) < dcfg.min_rounds:
            return False, f"rounds={sum(scores)}"

    return True, ""


def run(cfg: Config, dry_run: bool = False) -> dict[str, Any]:
    """Discover matches until ``target_matches`` accepted records exist."""
    layout = cfg.layout.ensure()
    dcfg = cfg.discover
    index_path = layout.matches_index

    existing = _load_existing(index_path)
    log.info("Existing accepted matches: %d (target %d)", len(existing), dcfg.target_matches)
    if len(existing) >= dcfg.target_matches:
        log.info("Target already met; nothing to do.")
        return {"accepted": len(existing), "new": 0, "examined": 0}

    client = FaceitClient(
        api_key=cfg.faceit_api_key or "",
        timeout=dcfg.request_timeout,
        requests_per_minute=dcfg.requests_per_minute,
    )

    players = client.top_players(dcfg.region, dcfg.num_players)
    if not players:
        raise FaceitError(f"Leaderboard for region {dcfg.region!r} returned no players")

    # Candidates already examined (accepted or not) so a restart does not
    # re-fetch payloads it has already judged.
    seen_bare: set[str] = {rec.get("bare_id", "") for rec in existing.values()}
    rejected: dict[str, str] = {}

    examined = 0
    new = 0

    for pidx, player in enumerate(players, start=1):
        if len(existing) + new >= dcfg.target_matches:
            break

        player_id = player.get("player_id")
        nickname = player.get("nickname", "?")
        elo = int(player.get("faceit_elo") or 0)
        if dcfg.min_elo and elo and elo < dcfg.min_elo:
            log.debug("Skip %s (elo %d < %d)", nickname, elo, dcfg.min_elo)
            continue
        if not player_id:
            continue

        try:
            history = client.player_history(player_id, dcfg.matches_per_player)
        except FaceitError as exc:
            log.warning("History failed for %s: %s", nickname, exc)
            continue

        log.info(
            "[%d/%d] %s (elo %d): %d candidate matches | accepted %d/%d",
            pidx, len(players), nickname, elo, len(history),
            len(existing) + new, dcfg.target_matches,
        )

        for item in history:
            if len(existing) + new >= dcfg.target_matches:
                break

            bare = item.get("match_id")
            if not bare or bare in seen_bare or bare in rejected:
                continue
            seen_bare.add(bare)
            examined += 1

            try:
                payload = client.match(bare)
            except FaceitError as exc:
                rejected[bare] = f"payload-error:{exc}"
                continue

            picked_map = read_picked_map(payload)
            ok, reason = _accepts(payload, cfg, picked_map)
            if not ok:
                rejected[bare] = reason
                log.debug("  reject %s (%s)", bare[:20], reason)
                continue

            demo_urls = read_demo_urls(payload)
            # The demo filename carries the canonical suffix; derive it from
            # the URL when possible rather than assuming "-1-1".
            canonical = idutil.canonical_id(bare)
            for url in demo_urls:
                found = idutil.match_id_from_path(url)
                if found and idutil.is_suffixed(found):
                    canonical = found
                    break

            record = {
                "match_id": canonical,
                "bare_id": bare,
                "map": picked_map,
                "demo_urls": demo_urls,
                "finished_at": read_finished_at(payload),
                "region": dcfg.region,
                "competition": payload.get("competition_name") or payload.get("competition_type"),
                "players": read_players(payload),
                "discovered_via": player_id,
            }

            if dry_run:
                log.info("  [dry-run] would accept %s (%s)", canonical, picked_map)
            else:
                append_jsonl(index_path, record)
            new += 1
            log.info("  accept %s  %s  (%d/%d)",
                     canonical[:24], picked_map, len(existing) + new, dcfg.target_matches)

    total = len(existing) + new
    log.info(
        "Discovery finished: %d accepted (%d new), %d examined, %d rejected",
        total, new, examined, len(rejected),
    )
    if total < dcfg.target_matches:
        log.warning(
            "Only %d/%d matches found. Raise discover.num_players or "
            "matches_per_player, widen discover.maps, or lower min_elo.",
            total, dcfg.target_matches,
        )

    # Reason histogram makes it obvious which filter is doing the cutting.
    if rejected:
        from collections import Counter

        hist = Counter(r.split(":")[0].split("=")[0] for r in rejected.values())
        log.info("Rejection reasons: %s", dict(hist.most_common()))

    return {"accepted": total, "new": new, "examined": examined, "rejected": len(rejected)}


def load_matches(cfg: Config) -> list[dict[str, Any]]:
    """Read the discovery index, deduplicated, sorted by match id."""
    recs = _load_existing(cfg.layout.matches_index)
    return [recs[k] for k in sorted(recs)]
