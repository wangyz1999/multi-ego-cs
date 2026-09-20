"""Thin, rate-limited client for the FACEIT Data API v4.

Only the handful of endpoints stage 01/02 need are wrapped. The client is
deliberately boring: synchronous, retrying, and rate-limited, because FACEIT
returns 429 aggressively and a discovery run that trips the limit loses more
time to backoff than it ever gains from concurrency.

An API key is required. Create one at https://developers.faceit.com (free,
"Data API" scope) and export it:

    export MECS_FACEIT_API_KEY=<key>
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import requests

from ..util.logging_setup import get_logger

log = get_logger("faceit")

BASE_URL = "https://open.faceit.com/data/v4"

# Retried; anything else is raised immediately.
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class FaceitError(RuntimeError):
    """Non-retryable API failure."""


class _RateLimiter:
    """Sliding-window limiter, safe across threads."""

    def __init__(self, max_per_minute: int) -> None:
        self._max = max(1, max_per_minute)
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._events and now - self._events[0] > 60.0:
                    self._events.popleft()
                if len(self._events) < self._max:
                    self._events.append(now)
                    return
                sleep_for = 60.0 - (now - self._events[0]) + 0.01
            time.sleep(max(0.01, sleep_for))


class FaceitClient:
    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        requests_per_minute: int = 250,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise FaceitError(
                "No FACEIT API key. Export MECS_FACEIT_API_KEY "
                "(create one at https://developers.faceit.com)."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        )
        self._timeout = timeout
        self._limiter = _RateLimiter(requests_per_minute)
        self._max_retries = max_retries

    # -- plumbing -----------------------------------------------------------
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        backoff = 2.0
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            self._limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)
            except requests.RequestException as exc:  # network flake
                last_exc = exc
                log.warning("GET %s failed (%s), attempt %d/%d", path, exc, attempt, self._max_retries)
                time.sleep(backoff)
                backoff *= 2
                continue

            if resp.status_code == 404:
                raise FaceitError(f"404 for {path}")
            if resp.status_code in _RETRY_STATUS:
                wait = float(resp.headers.get("Retry-After", backoff))
                log.warning(
                    "GET %s -> %d, sleeping %.1fs (attempt %d/%d)",
                    path, resp.status_code, wait, attempt, self._max_retries,
                )
                time.sleep(wait)
                backoff *= 2
                continue
            if not resp.ok:
                raise FaceitError(f"{resp.status_code} for {path}: {resp.text[:200]}")
            return resp.json()

        raise FaceitError(f"Exhausted retries for {path}") from last_exc

    # -- endpoints ----------------------------------------------------------
    def top_players(self, region: str, limit: int, game: str = "cs2") -> list[dict[str, Any]]:
        """Walk the regional leaderboard. Paginates in blocks of 100."""
        out: list[dict[str, Any]] = []
        offset = 0
        while len(out) < limit:
            page = self._get(
                f"/rankings/games/{game}/regions/{region}",
                {"offset": offset, "limit": min(100, limit - len(out))},
            )
            items = page.get("items") or []
            if not items:
                break
            out.extend(items)
            offset += len(items)
        log.info("Leaderboard %s/%s: %d players", game, region, len(out))
        return out[:limit]

    def player_history(
        self, player_id: str, limit: int, game: str = "cs2"
    ) -> list[dict[str, Any]]:
        """Recent matches for one player, newest first."""
        out: list[dict[str, Any]] = []
        offset = 0
        while len(out) < limit:
            page = self._get(
                f"/players/{player_id}/history",
                {"game": game, "offset": offset, "limit": min(100, limit - len(out))},
            )
            items = page.get("items") or []
            if not items:
                break
            out.extend(items)
            offset += len(items)
        return out[:limit]

    def match(self, match_id: str) -> dict[str, Any]:
        return self._get(f"/matches/{match_id}")

    def match_stats(self, match_id: str) -> dict[str, Any]:
        return self._get(f"/matches/{match_id}/stats")


# ---------------------------------------------------------------------------
# Payload readers
#
# These exist because the shape of a FACEIT match payload is not stable across
# the Data API (`/matches/{id}`) and the internal payload archived alongside
# older collections. Both are handled so a config can point at either.
# ---------------------------------------------------------------------------
def read_picked_map(payload: dict[str, Any]) -> str | None:
    """Return the map actually played, e.g. ``de_mirage``.

    ``voting.map.pick`` is the only field that reliably holds the *decision*.
    ``voting.map.entities`` is the candidate pool and often still lists every
    map that survived the ban phase, so reading it yields the wrong answer (or
    no answer) for roughly 60% of matches.
    """
    for container in (payload, payload.get("payload") or {}):
        voting = (container.get("voting") or {}).get("map") or {}
        pick = voting.get("pick")
        if isinstance(pick, list) and len(pick) == 1:
            return pick[0]
        if isinstance(pick, str) and pick:
            return pick
        entities = voting.get("entities")
        if isinstance(entities, list) and len(entities) == 1:
            ent = entities[0]
            if isinstance(ent, dict):
                return ent.get("game_map_id") or ent.get("class_name")
            if isinstance(ent, str):
                return ent
    return None


def read_demo_urls(payload: dict[str, Any]) -> list[str]:
    """Return CDN URLs for the match demo(s)."""
    for container in (payload, payload.get("payload") or {}):
        for key in ("demo_url", "demoURLs", "demo_urls"):
            val = container.get(key)
            if isinstance(val, list) and val:
                return [u for u in val if isinstance(u, str)]
            if isinstance(val, str) and val:
                return [val]
    return []


def read_finished_at(payload: dict[str, Any]) -> str | None:
    for container in (payload, payload.get("payload") or {}):
        for key in ("finishedAt", "finished_at"):
            val = container.get(key)
            if val:
                return str(val)
    return None


def read_players(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten both teams into a single player list with steam ids."""
    players: list[dict[str, Any]] = []
    for container in (payload, payload.get("payload") or {}):
        teams = container.get("teams") or {}
        if not isinstance(teams, dict):
            continue
        for faction, team in teams.items():
            roster = (team or {}).get("roster") or (team or {}).get("players") or []
            for p in roster:
                if not isinstance(p, dict):
                    continue
                players.append(
                    {
                        "faction": faction,
                        "player_id": p.get("player_id") or p.get("id"),
                        "nickname": p.get("nickname"),
                        "steamid": (p.get("game_player_id") or p.get("gamePlayerId")),
                    }
                )
        if players:
            break
    return players
