"""Stage 07 - assemble the upload-ready release tree.

Turns the working directories into something a stranger can load:

    release/
      manifest/clips.csv                 one row per (match, round, player)
      manifest/rounds.csv                one row per (match, round)
      manifest/matches.csv               one row per match
      manifest/match_round_partitioned.csv   legacy 4-column index
      manifest/summary.json              counts, coverage, per-map totals
      metadata/<match_id>.json
      state_action/match=…/round=…/<steamid>.parquet
      align/match=…/round=…/offsets.json
      video[_<variant>]/match=…/round=…/<steamid>.mp4
      demo/<match_id>.dem                (only if publish includes it)

Two things this stage exists to get right:

**Splits are assigned per match, never per round.** Rounds inside one match
share players, economy state and map callouts; splitting them across train and
test leaks. Assignment is a deterministic hash of the match id, so adding
matches later does not reshuffle the existing ones.

**Modality coverage is recorded, not assumed.** A match can have video but no
demo (the demo expired before stage 02 ran), which means no action data will
ever exist for it. The manifest carries ``has_video`` / ``has_actions`` /
``has_align`` per clip, and ``summary.json`` reports the totals, so a consumer
filters instead of discovering the gap as a missing-file crash.
"""

from __future__ import annotations

import concurrent.futures as cf
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..config import Config, resolve_workers
from ..util.io_utils import du_bytes, human_bytes, materialize, read_json, write_json_atomic
from ..util.logging_setup import get_logger

log = get_logger("package")

CLIP_FIELDS = [
    "match_id", "round_number", "steamid", "player_name", "team_number", "side",
    "map_name", "split", "alive_start_tick", "alive_end_tick", "alive_duration_ticks",
    "alive_duration_sec", "died_in_round", "has_video", "has_actions", "has_align",
    "video_offset_sec", "video_fps", "video_bytes",
]
ROUND_FIELDS = [
    "match_id", "round_number", "map_name", "split", "n_players", "n_video",
    "n_actions", "winner", "reason", "bomb_planted", "bomb_site",
    "round_offset_sec", "round_offset_spread_sec",
]
MATCH_FIELDS = [
    "match_id", "map_name", "split", "n_rounds", "n_clips", "n_video", "n_actions",
    "has_demo", "usable", "total_video_bytes", "total_video_sec",
]
LEGACY_FIELDS = ["index", "split", "match_id", "round_number"]


def assign_split(match_id: str, splits: dict[str, float], seed: int) -> str:
    """Deterministic, stable-under-growth split assignment.

    Hashing the id (rather than shuffling a list) means a match keeps its split
    no matter how many other matches are added later - important when a dataset
    is published in waves and downstream papers cite "the test split".
    """
    if not splits:
        return "train"
    digest = hashlib.sha256(f"{seed}:{match_id}".encode()).digest()
    # 53 bits keeps the ratio exact in float64.
    position = int.from_bytes(digest[:7], "big") / float(1 << 56)
    cumulative = 0.0
    total = sum(splits.values()) or 1.0
    for name in sorted(splits):
        cumulative += splits[name] / total
        if position < cumulative:
            return name
    return sorted(splits)[-1]


def _side_for(meta: dict[str, Any], round_number: int, team_number: int | None) -> str | None:
    for rinfo in meta.get("rounds") or []:
        if rinfo.get("round_number") != round_number:
            continue
        if team_number is None:
            return None
        if rinfo.get("t_team_number") == team_number:
            return "T"
        if rinfo.get("ct_team_number") == team_number:
            return "CT"
    return None


def _round_info(meta: dict[str, Any], round_number: int) -> dict[str, Any]:
    for rinfo in meta.get("rounds") or []:
        if rinfo.get("round_number") == round_number:
            return rinfo
    return {}


def _structure_from_video(layout: Any, match_id: str) -> dict[str, list[dict[str, Any]]]:
    """Recover (round -> players) from the video tree alone.

    Used for matches whose demo expired before stage 02 could fetch it. The
    recordings are already segmented per round and per player, so the *shape*
    of the match is recoverable from directory names even though the tick-level
    facts (alive windows, kills, sides) are gone for good.

    Entries carry nulls where a demo would have supplied ticks, and the clip
    rows they produce have has_actions = 0. That is the honest representation:
    the video exists, the actions never will.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    match_dir = layout.video / f"match={match_id}"
    if not match_dir.is_dir():
        return out
    for round_dir in sorted(match_dir.iterdir()):
        if not round_dir.is_dir() or not round_dir.name.startswith("round="):
            continue
        round_key = round_dir.name.removeprefix("round=")
        entries = [
            {
                "steamid": clip.stem,
                "player_name": None,
                "team_number": None,
                "alive_start_tick": None,
                "alive_end_tick": None,
                "alive_duration_ticks": None,
                "died_in_round": None,
            }
            for clip in sorted(round_dir.glob("*.mp4"))
        ]
        if entries:
            out[round_key] = entries
    return out


def _clip_duration_seconds(path: Path) -> float | None:
    """Clip length from the container, for matches with no tick data."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        finally:
            cap.release()
        if fps > 0 and frames > 0:
            return round(frames / fps, 3)
    except Exception:
        pass
    return None


def _transcode(args: tuple[str, str, dict[str, Any], str]) -> tuple[str, str]:
    """Run ffmpeg for one clip. Returns (destination, status)."""
    src, dst, spec, ffmpeg = args
    dst_path = Path(dst)
    if dst_path.exists() and dst_path.stat().st_size > 0:
        return dst, "exists"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_suffix(".part.mp4")

    cmd = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", src]
    filters = []
    if spec.get("width") and spec.get("height"):
        filters.append(f"scale={spec['width']}:{spec['height']}")
    if spec.get("fps"):
        filters.append(f"fps={spec['fps']}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-c:v", spec.get("vcodec", "libx264")]
    if spec.get("crf") is not None:
        cmd += ["-crf", str(spec["crf"])]
    if spec.get("preset"):
        cmd += ["-preset", spec["preset"]]
    cmd += ["-an"] if spec.get("drop_audio", True) else ["-c:a", "copy"]
    cmd += [str(tmp)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return dst, f"failed: {proc.stderr.strip()[:160]}"
    tmp.replace(dst_path)
    return dst, "ok"


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    include_unusable: bool = False,
    include_demos: bool = False,
    skip_video: bool = False,
    fresh: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()
    pcfg = cfg.package
    release = layout.release

    if fresh and release.exists():
        import shutil

        log.warning("--fresh: removing the existing release tree at %s", release)
        shutil.rmtree(release)

    (release / "manifest").mkdir(parents=True, exist_ok=True)

    # Candidates are the union of two populations:
    #   * matches with stage-03 metadata (full tick-level structure)
    #   * matches with video but no demo, whose structure comes from the video
    #     tree - these can never have action data, and saying so explicitly is
    #     the point of the coverage columns.
    with_metadata = sorted(p.stem for p in layout.metadata.glob("*.json")) if layout.metadata.exists() else []
    with_video = layout.match_ids_with_video()
    video_only = (
        [m for m in with_video if m not in set(with_metadata)]
        if pcfg.include_video_only_matches
        else []
    )
    candidates = sorted(set(with_metadata) | set(video_only))

    if match_ids:
        wanted = set(match_ids)
        candidates = [m for m in candidates if m in wanted]
        video_only = [m for m in video_only if m in wanted]
    if not candidates:
        log.error(
            "Nothing to package: no metadata under %s and no video under %s.",
            layout.metadata, layout.video,
        )
        return {"matches": 0}

    # Map names for video-only matches come from the discovery index.
    index_maps: dict[str, str] = {}
    if video_only:
        from ..util.io_utils import read_jsonl as _read_jsonl

        for rec in _read_jsonl(layout.matches_index):
            mid, mp = rec.get("match_id"), rec.get("map")
            if mid and mp:
                index_maps[mid] = mp

    link_mode = pcfg.link_mode if getattr(pcfg, "link_mode", None) else (
        "auto" if pcfg.use_hardlinks else "copy"
    )
    log.info(
        "Packaging %d matches into %s (%d metadata-backed, %d video-only, link_mode=%s)",
        len(candidates), release, len(candidates) - len(video_only), len(video_only), link_mode,
    )

    clip_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    transcode_jobs: list[tuple[str, str, dict[str, Any], str]] = []
    copy_plan: list[tuple[Path, Path]] = []
    excluded: list[str] = []
    per_map: Counter[str] = Counter()
    coverage = Counter()

    for match_id in candidates:
        meta_path = layout.metadata_file(match_id)
        if meta_path.exists():
            meta = read_json(meta_path)
            if not meta.get("usable", True) and not include_unusable:
                excluded.append(match_id)
                continue
            map_name = meta.get("map_name") or index_maps.get(match_id) or "unknown"
            alive = meta.get("player_alive_times") or {}
        else:
            # Video-only match: no demo was ever obtained for it.
            meta = {}
            map_name = index_maps.get(match_id) or "unknown"
            alive = _structure_from_video(layout, match_id)
            if not alive:
                log.warning("%s has neither metadata nor video clips; skipping.", match_id)
                continue

        split = assign_split(match_id, pcfg.splits, pcfg.split_seed)
        has_demo = layout.demo_file(match_id).exists()

        m_clips = m_video = m_actions = 0
        m_bytes = 0
        m_seconds = 0.0

        for round_key in sorted(alive, key=lambda r: int(r)):
            round_number = int(round_key)
            entries = alive[round_key]
            rinfo = _round_info(meta, round_number)

            offsets_path = layout.offsets_file(match_id, round_key)
            offsets: dict[str, Any] = {}
            if offsets_path.exists():
                try:
                    offsets = read_json(offsets_path)
                except Exception:
                    offsets = {}
            per_player_offsets = offsets.get("players") or {}

            r_video = r_actions = 0

            for entry in entries:
                steamid = entry["steamid"]
                video_src = layout.video_file(match_id, round_key, steamid)
                actions_src = layout.state_action_file(match_id, round_key, steamid)
                has_video = video_src.exists()
                has_actions = actions_src.exists()
                pinfo = per_player_offsets.get(steamid) or {}
                has_align = "offset_sec" in pinfo

                video_bytes = video_src.stat().st_size if has_video else 0
                ticks = entry.get("alive_duration_ticks")
                if ticks is not None:
                    duration_sec = ticks / (meta.get("tickrate") or cfg.actions.tickrate)
                else:
                    duration_sec = _clip_duration_seconds(video_src) if has_video else None

                clip_rows.append(
                    {
                        "match_id": match_id,
                        "round_number": round_number,
                        "steamid": steamid,
                        "player_name": entry.get("player_name"),
                        "team_number": entry.get("team_number"),
                        "side": _side_for(meta, round_number, entry.get("team_number")),
                        "map_name": map_name,
                        "split": split,
                        "alive_start_tick": entry["alive_start_tick"],
                        "alive_end_tick": entry["alive_end_tick"],
                        "alive_duration_ticks": entry["alive_duration_ticks"],
                        "alive_duration_sec": (
                            round(duration_sec, 3) if duration_sec is not None else None
                        ),
                        "died_in_round": (
                            int(bool(entry["died_in_round"]))
                            if entry.get("died_in_round") is not None
                            else None
                        ),
                        "has_video": int(has_video),
                        "has_actions": int(has_actions),
                        "has_align": int(has_align),
                        "video_offset_sec": pinfo.get("offset_sec"),
                        "video_fps": pinfo.get("video_fps"),
                        "video_bytes": video_bytes,
                    }
                )

                coverage["clips"] += 1
                coverage["video"] += int(has_video)
                coverage["actions"] += int(has_actions)
                coverage["align"] += int(has_align)
                m_clips += 1
                r_video += int(has_video)
                r_actions += int(has_actions)
                m_video += int(has_video)
                m_actions += int(has_actions)
                m_bytes += video_bytes
                m_seconds += duration_sec or 0.0

                # -- plan the release-tree writes -----------------------------
                if has_actions:
                    copy_plan.append(
                        (actions_src, release / "state_action" / f"match={match_id}"
                         / f"round={round_key}" / f"{steamid}.parquet")
                    )
                if has_video and not skip_video:
                    for variant, spec in (pcfg.variants or {"native": {}}).items():
                        subdir = "video" if variant == "native" else f"video_{variant}"
                        dst = release / subdir / f"match={match_id}" / f"round={round_key}" / f"{steamid}.mp4"
                        if spec:
                            transcode_jobs.append((str(video_src), str(dst), spec, pcfg.ffmpeg))
                        else:
                            copy_plan.append((video_src, dst))

            if offsets_path.exists():
                copy_plan.append(
                    (offsets_path, release / "align" / f"match={match_id}"
                     / f"round={round_key}" / "offsets.json")
                )

            round_rows.append(
                {
                    "match_id": match_id,
                    "round_number": round_number,
                    "map_name": map_name,
                    "split": split,
                    "n_players": len(entries),
                    "n_video": r_video,
                    "n_actions": r_actions,
                    "winner": rinfo.get("winner"),
                    "reason": rinfo.get("reason"),
                    "bomb_planted": int(rinfo.get("bomb_plant_tick") is not None),
                    "bomb_site": rinfo.get("bomb_site"),
                    "round_offset_sec": offsets.get("round_offset_sec"),
                    "round_offset_spread_sec": offsets.get("round_offset_spread_sec"),
                }
            )

        if meta_path.exists():
            copy_plan.append((meta_path, release / "metadata" / f"{match_id}.json"))
        if include_demos and has_demo:
            copy_plan.append((layout.demo_file(match_id), release / "demo" / f"{match_id}.dem"))

        per_map[map_name] += 1
        match_rows.append(
            {
                "match_id": match_id,
                "map_name": map_name,
                "split": split,
                "n_rounds": len(alive),
                "n_clips": m_clips,
                "n_video": m_video,
                "n_actions": m_actions,
                "has_demo": int(has_demo),
                "usable": int(bool(meta.get("usable", True))),
                "total_video_bytes": m_bytes,
                "total_video_sec": round(m_seconds, 2),
            }
        )

    # -- materialise the tree -------------------------------------------------
    placed: Counter[str] = Counter()
    for src, dst in copy_plan:
        placed[materialize(src, dst, mode=link_mode)] += 1
    log.info("Release tree: %s", dict(placed))

    if transcode_jobs:
        workers = min(resolve_workers(pcfg.workers), len(transcode_jobs))
        if not shutil.which(pcfg.ffmpeg):
            log.error("ffmpeg (%s) not found - skipping %d transcodes. "
                      "On HPC: `module load ffmpeg`.", pcfg.ffmpeg, len(transcode_jobs))
        else:
            log.info("Transcoding %d clips with %d worker(s)", len(transcode_jobs), workers)
            failures = 0
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                for i, (dst, status) in enumerate(pool.map(_transcode, transcode_jobs), start=1):
                    if status.startswith("failed"):
                        failures += 1
                        log.warning("[%d/%d] %s  %s", i, len(transcode_jobs), status, dst)
                    elif i % 500 == 0:
                        log.info("[%d/%d] transcoded", i, len(transcode_jobs))
            if failures:
                log.warning("%d transcodes failed", failures)

    # -- manifests ------------------------------------------------------------
    manifest_dir = release / "manifest"

    def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(path)
        log.info("Wrote %s (%d rows)", path.name, len(rows))

    clip_rows.sort(key=lambda r: (r["match_id"], r["round_number"], r["steamid"]))
    round_rows.sort(key=lambda r: (r["match_id"], r["round_number"]))
    match_rows.sort(key=lambda r: r["match_id"])

    _write_csv(manifest_dir / "clips.csv", CLIP_FIELDS, clip_rows)
    _write_csv(manifest_dir / "rounds.csv", ROUND_FIELDS, round_rows)
    _write_csv(manifest_dir / "matches.csv", MATCH_FIELDS, match_rows)

    # Legacy index, byte-compatible with the column layout published earlier.
    # Written twice on purpose: under manifest/ alongside the richer tables, and
    # at the release root, which is where the previously published dataset kept
    # it. Emitting the root copy means an upload *corrects* the old index in
    # place rather than leaving a stale, shorter one beside the new manifests.
    legacy = [
        {"index": i, "split": r["split"], "match_id": r["match_id"], "round_number": r["round_number"]}
        for i, r in enumerate(round_rows)
    ]
    _write_csv(manifest_dir / "match_round_partitioned.csv", LEGACY_FIELDS, legacy)
    _write_csv(release / "match_round_partitioned.csv", LEGACY_FIELDS, legacy)

    split_counts = Counter(r["split"] for r in match_rows)
    split_round_counts = Counter(r["split"] for r in round_rows)
    map_by_split: dict[str, dict[str, int]] = defaultdict(dict)
    for r in match_rows:
        map_by_split[r["map_name"]][r["split"]] = map_by_split[r["map_name"]].get(r["split"], 0) + 1

    summary = {
        "schema_version": 1,
        "matches": len(match_rows),
        "rounds": len(round_rows),
        "clips": len(clip_rows),
        "maps": dict(per_map.most_common()),
        "maps_by_split": {k: dict(v) for k, v in map_by_split.items()},
        "matches_by_split": dict(split_counts),
        "rounds_by_split": dict(split_round_counts),
        "coverage": {
            "clips": coverage["clips"],
            "with_video": coverage["video"],
            "with_actions": coverage["actions"],
            "with_alignment": coverage["align"],
            "video_pct": round(100 * coverage["video"] / max(coverage["clips"], 1), 2),
            "actions_pct": round(100 * coverage["actions"] / max(coverage["clips"], 1), 2),
            "alignment_pct": round(100 * coverage["align"] / max(coverage["clips"], 1), 2),
        },
        "matches_without_demo": [r["match_id"] for r in match_rows if not r["has_demo"]],
        "video_only_matches": sorted(video_only),
        "excluded_unusable": excluded,
        "total_video_sec": round(sum(r["total_video_sec"] for r in match_rows), 2),
        "total_video_hours": round(sum(r["total_video_sec"] for r in match_rows) / 3600, 2),
        "split_policy": {
            "granularity": pcfg.split_by,
            "ratios": pcfg.splits,
            "seed": pcfg.split_seed,
            "method": "sha256(seed:match_id) -> [0,1)",
        },
    }
    write_json_atomic(manifest_dir / "summary.json", summary)

    log.info("=== release summary ===")
    log.info("matches %d | rounds %d | clips %d | %.1f h video",
             summary["matches"], summary["rounds"], summary["clips"], summary["total_video_hours"])
    log.info("maps: %s", json.dumps(summary["maps"]))
    log.info("splits (matches): %s", json.dumps(summary["matches_by_split"]))
    log.info("coverage: video %.1f%% | actions %.1f%% | aligned %.1f%%",
             summary["coverage"]["video_pct"], summary["coverage"]["actions_pct"],
             summary["coverage"]["alignment_pct"])
    if summary["matches_without_demo"]:
        log.warning("%d matches have no demo and therefore no action data: %s",
                    len(summary["matches_without_demo"]),
                    ", ".join(m[:24] for m in summary["matches_without_demo"][:5]))
    if excluded:
        log.warning("%d matches excluded by validation (use --include-unusable to keep)", len(excluded))
    log.info("release size: %s", human_bytes(du_bytes(release)))

    return summary
