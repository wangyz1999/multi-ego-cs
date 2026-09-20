"""Stage 06 - align each video clip to the tick timeline.

The recorder starts capture a fraction of a second *after* it issues
``demo_resume``, and that latency varies per clip (GPU capture ramp-up, input
lag, the game hitching). So a clip's first frame is not tick
``alive_start_tick``, and naively pairing frame *i* with tick
``start + i * 64/fps`` drifts by a random few hundred milliseconds per clip -
enough to put a mouse flick on the wrong frame.

This stage measures that latency per clip by finding a landmark visible in both
clocks: the frame on which the HUD timer ticks over to a new second. A timer
transition is, by construction, a whole-second boundary in game time, which
gives one exact (video_time, game_time) correspondence per clip.

    offset = video_time_of_transition - game_time_of_transition

with the sign convention

    video_time = game_sec + offset
    game_sec   = video_time - offset

Robustness
----------
* Digit reads below ``min_match_score`` are discarded, as are readings that are
  arithmetically impossible (tens > 5, clock above the round time).
* A transition only counts when the reading before and after are both
  confident and exactly one second apart. Single-frame glitches - a flashbang,
  an overlay - therefore cannot fabricate a landmark.
* Per-player offsets are compared against the round median, and any player more
  than ``max_offset_deviation_seconds`` away is flagged. All ten clips of a
  round were started by the same automation within a few seconds, so a genuine
  outlier means a misread, not a real latency.

Output: ``align/match=<id>/round=<n>/offsets.json``.
"""

from __future__ import annotations

import concurrent.futures as cf
import statistics
from pathlib import Path
from typing import Any

from ..config import Config, resolve_workers
from ..util.io_utils import write_json_atomic
from ..util.logging_setup import get_logger
from .hud_timer import (
    load_templates,
    read_timer,
    resolve_templates_dir,
    scale_boxes,
    timer_to_game_sec,
)

log = get_logger("align")


def measure_clip(
    video_path: Path,
    templates_dir: str,
    boxes: list[tuple[int, int, int, int]],
    reference_size: tuple[int, int],
    round_clock_seconds: int,
    min_match_score: float,
    max_scan_seconds: float,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Find the first clean timer transition and derive the clip's offset."""
    import cv2

    templates = load_templates(templates_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "cannot open video"}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not fps or fps != fps or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or reference_size[0])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or reference_size[1])
        scaled = scale_boxes(boxes, width, height, *reference_size)

        max_frames = int(max_scan_seconds * fps)
        prev = None
        first_text: str | None = None
        low_score_frames = 0
        frames_read = 0

        for frame_index in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames_read += 1

            reading = read_timer(frame, templates, scaled)
            confident = (
                reading.score >= min_match_score
                and reading.is_plausible(round_clock_seconds)
            )
            if verbose:
                log.debug(
                    "  f%-4d %s score=%.3f%s",
                    frame_index, reading.text, reading.score,
                    "" if confident else "  (rejected)",
                )
            if not confident:
                low_score_frames += 1
                continue
            if first_text is None:
                first_text = reading.text

            if prev is not None:
                elapsed = prev.total_seconds - reading.total_seconds
                # A countdown transition of exactly one second is the landmark.
                if elapsed == 1:
                    game_sec = timer_to_game_sec(reading, round_clock_seconds)
                    video_sec = frame_index / fps
                    return {
                        "offset_sec": round(video_sec - game_sec, 6),
                        "video_fps": round(float(fps), 6),
                        "frame_size": [width, height],
                        "start_timer": first_text,
                        "transition_timer": reading.text,
                        "transition_frame": frame_index,
                        "transition_video_sec": round(video_sec, 6),
                        "transition_game_sec": round(game_sec, 6),
                        "match_score": round(reading.score, 4),
                        "rejected_frames": low_score_frames,
                    }
                if elapsed < 0 or elapsed > 1:
                    # Clock jumped: a misread, or the clip spans a round reset.
                    prev = reading
                    continue
            prev = reading

        return {
            "error": "no clean timer transition found",
            "scanned_frames": frames_read,
            "rejected_frames": low_score_frames,
            "video_fps": round(float(fps), 6),
            "frame_size": [width, height],
        }
    finally:
        cap.release()


def _process_round(args: tuple[str, str, str, str, dict[str, Any]]) -> tuple[str, str, str, str]:
    """Align every clip in one round. Returns (match, round, status, detail)."""
    match_id, round_num, video_round_dir, out_path, opts = args
    try:
        vdir = Path(video_round_dir)
        clips = sorted(vdir.glob("*.mp4"))
        if not clips:
            return match_id, round_num, "skipped", "no clips"

        players: dict[str, Any] = {}
        offsets: list[float] = []
        failures = 0

        for clip in clips:
            steamid = clip.stem
            result = measure_clip(
                clip,
                opts["templates_dir"],
                opts["boxes"],
                tuple(opts["reference_size"]),
                opts["round_clock_seconds"],
                opts["min_match_score"],
                opts["max_scan_seconds"],
            )
            if result is None or "error" in result:
                failures += 1
                players[steamid] = result or {"error": "unknown"}
                continue
            players[steamid] = result
            offsets.append(result["offset_sec"])

        if not offsets:
            payload = {
                "schema_version": 1,
                "match_id": match_id,
                "round": int(round_num),
                "players": players,
                "round_offset_sec": None,
                "usable": False,
                "note": "no player clip yielded a timer transition",
            }
            write_json_atomic(out_path, payload)
            return match_id, round_num, "failed", f"0/{len(clips)} clips aligned"

        median = statistics.median(offsets)
        tol = opts["max_offset_deviation_seconds"]
        outliers = []
        for steamid, info in players.items():
            if "offset_sec" not in info:
                continue
            deviation = abs(info["offset_sec"] - median)
            info["deviation_from_median_sec"] = round(deviation, 6)
            info["outlier"] = deviation > tol
            if info["outlier"]:
                outliers.append(steamid)

        trusted = [
            info["offset_sec"]
            for info in players.values()
            if "offset_sec" in info and not info.get("outlier")
        ]

        payload = {
            "schema_version": 1,
            "match_id": match_id,
            "round": int(round_num),
            "sign_convention": "video_time = game_sec + offset_sec",
            "round_clock_seconds": opts["round_clock_seconds"],
            "players": players,
            "round_offset_sec": round(statistics.mean(trusted), 6) if trusted else round(median, 6),
            "round_offset_median_sec": round(median, 6),
            "round_offset_spread_sec": round(max(offsets) - min(offsets), 6),
            "aligned_players": len(offsets),
            "total_players": len(clips),
            "outliers": outliers,
            "usable": len(trusted) >= 1,
        }
        write_json_atomic(out_path, payload)

        detail = f"{len(offsets)}/{len(clips)} aligned, median {median:+.3f}s, spread {payload['round_offset_spread_sec']:.3f}s"
        if outliers:
            detail += f", {len(outliers)} outlier(s)"
        return match_id, round_num, "ok" if not failures else "partial", detail

    except Exception as exc:
        return match_id, round_num, "failed", f"{type(exc).__name__}: {exc}"


def _iter_rounds(cfg: Config, match_ids: list[str] | None):
    layout = cfg.layout
    if not layout.video.exists():
        return
    for match_dir in sorted(layout.video.iterdir()):
        if not match_dir.is_dir() or not match_dir.name.startswith("match="):
            continue
        match_id = match_dir.name.removeprefix("match=")
        if match_ids and match_id not in set(match_ids):
            continue
        for round_dir in sorted(match_dir.iterdir()):
            if not round_dir.is_dir() or not round_dir.name.startswith("round="):
                continue
            yield match_id, round_dir.name.removeprefix("round="), round_dir


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()
    acfg = cfg.align
    templates_dir = str(resolve_templates_dir(acfg.templates_dir or None))
    log.info("Digit templates: %s", templates_dir)

    work = []
    for match_id, round_num, round_dir in _iter_rounds(cfg, match_ids):
        out = layout.offsets_file(match_id, round_num)
        if out.exists() and not force:
            continue
        work.append(
            (
                match_id,
                round_num,
                str(round_dir),
                str(out),
                {
                    "templates_dir": templates_dir,
                    "boxes": [tuple(b) for b in acfg.digit_boxes],
                    "reference_size": [acfg.reference_width, acfg.reference_height],
                    "round_clock_seconds": acfg.round_clock_seconds,
                    "min_match_score": acfg.min_match_score,
                    "max_scan_seconds": acfg.max_scan_seconds,
                    "max_offset_deviation_seconds": acfg.max_offset_deviation_seconds,
                },
            )
        )

    if not work:
        log.info("Nothing to align (use --force to recompute existing offsets).")
        return {"total": 0}

    workers = min(resolve_workers(acfg.workers), len(work))
    log.info("Aligning %d rounds with %d worker(s)", len(work), workers)

    tally: dict[str, int] = {}
    spreads: list[float] = []

    if workers == 1:
        results: Any = (_process_round(a) for a in work)
        pool = None
    else:
        pool = cf.ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_process_round, work)

    for i, (match_id, round_num, status, detail) in enumerate(results, start=1):
        tally[status] = tally.get(status, 0) + 1
        emit = log.info if status in ("ok", "skipped") else log.warning
        emit("[%d/%d] %-8s %s r%-3s %s", i, len(work), status, match_id[:24], round_num, detail)

    if pool is not None:
        pool.shutdown()

    log.info("Align summary: %s", tally)
    if tally.get("failed"):
        log.warning(
            "%d rounds produced no usable offset. Common causes: the HUD timer "
            "is not at the configured crop boxes (check align.digit_boxes against "
            "a real frame), the clip starts after the first second boundary "
            "(raise align.max_scan_seconds), or the capture resolution differs "
            "from align.reference_width/height.",
            tally["failed"],
        )
    return {"tally": tally, "total": len(work)}
