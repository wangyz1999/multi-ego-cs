"""Stage 04 - replay demos in CS2 and capture one clip per player per round.

**This stage runs on Windows, on a machine with CS2 installed.** Everything
else in the pipeline runs on Linux/HPC. It is the only human-supervised,
wall-clock-bound step: capture happens in real time, so recording a match takes
about as long as the match lasted. Budget roughly 45-60 minutes per match for
10 players x ~24 rounds, and see docs/COLLECTING_AT_SCALE.md for how to shard
across several machines.

How it works
------------
CS2 has no scriptable recording API, so the recorder drives the game the way a
human would: it focuses the window, types console commands, and toggles an
external capture tool's hotkey.

For each (round, player) in the stage-03 metadata:

    demo_gototick <alive_start_tick>
    spec_player <slot>
    demo_pause ; demo_gototick <alive_start_tick> ; demo_resume
    <capture hotkey on>   ... alive_duration_ticks / tickrate seconds ...
    <capture hotkey off>

The double ``demo_gototick`` around ``demo_pause`` is not redundant: seeking
while the demo plays lands a few ticks past the target, so the recorder seeks,
pauses, seeks again to land exactly, then resumes. Without it, clips start
mid-action and the stage-06 offsets grow a systematic bias.

Progress is appended to a log after every clip, so an interrupted session
(crash, reboot, closing the game) resumes at the next unrecorded clip rather
than starting the match over.

Capture tool
------------
Any hotkey-toggled recorder works (NVIDIA ShadowPlay, AMD ReLive, OBS with a
bound hotkey). Configure ``record.capture_output_dir`` to wherever it writes
finished clips and ``record.capture_hotkey`` to its toggle. The recorder claims
the newest file in that directory after each clip, so the directory must not be
shared with other recordings while a session runs.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Config
from ..util.io_utils import read_json
from ..util.logging_setup import get_logger

log = get_logger("record")

RECORD_LOG = "record_progress.tsv"


def _require_windows_deps() -> tuple[Any, Any, Any]:
    """Import the Windows-only automation stack with a useful error."""
    try:
        import psutil
        import pyautogui
        import pygetwindow
    except ImportError as exc:  # pragma: no cover - platform dependent
        raise RuntimeError(
            "Stage 04 needs the `record` extra on a Windows machine:\n"
            "    pip install 'multi-ego-cs[record]'\n"
            f"(missing: {exc.name})"
        ) from exc
    return pyautogui, pygetwindow, psutil


class ProgressLog:
    """Append-only record of completed clips, keyed (match, round, player)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._done: set[tuple[str, str, str]] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 4:
                    self._done.add((parts[1], parts[2], parts[3]))
        log.info("Progress log has %d completed clips", len(self._done))

    def done(self, match_id: str, round_num: str, steamid: str) -> bool:
        return (match_id, round_num, steamid) in self._done

    def mark(self, match_id: str, round_num: str, steamid: str, filename: str) -> None:
        self._done.add((match_id, round_num, steamid))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="seconds")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp}\t{match_id}\t{round_num}\t{steamid}\t{filename}\n")
            fh.flush()


class CS2Console:
    """Types commands into the CS2 developer console via keystroke injection."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.record
        self.pyautogui, self.gw, self.psutil = _require_windows_deps()
        self.pyautogui.FAILSAFE = True

    def focus_game(self, title: str = "Counter-Strike 2") -> bool:
        windows = self.gw.getWindowsWithTitle(title)
        if not windows:
            log.warning("No window titled %r", title)
            return False
        try:
            windows[0].activate()
        except Exception as exc:
            log.warning("Could not focus game window: %s", exc)
            return False
        time.sleep(0.5)
        return True

    def game_running(self) -> bool:
        for proc in self.psutil.process_iter(["name"]):
            name = (proc.info.get("name") or "").lower()
            if "cs2" in name:
                return True
        return False

    def launch_game(self) -> None:
        log.info("Launching CS2 via %s", self.cfg.cs2_launch_target)
        subprocess.Popen(["cmd", "/c", "start", "", self.cfg.cs2_launch_target], shell=False)
        time.sleep(self.cfg.game_boot_wait_seconds)

    def hide_cursor(self) -> None:
        # Parking the pointer keeps it out of the captured frame.
        self.pyautogui.moveTo(1, 1)

    def command(self, text: str, end_delay: float = 0.0) -> None:
        delay = self.cfg.command_pause_seconds
        self.pyautogui.press("`")
        time.sleep(delay)
        self.pyautogui.write(text)
        time.sleep(delay)
        self.pyautogui.press("enter")
        time.sleep(delay)
        self.pyautogui.press("`")
        self.hide_cursor()
        time.sleep(delay)
        log.debug("console: %s", text)
        if end_delay:
            time.sleep(end_delay)


class Capture:
    """Hotkey-toggled external screen recorder."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg.record
        self.pyautogui, _, _ = _require_windows_deps()
        self.output_dir = Path(self.cfg.capture_output_dir)
        self.recording = False
        if not self.cfg.capture_output_dir:
            raise RuntimeError(
                "record.capture_output_dir is unset - point it at your capture "
                "tool's output folder (e.g. the NVIDIA/ShadowPlay videos folder)."
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _toggle(self) -> None:
        self.pyautogui.hotkey(*self.cfg.capture_hotkey)

    def start(self) -> None:
        if self.recording:
            raise RuntimeError("capture already running")
        self._toggle()
        self.recording = True

    def stop(self) -> None:
        if not self.recording:
            raise RuntimeError("capture not running")
        self._toggle()
        self.recording = False

    def record_for(self, seconds: float) -> None:
        self.start()
        time.sleep(seconds)
        self.stop()

    def claim_newest(self, destination: Path, timeout: float = 20.0) -> Path:
        """Move the newest clip out of the capture folder to `destination`.

        Waits for the file to stop growing, because the capture tool finishes
        muxing after the hotkey stops recording.
        """
        deadline = time.monotonic() + timeout
        newest: Path | None = None
        while time.monotonic() < deadline:
            clips = sorted(
                self.output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if clips:
                candidate = clips[0]
                size = candidate.stat().st_size
                time.sleep(0.6)
                if candidate.exists() and candidate.stat().st_size == size and size > 0:
                    newest = candidate
                    break
            time.sleep(0.4)

        if newest is None:
            raise FileNotFoundError(f"No finished clip appeared in {self.output_dir}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(newest), str(destination))
        return destination


def record_match(cfg: Config, match_id: str, console: CS2Console, capture: Capture,
                 progress: ProgressLog) -> dict[str, int]:
    """Record every clip for one match. Returns per-outcome counts."""
    layout = cfg.layout
    rcfg = cfg.record

    metadata_path = layout.metadata_file(match_id)
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No metadata for {match_id}. Run `mecs metadata` before recording."
        )
    meta = read_json(metadata_path)
    alive = meta.get("player_alive_times") or {}
    if not alive:
        raise ValueError(f"{match_id}: metadata has no player_alive_times")

    # CS2 plays demos from its own csgo directory; copy in, and remove after.
    game_demo = Path(rcfg.cs2_csgo_dir) / f"{match_id}.dem"
    source_demo = layout.demo_file(match_id)
    if not game_demo.exists():
        if not source_demo.exists():
            raise FileNotFoundError(f"Missing demo: {source_demo}")
        log.info("Copying demo into game folder: %s", game_demo)
        shutil.copy2(source_demo, game_demo)

    counts = {"recorded": 0, "skipped": 0, "failed": 0}
    try:
        if not console.game_running():
            console.launch_game()
        console.focus_game()

        console.command(f"playdemo {match_id}")
        log.info("Waiting %.0fs for the demo to load", rcfg.demo_load_wait_seconds)
        time.sleep(rcfg.demo_load_wait_seconds)

        # Strip HUD elements that would otherwise burn debug text and the
        # spectator x-ray outline into every frame.
        console.command("r_show_build_info 0")
        console.command("spec_show_xray 0")
        console.command("demoui")

        for round_num in sorted(alive, key=lambda r: int(r)):
            entries = alive[round_num]
            for slot_index, entry in enumerate(entries, start=rcfg.spec_player_offset):
                steamid = entry["steamid"]
                if progress.done(match_id, round_num, steamid):
                    counts["skipped"] += 1
                    continue

                target = layout.video_file(match_id, round_num, steamid)
                if target.exists():
                    progress.mark(match_id, round_num, steamid, target.name)
                    counts["skipped"] += 1
                    continue

                duration = entry["alive_duration_ticks"] / rcfg.tickrate
                if duration <= 0:
                    log.warning(
                        "round %s player %s has zero alive duration - skipping",
                        round_num, entry.get("player_name"),
                    )
                    counts["skipped"] += 1
                    continue

                log.info(
                    "round %s  %s (%s)  slot %d  %.1fs",
                    round_num, entry.get("player_name"), steamid, slot_index, duration,
                )
                try:
                    if not console.game_running():
                        log.warning("CS2 exited; relaunching and reloading the demo")
                        console.launch_game()
                        console.focus_game()
                        console.command(f"playdemo {match_id}")
                        time.sleep(rcfg.demo_load_wait_seconds)
                        console.command("demoui")

                    console.focus_game()
                    start_tick = entry["alive_start_tick"]
                    console.command(f"demo_gototick {start_tick}", end_delay=1.0)
                    console.command("clear", end_delay=0.5)
                    console.command(f"spec_player {slot_index}")
                    console.command("demo_pause", end_delay=0.5)
                    console.command(f"demo_gototick {start_tick}", end_delay=1.0)
                    console.command("demo_resume")

                    capture.record_for(duration)
                    time.sleep(0.2)
                    console.command("demo_pause")

                    capture.claim_newest(target)
                    progress.mark(match_id, round_num, steamid, target.name)
                    counts["recorded"] += 1

                    time.sleep(1.0)
                    console.command("demo_resume", end_delay=0.1)

                except Exception as exc:
                    counts["failed"] += 1
                    log.error("round %s player %s failed: %s", round_num, steamid, exc)
                    if capture.recording:
                        try:
                            capture.stop()
                        except Exception:
                            pass

        console.command("disconnect")
        time.sleep(rcfg.between_demo_pause_seconds)
    finally:
        if game_demo.exists():
            game_demo.unlink(missing_ok=True)

    return counts


def run(
    cfg: Config,
    match_ids: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    layout = cfg.layout.ensure()

    candidates = [
        m for m in layout.match_ids_with_demos() if layout.metadata_file(m).exists()
    ]
    if match_ids:
        wanted = set(match_ids)
        candidates = [m for m in candidates if m in wanted]
    if not candidates:
        log.error("No (demo + metadata) pairs to record under %s", layout.root)
        return {"total": 0}

    if dry_run:
        total_clips = 0
        for match_id in candidates:
            meta = read_json(layout.metadata_file(match_id))
            alive = meta.get("player_alive_times") or {}
            clips = sum(len(v) for v in alive.values())
            seconds = sum(
                e["alive_duration_ticks"] / cfg.record.tickrate
                for v in alive.values()
                for e in v
            )
            total_clips += clips
            log.info(
                "%s  %s  %d rounds  %d clips  %.1f min of capture",
                match_id[:24], meta.get("map_name"), len(alive), clips, seconds / 60,
            )
        log.info("Total: %d matches, %d clips", len(candidates), total_clips)
        return {"total": len(candidates), "clips": total_clips, "dry_run": True}

    if sys.platform != "win32":
        raise RuntimeError(
            "Stage 04 records from a live CS2 client and only runs on Windows. "
            f"This host is {sys.platform!r}. Use `--dry-run` to preview the plan, "
            "and run the real capture on the gaming machine."
        )

    console = CS2Console(cfg)
    capture = Capture(cfg)
    progress = ProgressLog(layout.root / RECORD_LOG)

    totals = {"recorded": 0, "skipped": 0, "failed": 0}
    for i, match_id in enumerate(candidates, start=1):
        log.info("=== [%d/%d] recording %s ===", i, len(candidates), match_id)
        try:
            counts = record_match(cfg, match_id, console, capture, progress)
        except Exception as exc:
            log.error("Match %s aborted: %s", match_id, exc)
            continue
        for k, v in counts.items():
            totals[k] += v
        log.info("[%d/%d] %s -> %s", i, len(candidates), match_id[:24], counts)

    log.info("Recording totals: %s", totals)
    return totals
