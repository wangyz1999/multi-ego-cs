"""Canonical on-disk layout for a multi-ego-cs collection.

Every stage addresses data through this module so that the layout is defined
exactly once. The layout is Hive-style partitioned, which lets polars/pandas/
pyarrow and the Hugging Face `datasets` loader all discover it without a custom
reader:

    <root>/
      matches.jsonl                         stage 01 - discovery index
      demo/<match_id>.dem                   stage 02
      metadata/<match_id>.json              stage 03
      video/match=<id>/round=<n>/<steam>.mp4        stage 04
      state_action/match=<id>/round=<n>/<steam>.parquet   stage 05
      align/match=<id>/round=<n>/offsets.json       stage 06
      release/                              stage 07 - upload-ready tree
      logs/                                 per-stage run logs

`match_id` is the FACEIT match id exactly as it appears in the demo filename,
e.g. ``1-0076bc6b-4ce9-45fa-8e0b-35fd140ddd60-1-1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Subtree names, exported so stages never hard-code strings.
DEMO_DIR = "demo"
METADATA_DIR = "metadata"
VIDEO_DIR = "video"
STATE_ACTION_DIR = "state_action"
ALIGN_DIR = "align"
RELEASE_DIR = "release"
LOGS_DIR = "logs"

MATCHES_INDEX = "matches.jsonl"

VIDEO_SUFFIX = ".mp4"
TRAJECTORY_SUFFIX = ".parquet"
OFFSETS_FILENAME = "offsets.json"


@dataclass(frozen=True)
class Layout:
    """Resolves every path in a collection from a single data root."""

    root: Path

    def __post_init__(self) -> None:
        # Expand both ~ and $VARS so a committed config can say
        # /scratch1/$USER/... and still work for every user of the repo.
        expanded = os.path.expandvars(str(self.root))
        object.__setattr__(self, "root", Path(expanded).expanduser().resolve())

    # -- top-level subtrees -------------------------------------------------
    @property
    def matches_index(self) -> Path:
        return self.root / MATCHES_INDEX

    @property
    def demo(self) -> Path:
        return self.root / DEMO_DIR

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_DIR

    @property
    def video(self) -> Path:
        return self.root / VIDEO_DIR

    @property
    def state_action(self) -> Path:
        return self.root / STATE_ACTION_DIR

    @property
    def align(self) -> Path:
        return self.root / ALIGN_DIR

    @property
    def release(self) -> Path:
        return self.root / RELEASE_DIR

    @property
    def logs(self) -> Path:
        return self.root / LOGS_DIR

    # -- per-match paths ----------------------------------------------------
    def demo_file(self, match_id: str) -> Path:
        return self.demo / f"{match_id}.dem"

    def demo_archive(self, match_id: str) -> Path:
        """Compressed form as served by the FACEIT CDN."""
        return self.demo / f"{match_id}.dem.zst"

    def metadata_file(self, match_id: str) -> Path:
        return self.metadata / f"{match_id}.json"

    # -- per-round / per-player paths ---------------------------------------
    def video_round(self, match_id: str, round_num: int | str) -> Path:
        return self.video / f"match={match_id}" / f"round={round_num}"

    def video_file(self, match_id: str, round_num: int | str, steamid: str) -> Path:
        return self.video_round(match_id, round_num) / f"{steamid}{VIDEO_SUFFIX}"

    def state_action_round(self, match_id: str, round_num: int | str) -> Path:
        return self.state_action / f"match={match_id}" / f"round={round_num}"

    def state_action_file(self, match_id: str, round_num: int | str, steamid: str) -> Path:
        return self.state_action_round(match_id, round_num) / f"{steamid}{TRAJECTORY_SUFFIX}"

    def align_round(self, match_id: str, round_num: int | str) -> Path:
        return self.align / f"match={match_id}" / f"round={round_num}"

    def offsets_file(self, match_id: str, round_num: int | str) -> Path:
        return self.align_round(match_id, round_num) / OFFSETS_FILENAME

    # -- helpers ------------------------------------------------------------
    def ensure(self) -> Layout:
        """Create every subtree. Safe to call repeatedly."""
        for p in (
            self.demo,
            self.metadata,
            self.video,
            self.state_action,
            self.align,
            self.release,
            self.logs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def match_ids_with_demos(self) -> list[str]:
        if not self.demo.exists():
            return []
        return sorted(p.stem for p in self.demo.glob(f"*{'.dem'}") if p.is_file())

    def match_ids_with_video(self) -> list[str]:
        if not self.video.exists():
            return []
        return sorted(
            p.name.removeprefix("match=") for p in self.video.iterdir() if p.name.startswith("match=")
        )
