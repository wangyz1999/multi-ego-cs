"""Typed configuration, loaded from YAML with environment-variable overrides.

Precedence (highest wins):

    1. explicit CLI flags
    2. environment variables  (MECS_*)
    3. the YAML config file
    4. dataclass defaults

Only two settings are *secret* and therefore env-only by design, so that a
config file can be committed without leaking anything:

    MECS_FACEIT_API_KEY   FACEIT Data API key (stage 01)
    HF_TOKEN              Hugging Face write token (stage 08)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .paths import Layout

# Competitive CS2 round clock, in seconds, at the moment the round goes live.
# MR12 with default `mp_roundtime` is 1:55.  Stage 06 needs this to convert a
# HUD timer reading into game time.
DEFAULT_ROUND_CLOCK_SECONDS = 115

# CS2 records demos at 64 ticks/second.
DEFAULT_TICKRATE = 64


@dataclass
class DiscoverConfig:
    """Stage 01 - choose which matches to collect."""

    target_matches: int = 100
    region: str = "EU"
    # How many top-ranked players to walk. Discovery fans out over players
    # because FACEIT exposes match history per player, not globally.
    num_players: int = 200
    matches_per_player: int = 20
    min_elo: int = 2000
    # Restrict to these maps; empty list means "accept any map".
    maps: list[str] = field(default_factory=list)
    # Only keep matches whose in-game voice comms are predominantly this
    # language (requires the voice transcription extra); empty disables it.
    language: str = ""
    # Skip matches shorter than this many rounds (forfeits, early leaves).
    min_rounds: int = 16
    request_timeout: float = 30.0
    # FACEIT rate-limits the Data API; stay well under it.
    requests_per_minute: int = 250


@dataclass
class DownloadConfig:
    """Stage 02 - fetch .dem files."""

    max_total_gb: float = 2000.0
    concurrency: int = 4
    retries: int = 3
    retry_backoff_seconds: float = 5.0
    # FACEIT serves demos as zstd archives; decompress after download.
    decompress: bool = True
    keep_archive: bool = False
    # Browser fallback is only needed when the CDN URL is not in the match
    # payload (older matches). Requires the `download` extra.
    allow_browser_fallback: bool = True
    browser_headless: bool = True


@dataclass
class RecordConfig:
    """Stage 04 - drive CS2 to replay demos and capture per-player video.

    Windows-only. Every path here is machine-specific and must be set by the
    operator; the defaults are the common Steam install locations.
    """

    cs2_csgo_dir: str = r"C:\Program Files (x86)\Steam\steamapps\common\Counter-Strike Global Offensive\game\csgo"
    cs2_launch_target: str = "steam://rungameid/730"
    # Where the external capture tool drops finished clips. The recorder moves
    # and renames them from here.
    capture_output_dir: str = ""
    # Capture hotkey (start/stop toggle) as understood by pyautogui.
    capture_hotkey: tuple[str, ...] = ("alt", "f9")
    demo_load_wait_seconds: float = 60.0
    between_demo_pause_seconds: float = 10.0
    command_pause_seconds: float = 0.1
    game_boot_wait_seconds: float = 40.0
    # Spectator slot of the first player in the metadata player list. CS2's
    # `spec_player N` is 1-based over a list that begins after the observer
    # slots; 4 matches the stock competitive layout.
    spec_player_offset: int = 4
    tickrate: int = DEFAULT_TICKRATE


@dataclass
class ActionsConfig:
    """Stage 05 - tick-level state/action extraction."""

    tickrate: int = DEFAULT_TICKRATE
    workers: int = 0  # 0 -> auto (85% of visible CPUs)
    cpu_fraction: float = 0.85
    compression: str = "zstd"
    compression_level: int = 3
    # Emit the raw `buttons` bitfield alongside the decoded k_* columns.
    keep_raw_buttons: bool = True
    rounds: list[int] = field(default_factory=list)  # empty -> all rounds


@dataclass
class AlignConfig:
    """Stage 06 - HUD-timer pattern matching to align video with ticks."""

    # Digit crop boxes (x1, y1, x2, y2) in the *reference* frame geometry.
    # Scaled automatically for other resolutions.
    reference_width: int = 1280
    reference_height: int = 720
    digit_boxes: list[tuple[int, int, int, int]] = field(
        default_factory=lambda: [
            (624, 5, 632, 18),  # minutes
            (638, 5, 646, 18),  # tens of seconds
            (647, 5, 655, 18),  # units of seconds
        ]
    )
    round_clock_seconds: int = DEFAULT_ROUND_CLOCK_SECONDS
    # Give up on a clip after this many frames without a clean timer tick.
    max_scan_seconds: float = 12.0
    # Minimum normalised-cross-correlation score to trust a digit read.
    min_match_score: float = 0.35
    # Reject a per-player offset that deviates from the round median by more
    # than this many seconds.
    max_offset_deviation_seconds: float = 2.0
    workers: int = 0
    templates_dir: str = ""  # empty -> packaged assets


@dataclass
class PackageConfig:
    """Stage 07 - build the upload-ready release tree."""

    # Video variants to emit. Each entry becomes video_<name>/ in the release.
    # An empty dict means "publish the source recordings unchanged".
    variants: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "native": {},  # copy/link source recordings as-is
        }
    )
    splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    split_seed: int = 1999
    # Split at match granularity so no match straddles two splits.
    split_by: str = "match"
    ffmpeg: str = "ffmpeg"
    workers: int = 0
    # How to place untranscoded files into the release tree:
    #   auto     hard-link, else symlink, else copy   (cheapest; default)
    #   symlink  always a pointer
    #   copy     a self-contained tree you can tar and move
    link_mode: str = "auto"
    # Deprecated alias kept for older configs; `link_mode` wins when set.
    use_hardlinks: bool = True
    # Include matches that have video but no demo (and therefore no action
    # data). The dataset is explicitly multi-modal-with-gaps, so this defaults
    # on and coverage is reported per clip.
    include_video_only_matches: bool = True


@dataclass
class PublishConfig:
    """Stage 08 - upload to the Hugging Face Hub."""

    repo_id: str = ""
    repo_type: str = "dataset"
    private: bool = False
    revision: str = "main"
    # Upload in chunks so a dropped connection costs one chunk, not the run.
    files_per_commit: int = 400
    commit_message: str = "Add multi-ego-cs release"
    # Directories under release/ to upload, in this order.
    include: list[str] = field(
        default_factory=lambda: [
            "README.md",
            "match_round_partitioned.csv",
            "manifest",
            "metadata",
            "align",
            "state_action",
            "video",
            "demo",
        ]
    )
    max_workers: int = 8
    enable_hf_transfer: bool = True


@dataclass
class Config:
    """Root configuration object."""

    data_root: str = "./data"
    discover: DiscoverConfig = field(default_factory=DiscoverConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    package: PackageConfig = field(default_factory=PackageConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    # -- secrets: never serialised, read from the environment only ----------
    @property
    def faceit_api_key(self) -> str | None:
        return os.environ.get("MECS_FACEIT_API_KEY") or os.environ.get("FACEIT_API_KEY")

    @property
    def hf_token(self) -> str | None:
        """Hugging Face write token.

        Environment first, then whatever `hf auth login` stored. Falling back to
        the credential store means an interactive login is enough and the token
        never has to be pasted into a job script or a config file - which is
        also why this is a lookup and not a setting.
        """
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env:
            return env
        try:
            from huggingface_hub import get_token

            return get_token()
        except Exception:
            return None

    @property
    def layout(self) -> Layout:
        return Layout(Path(self.data_root))

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _section(dc_type: type, value: Any) -> Any:
    """Build one config section from a plain dict, rejecting unknown keys.

    Rejecting rather than ignoring is deliberate: a typo like `taget_matches`
    would otherwise silently leave the default in place, and the run would look
    configured while doing something else entirely.
    """
    if not isinstance(value, dict):
        if value is None:
            return dc_type()
        raise ValueError(f"Expected a mapping for {dc_type.__name__}, got {type(value).__name__}")
    known = {f.name for f in fields(dc_type)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in the '{dc_type.__name__}' section. "
            f"Valid keys: {sorted(known)}"
        )
    return dc_type(**value)


_ENV_PREFIX = "MECS_"

# Environment overrides for the handful of settings worth setting per-run.
_ENV_MAP: dict[str, tuple[str, str, type]] = {
    f"{_ENV_PREFIX}DATA_ROOT": ("", "data_root", str),
    f"{_ENV_PREFIX}TARGET_MATCHES": ("discover", "target_matches", int),
    f"{_ENV_PREFIX}REGION": ("discover", "region", str),
    f"{_ENV_PREFIX}REPO_ID": ("publish", "repo_id", str),
    f"{_ENV_PREFIX}WORKERS": ("actions", "workers", int),
}


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from `path` (if given), then apply env overrides."""
    raw: dict[str, Any] = {}
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        raw = yaml.safe_load(p.read_text()) or {}

    cfg = Config(
        data_root=raw.get("data_root", Config.data_root),
        discover=_section(DiscoverConfig, raw.get("discover", {})),
        download=_section(DownloadConfig, raw.get("download", {})),
        record=_section(RecordConfig, raw.get("record", {})),
        actions=_section(ActionsConfig, raw.get("actions", {})),
        align=_section(AlignConfig, raw.get("align", {})),
        package=_section(PackageConfig, raw.get("package", {})),
        publish=_section(PublishConfig, raw.get("publish", {})),
    )

    for env_name, (section, attr, caster) in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is None or val == "":
            continue
        target = cfg if section == "" else getattr(cfg, section)
        setattr(target, attr, caster(val))

    return cfg


def resolve_workers(requested: int, fraction: float = 0.85) -> int:
    """Translate a `workers: 0` setting into a concrete count.

    Respects SLURM's allocation when present, so a job that asked for 16 CPUs
    does not try to spawn one worker per core on a 128-core node.
    """
    if requested and requested > 0:
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        available = int(slurm)
    else:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, int(available * fraction))
