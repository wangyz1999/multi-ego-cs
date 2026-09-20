"""Read the CS2 round timer off a video frame by template matching.

This is the measurement device behind stage 06. The recordings are screen
captures with no embedded tick information, so the only common clock visible in
both the video and the demo is the round timer the HUD draws at the top of the
screen. Reading it turns "frame 37" into "game second 4".

Method
------
The timer is rendered in a fixed font at a fixed screen position, so a full OCR
engine is unnecessary and unwelcome (slow, and it hallucinates on 8x13 crops).
Instead three small regions - minutes, tens of seconds, units of seconds - are
cropped and compared against ten reference glyphs with normalised
cross-correlation (``cv2.TM_CCOEFF_NORMED``). Normalised correlation is
brightness- and contrast-invariant, which is what makes it survive h264
compression artefacts and the wildly varying game background behind the HUD.

Each read returns the best-matching digit *and* its correlation score, so a
caller can reject a frame where the timer is obscured (kill feed overlay, a
flashbang whiting out the HUD) rather than silently trusting a bad digit.

Geometry
--------
Crop boxes are expressed in a reference frame size (1280x720 by default) and
scaled to whatever the clip actually is, so 1080p or 1440p captures work
without new constants. Templates are resized to the scaled crop the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

TEMPLATE_FILENAMES = {d: f"{d}.png" for d in range(10)}


def packaged_templates_dir() -> Path:
    """Digit glyphs shipped inside the wheel."""
    return Path(__file__).resolve().parent.parent / "assets" / "digit_templates"


def _repo_templates_dir() -> Path:
    """Glyphs as laid out in a source checkout."""
    return Path(__file__).resolve().parents[3] / "assets" / "digit_templates"


def resolve_templates_dir(override: str | Path | None = None) -> Path:
    for candidate in (override, packaged_templates_dir(), _repo_templates_dir()):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError(
        "Could not locate digit templates. Pass align.templates_dir explicitly."
    )


@lru_cache(maxsize=4)
def load_templates(templates_dir: str) -> dict[int, np.ndarray]:
    """Load the ten grayscale digit glyphs as float32 arrays."""
    import cv2

    directory = Path(templates_dir)
    out: dict[int, np.ndarray] = {}
    for digit, name in TEMPLATE_FILENAMES.items():
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"Missing digit template: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unreadable digit template: {path}")
        out[digit] = img.astype(np.float32)
    return out


@dataclass(frozen=True)
class DigitRead:
    digit: int
    score: float


@dataclass(frozen=True)
class TimerRead:
    """One timer observation: ``M:SS`` plus the weakest digit's score."""

    minutes: int
    tens: int
    units: int
    score: float

    @property
    def total_seconds(self) -> int:
        return self.minutes * 60 + self.tens * 10 + self.units

    @property
    def text(self) -> str:
        return f"{self.minutes}:{self.tens}{self.units}"

    def is_plausible(self, round_clock_seconds: int) -> bool:
        """Reject impossible readings before they poison an offset.

        The clock counts *down* from the round time, so a reading above it is
        a misread. ``tens`` above 5 is likewise impossible in base-60.
        """
        return self.tens <= 5 and 0 <= self.total_seconds <= round_clock_seconds


def scale_boxes(
    boxes: list[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    reference_width: int,
    reference_height: int,
) -> list[tuple[int, int, int, int]]:
    """Map reference-geometry crop boxes onto an actual frame size.

    CS2 lays out its HUD the way most game engines do: **element size scales
    with screen height, and the top bar is anchored to the horizontal centre.**
    It does *not* stretch with width. So a 1024x720 (4:3) capture draws the
    timer at exactly the same pixel size as a 1280x720 (16:9) one, just centred
    on x=512 instead of x=640.

    Scaling x linearly by width - the obvious implementation - is therefore
    wrong for any aspect ratio other than the reference. Measured on real
    1024x720 captures: linear-width scaling reads the timer with correlation
    0.249 (garbage), while this centre-anchored model reads it at 0.989. At
    1280x720 the two are identical, so the reference case is unaffected.
    """
    if frame_width == reference_width and frame_height == reference_height:
        return list(boxes)

    scale = frame_height / reference_height
    ref_centre_x = reference_width / 2
    centre_x = frame_width / 2

    scaled: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        nx1 = int(round(centre_x + (x1 - ref_centre_x) * scale))
        nx2 = int(round(centre_x + (x2 - ref_centre_x) * scale))
        ny1 = int(round(y1 * scale))
        ny2 = int(round(y2 * scale))
        # Guarantee a non-degenerate crop even at small scale factors.
        scaled.append((nx1, ny1, max(nx2, nx1 + 1), max(ny2, ny1 + 1)))
    return scaled


def match_digit(crop_gray: np.ndarray, templates: dict[int, np.ndarray]) -> DigitRead:
    """Best-matching digit for one crop, with its correlation score."""
    import cv2

    crop = crop_gray.astype(np.float32)
    best = DigitRead(digit=-1, score=-np.inf)
    for digit, template in templates.items():
        if template.shape != crop.shape:
            template = cv2.resize(
                template, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        # Equal-sized inputs make this a single correlation coefficient.
        result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        score = float(result[0, 0])
        if score > best.score:
            best = DigitRead(digit=digit, score=score)
    return best


def read_timer(
    frame: np.ndarray,
    templates: dict[int, np.ndarray],
    boxes: list[tuple[int, int, int, int]],
) -> TimerRead:
    """Read ``M:SS`` from one BGR frame."""
    import cv2

    reads: list[DigitRead] = []
    for x1, y1, x2, y2 in boxes:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return TimerRead(-1, -1, -1, -np.inf)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        reads.append(match_digit(gray, templates))

    # The weakest digit bounds confidence in the reading as a whole.
    return TimerRead(
        minutes=reads[0].digit,
        tens=reads[1].digit,
        units=reads[2].digit,
        score=min(r.score for r in reads),
    )


def timer_to_game_sec(read: TimerRead, round_clock_seconds: int) -> float:
    """Convert a timer reading to game seconds since the round went live.

    ``game_sec`` is zero at ``freeze_end`` - the instant the round goes live -
    which is exactly when the HUD clock shows the full round time. So game time
    is the amount already counted off the clock.
    """
    return float(round_clock_seconds - read.total_seconds)
