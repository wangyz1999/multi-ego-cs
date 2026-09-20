"""Crop geometry and timer arithmetic.

The geometry test encodes a real bug: CS2's HUD scales with screen *height*
and is anchored to the horizontal *centre*, so scaling x by width puts the
crop in the wrong place on any non-16:9 capture. Measured on real 1024x720
clips, linear-width scaling read the timer at correlation 0.249; the
centre-anchored model read it at 0.989.
"""

import pytest

from multi_ego_cs.stages.hud_timer import TimerRead, scale_boxes, timer_to_game_sec

REF = (1280, 720)
BOXES = [(624, 5, 632, 18), (638, 5, 646, 18), (647, 5, 655, 18)]


def test_reference_geometry_is_identity():
    assert scale_boxes(BOXES, 1280, 720, *REF) == BOXES


def test_four_three_capture_keeps_glyph_size_and_recentres():
    # Same height -> same glyph size; only the centre moves (640 -> 512).
    scaled = scale_boxes(BOXES, 1024, 720, *REF)
    for (x1, _, x2, _), (rx1, _, rx2, _) in zip(scaled, BOXES):
        assert x2 - x1 == rx2 - rx1          # width preserved
        assert x1 == rx1 - (1280 - 1024) // 2  # shifted by the centre delta
    assert scaled[0][:2] == (496, 5)


def test_1080p_scales_glyphs_by_height():
    scaled = scale_boxes(BOXES, 1920, 1080, *REF)
    # 1.5x taller -> 1.5x wider glyphs, still centred.
    assert scaled[0][3] - scaled[0][1] == pytest.approx((18 - 5) * 1.5, abs=1)
    assert scaled[0][2] - scaled[0][0] == pytest.approx((632 - 624) * 1.5, abs=1)
    centre = (scaled[0][0] + scaled[-1][2]) / 2
    assert centre == pytest.approx(1920 / 2, abs=2)


def test_boxes_never_degenerate():
    for w, h in ((320, 180), (640, 360), (1024, 720), (2560, 1440)):
        for x1, y1, x2, y2 in scale_boxes(BOXES, w, h, *REF):
            assert x2 > x1 and y2 > y1


def test_timer_to_game_sec():
    # game_sec is 0 when the clock shows the full round time.
    assert timer_to_game_sec(TimerRead(1, 5, 5, 1.0), 115) == 0.0
    assert timer_to_game_sec(TimerRead(1, 5, 4, 1.0), 115) == 1.0
    assert timer_to_game_sec(TimerRead(0, 0, 0, 1.0), 115) == 115.0


def test_plausibility_rejects_impossible_readings():
    assert TimerRead(1, 5, 5, 1.0).is_plausible(115)
    assert not TimerRead(1, 9, 9, 1.0).is_plausible(115)   # tens > 5
    assert not TimerRead(9, 5, 5, 1.0).is_plausible(115)   # above round time


def test_timer_text_and_total():
    r = TimerRead(1, 3, 7, 0.9)
    assert r.text == "1:37"
    assert r.total_seconds == 97
