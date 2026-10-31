"""Minimal checks for court masking and ball gating. Run: python test_vision.py"""
import numpy as np

from vision import _pick_ball, court_mask


def test_court():
    frame = np.full((720, 1280, 3), (60, 60, 60), np.uint8)
    frame[250:700, 60:1220] = (254, 209, 195)  # bright blue floor
    mask, poly, bbox = court_mask(frame)
    assert poly and bbox is not None
    assert mask[500, 640] == 255
    assert mask[100, 640] == 0


def test_ball_gate():
    state = {"pos": None, "vel": np.zeros(2), "miss": 0}
    court = (0, 0, 720, 1280)
    first = [400, 600, 420, 620]
    assert _pick_ball([(0.8, first)], court, 720, 1280, state) == first
    near = [430, 600, 450, 620]
    assert _pick_ball([(0.5, near)], court, 720, 1280, state) == near
    far = [80, 100, 100, 120]
    assert _pick_ball([(0.9, far)], court, 720, 1280, state) is None


if __name__ == "__main__":
    test_court()
    test_ball_gate()
    print("vision checks passed")
