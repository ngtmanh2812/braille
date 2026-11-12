import cv2
import numpy as np

from dot_detection import detect_dots, draw_detections


def test_detects_centers_and_ignores_tiny_noise():
    binary = np.zeros((100, 140), np.uint8)
    cv2.circle(binary, (30, 35), 7, 255, -1)
    cv2.circle(binary, (90, 60), 7, 255, -1)
    binary[5, 5] = 255
    dots = detect_dots(binary)
    assert len(dots) == 2
    assert dots[0].center == (30.0, 35.0)
    assert dots[1].center == (90.0, 60.0)
    assert draw_detections(binary, dots).shape == (100, 140, 3)


def test_empty_mask_has_no_dots():
    assert detect_dots(np.zeros((20, 20), np.uint8)) == []
