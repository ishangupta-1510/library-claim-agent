"""Scale is tested on synthetic scenes with exact ground truth.

The scene is a flat "shelf face" in centimetres containing a marker and a
spine-sized rectangle; it is rendered through a known camera perspective.
"""

import cv2
import numpy as np
import pytest

from library_claim.stages.scale import best_scale, rectify, render_marker

PX_PER_CM = 20  # resolution of the flat scene before the camera warp


def _scene(marker_cm: float, spine_cm: tuple[float, float], spine_at_cm: tuple[float, float]):
    """A 100 x 60 cm shelf face with a marker at (5, 5) cm and one spine."""
    face = np.full((60 * PX_PER_CM, 100 * PX_PER_CM, 3), 200, np.uint8)
    marker = render_marker(0, 600)
    inner = 600  # render_marker pads; the inner black square is 600 px
    pad = (marker.shape[0] - inner) // 2
    side = int(marker_cm * PX_PER_CM)
    scaled = cv2.resize(marker, None, fx=side / inner, fy=side / inner, interpolation=cv2.INTER_NEAREST)
    off = int(round(pad * side / inner))
    x, y = 5 * PX_PER_CM - off, 5 * PX_PER_CM - off
    face[y : y + scaled.shape[0], x : x + scaled.shape[1]] = scaled[..., None]
    sx, sy = (int(v * PX_PER_CM) for v in spine_at_cm)
    face[sy : sy + int(spine_cm[1] * PX_PER_CM), sx : sx + int(spine_cm[0] * PX_PER_CM)] = (30, 60, 140)
    return face


def _photograph(face: np.ndarray, tilt: float):
    """Warp the flat face as a camera would see it from an angle."""
    h, w = face.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = tilt * w
    dst = np.float32([[d, d * 0.3], [w - d * 0.2, 0], [w, h], [0, h - d * 0.4]])
    homography = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(face, homography, (w, h), borderValue=(90, 90, 90)), homography


@pytest.mark.parametrize("tilt", [0.0, 0.08, 0.15])
def test_spine_measured_on_marker_plane(tilt):
    spine = (3.2, 23.4)  # a typical paperback spine: thickness x height
    face = _scene(marker_cm=15.0, spine_cm=spine, spine_at_cm=(40, 20))
    photo, homography = _photograph(face, tilt)

    scale = best_scale(photo, marker_size_cm=15.0)
    assert scale is not None and scale.marker_id == 0

    # Detect the spine where the pipeline does: on the head-on rectified view.
    view = rectify(photo, scale, px_per_cm=12)
    spine_mask = cv2.inRange(view.image, (20, 50, 125), (40, 70, 155))
    ys, xs = np.nonzero(spine_mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    width, height = view.box_cm(box)
    assert width == pytest.approx(spine[0], rel=0.05)
    assert height == pytest.approx(spine[1], rel=0.03)


def test_raw_photo_boxes_overstate_tilted_spines():
    """Documents why rectification exists: the raw-photo box is badly wrong."""
    spine = (3.2, 23.4)
    face = _scene(marker_cm=15.0, spine_cm=spine, spine_at_cm=(40, 20))
    photo, homography = _photograph(face, 0.15)
    scale = best_scale(photo, marker_size_cm=15.0)
    corners = np.float32([[40, 20], [43.2, 20], [43.2, 43.4], [40, 43.4]]) * PX_PER_CM
    in_photo = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    x0, y0 = in_photo.min(axis=0)
    x1, y1 = in_photo.max(axis=0)
    from library_claim.stages.scale import measure_box

    width, _ = measure_box(scale, (x0, y0, x1, y1))
    assert width > spine[0] * 1.5


def test_no_marker_means_no_scale():
    blank = np.full((480, 640, 3), 128, np.uint8)
    assert best_scale(blank, marker_size_cm=15.0) is None
