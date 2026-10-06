"""A camera panning across a textured shelf face: scale must survive frames with no marker."""

import cv2
import numpy as np
import pytest

from library_claim.stages.quality import assess
from library_claim.stages.scale import best_scale, render_marker, to_plane_cm
from library_claim.stages.tracking import register_sequence

PX_PER_CM = 10
MARKER_CM = 15.0


def _shelf_face(seed=7):
    """200 x 80 cm of book-like vertical stripes with a marker at the far left."""
    rng = np.random.default_rng(seed)
    face = np.full((80 * PX_PER_CM, 200 * PX_PER_CM, 3), 220, np.uint8)
    x = 25 * PX_PER_CM
    while x < face.shape[1] - 10:
        w = int(rng.integers(15, 50))
        color = tuple(int(c) for c in rng.integers(20, 230, 3))
        face[60:-60, x : x + w] = color
        # Spine "text": small random blocks give ORB something to match.
        for _ in range(6):
            ty = int(rng.integers(80, face.shape[0] - 100))
            face[ty : ty + 12, x + 3 : x + w - 3] = 255 - np.array(color)
        x += w + 2
    marker = render_marker(0, 300)
    inner = 300
    side = int(MARKER_CM * PX_PER_CM)
    scaled = cv2.resize(marker, None, fx=side / inner, fy=side / inner, interpolation=cv2.INTER_NEAREST)
    off = int(round((marker.shape[0] - inner) / 2 * side / inner))
    face[200 - off : 200 - off + scaled.shape[0], 40 - off : 40 - off + scaled.shape[1]] = scaled[..., None]
    return face


def _pan(face, n=8, width=640, height=480):
    """Crops sliding right with a slight zoom, as a handheld pan would look."""
    frames, truths = [], []
    for k in range(n):
        x0 = k * 160
        src = np.float32([[x0, 0], [x0 + 700, 0], [x0 + 700, 800], [x0, 800]])
        dst = np.float32([[0, 0], [width, 10], [width, height], [0, height - 10]])
        homography = cv2.getPerspectiveTransform(src, dst)
        frames.append((f"f{k}", cv2.warpPerspective(face, homography, (width, height))))
        truths.append(np.linalg.inv(homography))  # frame px -> face px
    return frames, truths


def test_scale_propagates_to_frames_without_marker():
    face = _shelf_face()
    frames, truths = _pan(face)
    anchor = best_scale(frames[0][1], MARKER_CM)
    assert anchor is not None
    assert best_scale(frames[5][1], MARKER_CM) is None  # marker long out of view

    track = register_sequence(frames, {"f0": anchor}, "unit-A", MARKER_CM)
    assert len(track.plane_from_frame) == len(frames)

    # A point in frame 5 must land on the same face position (in cm) as ground truth.
    point = np.array([[320.0, 240.0]])
    truth_px = cv2.perspectiveTransform(point.reshape(-1, 1, 2), truths[5]).reshape(2)
    got_cm = to_plane_cm(track.scale_for("f5"), point)[0]
    marker_origin_cm = np.array([4.0, 20.0])  # marker top-left sits at (4, 20) cm on the face
    expected_cm = truth_px / PX_PER_CM - marker_origin_cm
    assert np.linalg.norm(got_cm - expected_cm) < 1.5  # cm, after five chained links


def test_unrelated_frame_breaks_the_chain():
    face = _shelf_face()
    frames, _ = _pan(face, n=3)
    noise = np.random.default_rng(1).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    frames.append(("elsewhere", noise))
    anchor = best_scale(frames[0][1], MARKER_CM)
    track = register_sequence(frames, {"f0": anchor}, "unit-A", MARKER_CM)
    assert "elsewhere" not in track.plane_from_frame


def test_quality_flags_blur_and_darkness():
    face = _shelf_face()
    sharp = face[:480, :640]
    assert assess(sharp).usable
    blurred = cv2.GaussianBlur(sharp, (31, 31), 12)
    assert assess(blurred).blurry
    assert assess((sharp * 0.1).astype(np.uint8)).dark


@pytest.mark.parametrize("n", [1])
def test_anchor_alone_registers(n):
    face = _shelf_face()
    frames, _ = _pan(face, n=n)
    anchor = best_scale(frames[0][1], MARKER_CM)
    track = register_sequence(frames, {"f0": anchor}, "unit-A", MARKER_CM)
    assert list(track.plane_from_frame) == ["f0"]
