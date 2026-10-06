"""Metric scale from a reference marker.

Method: an ArUco marker of known physical size (printed, or shown on a laptop
screen and measured once with a tape) sits on the front plane of a shelving
unit. Its four corners give a homography from image pixels to centimetres
on that plane, which corrects for camera angle as well as distance. Spines on
the same shelf face lie (approximately) in that plane, so their corners map
straight to centimetres.

Limits, stated rather than hidden:
- Objects recessed behind the marker plane come out slightly small, by the
  ratio of depths (about 3% for a 5 cm recess at 1.5 m).
- A frame without a marker gets its scale from the tracking stage, which
  chains it to a frame that has one. With no chain, the dimensions are None.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Marker:
    id: int
    corners: np.ndarray  # 4x2, clockwise from top-left, image pixels


@dataclass
class PlaneScale:
    """Image-pixel → plane-centimetre homography, with its provenance."""

    homography: np.ndarray  # 3x3
    marker_id: int
    marker_size_cm: float
    marker_px: float  # mean side length in pixels, for quality checks
    reprojection_error_cm: float

    @property
    def cm_per_px(self) -> float:
        return self.marker_size_cm / self.marker_px


def _detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Screens and glossy prints glare; subpixel refinement keeps the corners stable.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)


def detect_markers(image: np.ndarray) -> list[Marker]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return []
    return [Marker(id=int(i), corners=c.reshape(4, 2).astype(np.float64)) for c, i in zip(corners, ids.flatten())]


def plane_scale(marker: Marker, marker_size_cm: float) -> PlaneScale:
    """Homography mapping image pixels onto the marker's plane in centimetres."""
    s = marker_size_cm
    target = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float64)
    homography, _ = cv2.findHomography(marker.corners, target)
    mapped = cv2.perspectiveTransform(marker.corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = float(np.abs(mapped - target).max())
    sides = np.linalg.norm(np.roll(marker.corners, -1, axis=0) - marker.corners, axis=1)
    return PlaneScale(homography, marker.id, s, float(sides.mean()), error)


def to_plane_cm(scale: PlaneScale, points_px: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, scale.homography).reshape(-1, 2)


def measure_box(scale: PlaneScale, box_px: tuple[float, float, float, float]) -> tuple[float, float]:
    """Width and height in cm of an axis-aligned image box (x0, y0, x1, y1) on the plane.

    Uses the mapped edge lengths (mean of opposite edges), so mild perspective
    across the box averages out.
    """
    x0, y0, x1, y1 = box_px
    quad = to_plane_cm(scale, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    top, right, bottom, left = (np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4))
    return float((top + bottom) / 2), float((left + right) / 2)


@dataclass
class Rectified:
    """A frame re-projected head-on onto the shelf plane at a fixed resolution."""

    image: np.ndarray
    px_per_cm: float
    origin_cm: tuple[float, float]  # plane coordinate of the image's top-left pixel
    image_to_rectified: np.ndarray  # 3x3 homography, original pixels -> rectified pixels

    def box_cm(self, box_px: tuple[float, float, float, float]) -> tuple[float, float]:
        """Width and height in cm of an upright box drawn on the rectified image."""
        x0, y0, x1, y1 = box_px
        return (x1 - x0) / self.px_per_cm, (y1 - y0) / self.px_per_cm


def rectify(
    image: np.ndarray,
    scale: PlaneScale,
    px_per_cm: float = 12.0,
    max_extent_cm: float = 400.0,
) -> Rectified:
    """Warp the frame onto the marker plane so spines become upright rectangles.

    Measuring axis-aligned boxes in the raw photo overstates thickness badly
    when the phone is not square to the shelf (a 3.2 cm spine read as 7.3 cm in
    tests). In the rectified view, a box's size divided by px_per_cm is the
    real size on the plane.
    """
    h, w = image.shape[:2]
    corners = to_plane_cm(scale, np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    # Points near the horizon map towards infinity; clamp the canvas to a sane extent.
    corners = np.clip(corners, -max_extent_cm, max_extent_cm)
    (min_x, min_y), (max_x, max_y) = corners.min(axis=0), corners.max(axis=0)
    to_pixels = np.array([[px_per_cm, 0, -min_x * px_per_cm], [0, px_per_cm, -min_y * px_per_cm], [0, 0, 1]])
    homography = to_pixels @ scale.homography
    size = (int(np.ceil((max_x - min_x) * px_per_cm)), int(np.ceil((max_y - min_y) * px_per_cm)))
    warped = cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return Rectified(warped, px_per_cm, (float(min_x), float(min_y)), homography)


def best_scale(image: np.ndarray, marker_size_cm: float, min_marker_px: float = 40) -> PlaneScale | None:
    """The most reliable marker in the frame: the largest one above a size floor."""
    markers = detect_markers(image)
    scales = [plane_scale(m, marker_size_cm) for m in markers]
    scales = [s for s in scales if s.marker_px >= min_marker_px]
    return max(scales, key=lambda s: s.marker_px, default=None)


def render_marker(marker_id: int = 0, side_px: int = 800) -> np.ndarray:
    """The marker image, with a white quiet zone so detection works on screens."""
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, side_px)
    pad = side_px // 6
    return cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
