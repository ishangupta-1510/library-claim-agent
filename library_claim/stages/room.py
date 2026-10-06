"""Room surface areas from metric points captured during the sweep.

Scale source: ARCore motion tracking (WebXR hit tests on the phone). Every
point is in metres in one world frame, so no separate measuring pass is
needed: the agent asks the user to aim at each floor corner and once at the
wall-ceiling line as they walk.

Method:
- Floor: the corner points form a polygon (in walk order). Area by the
  shoelace formula, so L-shaped and other non-rectangular rooms are exact.
- Length/width: sides of the minimum-area rectangle around the polygon.
- Height: ceiling point height minus the floor plane height.
- Walls: perimeter x height. This is gross wall area; doors and windows
  are not subtracted (stated assumption, see README).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

M2_TO_FT2 = 10.7639
RECTANGLE_FILL = 0.95  # polygon area / bounding-rectangle area above this = rectangular room


@dataclass
class RoomGeometry:
    length_m: float
    width_m: float
    height_m: float | None
    floor_area_m2: float
    wall_area_m2: float | None
    perimeter_m: float
    shape: str
    corners: int


def polygon_area(points_xz: np.ndarray) -> float:
    x, z = points_xz[:, 0], points_xz[:, 1]
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) / 2)


def perimeter(points_xz: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(points_xz, -1, axis=0) - points_xz, axis=1).sum())


def order_corners(points_xz: np.ndarray) -> np.ndarray:
    """Put corners in angular order around their centroid.

    Users rarely tap corners strictly in walk order; sorting by angle fixes a
    crossed polygon for convex rooms. Non-convex rooms (L-shapes) rely on the
    walk order instead, so this is only applied when the given order crosses.
    """
    centroid = points_xz.mean(axis=0)
    angles = np.arctan2(points_xz[:, 1] - centroid[1], points_xz[:, 0] - centroid[0])
    return points_xz[np.argsort(angles)]


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return orient(p1, p2, p3) * orient(p1, p2, p4) < 0 and orient(p3, p4, p1) * orient(p3, p4, p2) < 0


def self_intersects(points_xz: np.ndarray) -> bool:
    n = len(points_xz)
    edges = [(points_xz[i], points_xz[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) in (1, n - 1):
                continue
            if _segments_cross(*edges[i], *edges[j]):
                return True
    return False


CEILING_RANGE_M = (1.8, 6.0)  # plausible room heights; outside it the ceiling tap hit something else


def measure_room(floor_corners_m: list[tuple[float, float, float]], ceiling_y_m: float | None) -> RoomGeometry:
    """Floor corners as (x, y, z) world points in metres; y is up."""
    if len(floor_corners_m) < 3:
        raise ValueError("At least 3 floor corners are needed")
    pts = np.array(floor_corners_m, dtype=np.float64)
    floor_y = float(np.median(pts[:, 1]))
    xz = pts[:, [0, 2]]
    if self_intersects(xz):
        xz = order_corners(xz)

    area = polygon_area(xz)
    perim = perimeter(xz)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(xz.astype(np.float32))
    length, width = max(side_a, side_b), min(side_a, side_b)
    fill = area / (length * width) if length * width else 0
    shape = "rectangle" if fill >= RECTANGLE_FILL and len(xz) == 4 else f"polygon ({len(xz)} corners, fills {fill:.0%} of its bounding rectangle)"

    height = round(ceiling_y_m - floor_y, 3) if ceiling_y_m is not None else None
    if height is not None and not CEILING_RANGE_M[0] <= height <= CEILING_RANGE_M[1]:
        # A "ceiling" tap that hit a table or a shelf top: no height, so no wall area, rather than a wrong one.
        height = None
    walls = round(perim * height, 2) if height else None
    return RoomGeometry(
        length_m=round(float(length), 3), width_m=round(float(width), 3), height_m=height,
        floor_area_m2=round(area, 2), wall_area_m2=walls, perimeter_m=round(perim, 3),
        shape=shape, corners=len(xz),
    )


def to_ft2(m2: float | None) -> float | None:
    return round(m2 * M2_TO_FT2, 1) if m2 is not None else None
