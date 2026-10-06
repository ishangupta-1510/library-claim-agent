"""Chain frames onto a shelf plane so every frame shares one coordinate system.

A shelving unit's face is (close to) a plane, and two views of a plane are
related exactly by a homography. So once one frame of a unit contains the
marker, every overlapping frame can be mapped onto the same centimetre grid:

    plane_from_frame_k = plane_from_frame_j @ frame_j_from_frame_k

That gives scale to frames without a marker, and puts the same spine seen in
several frames at the same plane position, which is how duplicates merge.

A chain is only trusted while each link has enough RANSAC inliers. A weak
link breaks the chain: those frames stay unscaled rather than getting a
drifting, wrong scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scale import PlaneScale

MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.35
MAX_CHAIN = 30  # links from the marker frame; beyond this drift is not trusted


@dataclass
class Link:
    homography: np.ndarray | None  # maps frame-k pixels into frame-j pixels
    inliers: int
    inlier_ratio: float

    @property
    def ok(self) -> bool:
        return self.homography is not None and self.inliers >= MIN_INLIERS and self.inlier_ratio >= MIN_INLIER_RATIO


def _features(gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    return orb.detectAndCompute(gray, None)


def link(frame_j: np.ndarray, frame_k: np.ndarray) -> Link:
    """Homography taking frame_k pixels into frame_j pixels (both BGR or gray)."""
    gj = cv2.cvtColor(frame_j, cv2.COLOR_BGR2GRAY) if frame_j.ndim == 3 else frame_j
    gk = cv2.cvtColor(frame_k, cv2.COLOR_BGR2GRAY) if frame_k.ndim == 3 else frame_k
    kj, dj = _features(gj)
    kk, dk = _features(gk)
    if dj is None or dk is None or len(kj) < 8 or len(kk) < 8:
        return Link(None, 0, 0.0)
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dk, dj, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return Link(None, len(good), 0.0)
    src = np.float32([kk[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kj[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return Link(homography, inliers, inliers / len(good))


@dataclass
class PlaneTrack:
    """Frames registered to one shelving unit's plane."""

    plane_id: str
    marker_size_cm: float
    # frame id -> 3x3 homography from that frame's pixels to plane cm
    plane_from_frame: dict[str, np.ndarray] = field(default_factory=dict)
    hops: dict[str, int] = field(default_factory=dict)

    def scale_for(self, frame_id: str) -> PlaneScale | None:
        homography = self.plane_from_frame.get(frame_id)
        if homography is None:
            return None
        return PlaneScale(homography, marker_id=-1, marker_size_cm=self.marker_size_cm, marker_px=0.0, reprojection_error_cm=0.0)


def register_sequence(
    frames: list[tuple[str, np.ndarray]],
    anchors: dict[str, PlaneScale],
    plane_id: str,
    marker_size_cm: float,
) -> PlaneTrack:
    """Register an ordered run of frames to the plane of their marker anchors.

    `anchors` maps frame id -> scale measured directly from a visible marker.
    Scale then propagates forwards and backwards from each anchor through
    consecutive links, stopping at the first weak link or after MAX_CHAIN hops.
    """
    track = PlaneTrack(plane_id, marker_size_cm)
    ids = [fid for fid, _ in frames]
    images = dict(frames)
    for fid, scale in anchors.items():
        track.plane_from_frame[fid] = scale.homography
        track.hops[fid] = 0

    links: dict[tuple[str, str], Link] = {}

    def get_link(a: str, b: str) -> Link:  # maps b pixels into a pixels
        if (a, b) not in links:
            links[(a, b)] = link(images[a], images[b])
        return links[(a, b)]

    for anchor in anchors:
        start = ids.index(anchor)
        for step in (1, -1):
            prev = anchor
            i = start + step
            while 0 <= i < len(ids):
                fid = ids[i]
                hops = track.hops[prev] + 1
                if hops > MAX_CHAIN:
                    break
                lk = get_link(prev, fid)
                if not lk.ok:
                    break
                if fid not in track.hops or hops < track.hops[fid]:
                    track.plane_from_frame[fid] = track.plane_from_frame[prev] @ lk.homography
                    track.hops[fid] = hops
                prev = fid
                i += step
    return track
