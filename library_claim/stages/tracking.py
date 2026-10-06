"""Chain frames onto a shelf plane so every frame shares one coordinate system.

A shelving unit's face is (close to) a plane, and two views of a plane are
related exactly by a homography. So once one frame of a unit contains the
marker, every overlapping frame can be mapped onto the same centimetre grid:

    plane_from_frame_k = plane_from_frame_j @ frame_j_from_frame_k

That gives scale to frames without a marker, and puts the same spine seen in
several frames at the same plane position, which is how duplicates merge.

Features are SIFT, computed once per frame and reused for every link attempt.
ORB was tried first and lost the chain at row changes and on tilted views; it
also recomputed features for every candidate pair, which made one sweep take
over ten minutes on CPU.

A link is trusted only with enough RANSAC inliers. A weak link breaks the
chain: those frames stay unscaled rather than getting a drifting, wrong scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .scale import PlaneScale

MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.30
MAX_CHAIN = 30  # links from the marker frame; beyond this drift is not trusted
FEATURE_WIDTH = 960  # features are computed on a downscaled copy; homographies are rescaled back


@dataclass
class Features:
    keypoints: np.ndarray  # Nx2 float32, in full-resolution pixels
    descriptors: np.ndarray | None  # float32 SIFT descriptors

    def compact(self) -> "Features":
        """SIFT descriptor values fit in 0-255; uint8 storage is 4x smaller for long-lived keyframes."""
        if self.descriptors is None:
            return self
        return Features(self.keypoints, np.clip(self.descriptors, 0, 255).astype(np.uint8))


def features(image: np.ndarray) -> Features:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, FEATURE_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    keypoints, descriptors = cv2.SIFT_create(nfeatures=2500).detectAndCompute(small, None)
    points = np.float32([k.pt for k in keypoints]) / scale if keypoints else np.zeros((0, 2), np.float32)
    return Features(points, descriptors)


# Approximate nearest neighbours (KD-trees) instead of brute force: several times faster
# for 2500 x 2500 SIFT descriptors, with the same ratio test and RANSAC afterwards.
_MATCHER = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 48})


@dataclass
class Link:
    homography: np.ndarray | None  # maps frame-k pixels into frame-j pixels
    inliers: int
    inlier_ratio: float

    @property
    def ok(self) -> bool:
        return self.homography is not None and self.inliers >= MIN_INLIERS and self.inlier_ratio >= MIN_INLIER_RATIO


def link_features(fj: Features, fk: Features) -> Link:
    """Homography taking frame_k pixels into frame_j pixels."""
    if fj.descriptors is None or fk.descriptors is None or len(fj.keypoints) < 8 or len(fk.keypoints) < 8:
        return Link(None, 0, 0.0)
    matches = _MATCHER.knnMatch(fk.descriptors.astype(np.float32), fj.descriptors.astype(np.float32), k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return Link(None, len(good), 0.0)
    src = fk.keypoints[[m.queryIdx for m in good]].reshape(-1, 1, 2)
    dst = fj.keypoints[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return Link(homography, inliers, inliers / len(good))


def link(frame_j: np.ndarray, frame_k: np.ndarray) -> Link:
    """Convenience: link two images directly (computes features for both)."""
    return link_features(features(frame_j), features(frame_k))


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

    Scale propagates forwards and backwards from each anchor through
    consecutive links, stopping at the first weak link or after MAX_CHAIN hops.
    """
    track = PlaneTrack(plane_id, marker_size_cm)
    ids = [fid for fid, _ in frames]
    feats = {fid: features(image) for fid, image in frames}
    for fid, scale in anchors.items():
        track.plane_from_frame[fid] = scale.homography
        track.hops[fid] = 0

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
                lk = link_features(feats[prev], feats[fid])
                if not lk.ok:
                    break
                if fid not in track.hops or hops < track.hops[fid]:
                    track.plane_from_frame[fid] = track.plane_from_frame[prev] @ lk.homography
                    track.hops[fid] = hops
                prev = fid
                i += step
    return track
