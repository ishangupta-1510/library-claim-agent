"""Per-frame capture quality: the facts the live agent uses to direct the sweep.

All checks are image statistics, so "slow down" or "too much glare" is
grounded in a measurement rather than the model's impression.

Blur uses the "blur effect" of Crete-Roffet et al. (2007): how much detail a
frame loses when it is blurred further. A sharp frame loses a lot, an already
blurred one loses little, and because it is a ratio it does not depend on how
much texture the scene has. Two earlier measures failed calibration:
- Variance of the Laplacian: a sharp plain shelf (11) scored below a blurred
  bookshop photo (34); no threshold separates them.
- The same edge strength judged relative to recent frames: it flagged every
  frame that simply showed an emptier part of the shelf.
Calibration (dev_data photos + synthetic frames): real sharp 0.61-0.71, real
blurred 0.78-0.92; rendered synthetic frames are softer (sharp 0.81-0.89,
blurred 0.93-0.97). So a frame is blurry if it is smeared outright
(> BLUR_ABSOLUTE) or clearly blurrier than this sweep's own recent frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import cv2
import numpy as np

BLUR_ABSOLUTE = 0.92  # smeared by any standard
BLUR_ABOVE_RECENT = 0.07  # this much blurrier than the recent median = motion blur
HISTORY = 10
MIN_HISTORY = 3
FEATURELESS_EDGE = 6.0  # p99.5 |Laplacian| below this: a plain wall or the floor, nothing to read
GLARE_MAX = 0.06  # share of blown-out pixels (outside the marker) above this = glare
DARK_MAX_MEAN = 45.0  # mean brightness below this = too dark to read spines
WORK_WIDTH = 1280


@dataclass
class FrameQuality:
    blur_effect: float
    reference_blur: float | None
    edge_strength: float
    glare_fraction: float
    brightness: float
    blurry: bool
    glare: bool
    dark: bool
    featureless: bool

    @property
    def usable(self) -> bool:
        return not (self.blurry or self.dark)

    def problems(self) -> list[str]:
        flags = (("blur", self.blurry), ("glare", self.glare), ("dark", self.dark))
        return [name for name, bad in flags if bad]

    def to_dict(self) -> dict:
        return {**asdict(self), "usable": self.usable, "problems": self.problems()}


def blur_effect(gray: np.ndarray) -> float:
    """0 = sharp, 1 = completely blurred; the worse of the two directions (catches motion blur)."""
    image = gray.astype(np.float32)
    worst = 0.0
    for axis, kernel in ((0, (1, 9)), (1, (9, 1))):  # vertical, then horizontal re-blur
        reblurred = cv2.blur(image, kernel)
        d_image = np.abs(np.diff(image, axis=axis))
        d_blurred = np.abs(np.diff(reblurred, axis=axis))
        lost = np.maximum(0, d_image - d_blurred).sum()
        total = d_image.sum()
        worst = max(worst, float((total - lost) / total) if total > 0 else 1.0)
    return worst


def edge_strength(gray: np.ndarray) -> float:
    """99.5th percentile of |Laplacian|: is there anything in the frame at all?"""
    return float(np.percentile(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), 99.5))


class QualityMeter:
    """Scores frames against the sweep's own recent history."""

    def __init__(self) -> None:
        self.recent: deque[float] = deque(maxlen=HISTORY)

    def assess(self, image: np.ndarray, exclude_quads: list[np.ndarray] | None = None) -> FrameQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        scale = WORK_WIDTH / gray.shape[1]
        if scale != 1:
            gray = cv2.resize(gray, (WORK_WIDTH, int(gray.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        effect = blur_effect(gray)
        edges = edge_strength(gray)
        featureless = edges < FEATURELESS_EDGE
        reference = float(np.median(self.recent)) if len(self.recent) >= MIN_HISTORY else None
        blurry = not featureless and (
            effect > BLUR_ABSOLUTE or (reference is not None and effect > reference + BLUR_ABOVE_RECENT)
        )

        # Glare ignores the marker: its white quiet zone (or a bright laptop screen) is meant to be bright.
        mask = np.ones(gray.shape, np.uint8)
        for quad in exclude_quads or []:
            centre = quad.mean(axis=0)
            grown = (centre + (quad - centre) * 1.6) * scale
            cv2.fillConvexPoly(mask, grown.astype(np.int32), 0)
        saturated = (gray >= 250) & (mask > 0)
        glare = float(saturated.sum() / max(1, mask.sum()))
        brightness = float(gray.mean())
        dark = brightness < DARK_MAX_MEAN

        quality = FrameQuality(
            blur_effect=round(effect, 3),
            reference_blur=round(reference, 3) if reference is not None else None,
            edge_strength=round(edges, 1),
            glare_fraction=round(glare, 4),
            brightness=round(brightness, 1),
            blurry=blurry,
            glare=glare > GLARE_MAX,
            dark=dark,
            featureless=featureless,
        )
        if quality.usable and not featureless:
            self.recent.append(effect)
        return quality


def assess(image: np.ndarray) -> FrameQuality:
    """Single-frame check with no history: only outright smearing counts as blur."""
    return QualityMeter().assess(image)
