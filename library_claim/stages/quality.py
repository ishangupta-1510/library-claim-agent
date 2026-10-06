"""Per-frame capture quality: the facts the live agent uses to direct the sweep.

All checks are deterministic image statistics, so the agent's "slow down" or
"too much glare" is grounded in a measurement, not the model's impression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Thresholds calibrated on 1280-px-wide frames; see tests for the reasoning.
BLUR_MIN = 60.0  # variance of the Laplacian below this = motion blur / out of focus
GLARE_MAX = 0.06  # share of blown-out pixels above this = glare
DARK_MAX_MEAN = 45.0  # mean brightness below this = too dark to read spines


@dataclass
class FrameQuality:
    sharpness: float
    glare_fraction: float
    brightness: float
    blurry: bool
    glare: bool
    dark: bool

    @property
    def usable(self) -> bool:
        return not (self.blurry or self.dark)

    def problems(self) -> list[str]:
        return [name for name, bad in (("blur", self.blurry), ("glare", self.glare), ("dark", self.dark)) if bad]

    def to_dict(self) -> dict:
        return {**asdict(self), "usable": self.usable, "problems": self.problems()}


def assess(image: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Normalise width so the sharpness threshold means the same thing at any resolution.
    if gray.shape[1] != 1280:
        gray = cv2.resize(gray, (1280, int(gray.shape[0] * 1280 / gray.shape[1])), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare = float((gray >= 250).mean())
    brightness = float(gray.mean())
    return FrameQuality(
        sharpness=round(sharpness, 1),
        glare_fraction=round(glare, 4),
        brightness=round(brightness, 1),
        blurry=sharpness < BLUR_MIN,
        glare=glare > GLARE_MAX,
        dark=brightness < DARK_MAX_MEAN,
    )
