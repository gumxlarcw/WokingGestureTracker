import math
from typing import Optional
import numpy as np

class OneEuroFilter:
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x.astype(np.float32).copy()
            self.dx_prev = np.zeros_like(self.x_prev)
            return self.x_prev.copy()

        dt = max(t - float(self.t_prev), 1e-6)
        self.t_prev = t
        assert self.x_prev is not None and self.dx_prev is not None

        dx = (x - self.x_prev) / dt
        ad = self._alpha(self.dcutoff, dt)
        dx_hat = ad * dx + (1.0 - ad) * self.dx_prev

        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.mincutoff + self.beta * speed
        a = self._alpha(float(cutoff), dt)

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
