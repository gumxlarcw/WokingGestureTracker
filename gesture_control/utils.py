from typing import Tuple
import numpy as np

def norm_dist_xy(a_xy, b_xy) -> float:
    dx, dy = (a_xy[0] - b_xy[0]), (a_xy[1] - b_xy[1])
    return float((dx*dx + dy*dy) ** 0.5)

def map_to_screen(nx: float, ny: float, W: int, H: int) -> Tuple[int,int]:
    x = int(np.clip(nx, 0.0, 1.0) * W)
    y = int(np.clip(ny, 0.0, 1.0) * H)
    return x, y

def var2(xs, ys) -> float:
    return float(np.var(xs) + np.var(ys))
