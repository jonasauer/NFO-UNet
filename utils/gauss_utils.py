from typing import Tuple

import numpy as np

from utils.bb_utils import BoundingBox


def __gauss_dens_norm(x: float, mu: float, sigma: float) -> float:
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.e ** (-np.power((x - mu) / sigma, 2) / 2) if sigma != 0 else 0


def generate_gauss(shape: Tuple[int, int], mu: Tuple[float, float], sigma: Tuple[float, float]) -> np.ndarray:
    height, width = shape
    kernel_width = np.linspace(0, width, width)
    kernel_height = np.linspace(0, height, height)

    for i in range(width):
        kernel_width[i] = __gauss_dens_norm(kernel_width[i], mu[1], sigma[1])
    for i in range(height):
        kernel_height[i] = __gauss_dens_norm(kernel_height[i], mu[0], sigma[0])

    kernel_2d = np.outer(kernel_height.T, kernel_width.T)
    kernel_max = kernel_2d.max()
    return kernel_2d if kernel_max == 0 else kernel_2d * 255.0 / kernel_2d.max()


def generate_circle(shape: Tuple[int, int], radius: float, pos: Tuple[int, int]):
    radius_squared = (radius * shape[0]) ** 2
    pos_bb = BoundingBox(pos[0], pos[1], 0, 0)
    hm = np.zeros(shape, dtype=np.uint8)
    for h in range(hm.shape[0]):
        for w in range(hm.shape[1]):
            curr_loc = (w + 0.5, h + 0.5)
            hm[h, w] = 255 if pos_bb.dist_squared(curr_loc) < radius_squared else 0
    return hm
