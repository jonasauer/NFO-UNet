from typing import List

import numpy as np

from utils.bb_utils import BoundingBox


class MovingObject:
    def __init__(self, img: np.ndarray, anchor: List[int], speed: List[int], boundaries: List[int]):
        self.img = img
        self.shape = img.shape
        self.anchor = anchor
        self.speed = speed
        self.boundaries = boundaries

    def move(self):
        self.anchor[0] += self.speed[0]
        self.anchor[1] += self.speed[1]
        if self.anchor[0] < 0:
            self.anchor[0] = -self.anchor[0]
            self.speed[0] = -self.speed[0]
        if self.anchor[1] < 0:
            self.anchor[1] = -self.anchor[1]
            self.speed[1] = -self.speed[1]
        if self.anchor[0] + self.shape[0] >= self.boundaries[0]:
            diff = self.boundaries[0] - (self.anchor[0] + self.shape[0])
            self.anchor[0] = self.boundaries[0] - self.shape[0] + diff
            self.speed[0] = -self.speed[0]
        if self.anchor[1] + self.shape[1] >= self.boundaries[1]:
            diff = self.boundaries[1] - (self.anchor[1] + self.shape[1])
            self.anchor[1] = self.boundaries[1] - self.shape[1] + diff
            self.speed[1] = -self.speed[1]

    def center(self):
        return self.anchor[0] + self.shape[0] // 2, self.anchor[1] + self.shape[1] // 2

    def bb(self):
        return BoundingBox(self.anchor[1], self.anchor[0], self.shape[1], self.shape[0])