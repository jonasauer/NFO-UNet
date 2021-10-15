import numpy as np

from eval.abstract_eval import AbstractEval
from utils.bb_utils import BoundingBox


class MaxEval(AbstractEval):

    def __init__(self, max_dist_error: float, init_thresh: float = None):
        super().__init__(max_dist_error, init_thresh)

    def extract_centers(self, hm):
        max_index = np.argmax(hm)
        bbs = [] if self.init_thresh and np.amax(hm) <= 0 else [BoundingBox(max_index % 224, max_index // 224, 0, 0)]
        # calculate center using bounding box
        centers = [bb.center() for bb in bbs]
        s = hm.shape
        centers_norm = [bb.scale((1 / s[1], 1 / s[0])).center() for bb in bbs]
        return centers, centers_norm
