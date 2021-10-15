import math
from typing import Tuple

import cv2
import numpy as np

from utils.bb_utils import BoundingBox


class EvaluationMeasure:

    def __init__(self, threshold, dist_tol):
        self.threshold = threshold
        self.dist_tol = dist_tol
        self.f_p = 0
        self.t_p = 0
        self.f_n = 0
        self.samples = 0

    def reset(self):
        self.f_p = 0
        self.t_p = 0
        self.f_n = 0
        self.samples = 0

    def get_data(self) -> Tuple[int, int, int]:
        return self.t_p, self.f_p, self.f_n

    def evaluate(self, out, gt):
        out = out.cpu().detach().numpy()

        for i in range(len(out)):
            self.samples += 1
            img_out = out[i]
            img_out += abs(img_out.min())
            img_out /= abs(img_out.max())
            img_out = np.ndarray.astype(img_out * 255.0, np.uint8)
            img_out = np.transpose(img_out, (1, 2, 0))

            _, thresh = cv2.threshold(img_out, self.threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            pred = [cv2.boundingRect(c) for c in contours]
            gt_bbs = [BoundingBox(x=bb[1], y=bb[0], w=bb[3], h=bb[2]) for bb in gt[i]]
            pr_bbs = [BoundingBox(x=pr[0], y=pr[1], w=pr[2], h=pr[2]) for pr in pred]

            gt_eval = []
            for gt_bb in gt_bbs:
                bb_eval = False
                gt_x, gt_y = gt_bb.center()
                for pr_bb in pr_bbs:
                    pr_x, pr_y = pr_bb.center()
                    if math.sqrt((gt_y - pr_y) ** 2 + (gt_x - pr_x) ** 2) < self.dist_tol:
                        bb_eval = True
                gt_eval.append(bb_eval)
                if bb_eval:
                    self.t_p += 1
                else:
                    self.f_n += 1

            pred_eval = []
            for pr_bb in pr_bbs:
                bb_eval = False
                pr_x, pr_y = pr_bb.center()
                for gt_bb in gt_bbs:
                    gt_x, gt_y = gt_bb.center()
                    if math.sqrt((gt_y - pr_y) ** 2 + (gt_x - pr_x) ** 2) < self.dist_tol:
                        bb_eval = True
                pred_eval.append(bb_eval)
                if not bb_eval:
                    self.f_p += 1