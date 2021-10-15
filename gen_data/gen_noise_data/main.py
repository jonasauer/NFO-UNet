import os
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

from gen_data.MovingObject import MovingObject
from gen_data.gen_noise_data import noise_config as conf
from utils.bb_utils import BoundingBox, save_bbs
from utils.fs_utils import ensure_dir
from utils.gauss_utils import generate_gauss, generate_circle
from utils.img_utils import overlay


def create_obj():
    r = np.random.randint
    scale = r(c.obj_scale_interval[0], c.obj_scale_interval[1] + 1)
    img = r(0, 256, (scale, scale))
    x = r(0, c.img_size - scale)
    y = r(0, c.img_size - scale)
    speed_x = r(c.move_speed_interval[0], c.move_speed_interval[1] + 1)
    speed_y = r(c.move_speed_interval[0], c.move_speed_interval[1] + 1)
    speed_x = speed_x if np.random.uniform(0, 1) < 0.5 else -speed_x
    speed_y = speed_y if np.random.uniform(0, 1) < 0.5 else -speed_y
    return MovingObject(img, [y, x], [speed_y, speed_x], [c.img_size, c.img_size])


def gen_seq(out_dir):
    ensure_dir(out_dir)
    obj = create_obj()
    bg = np.random.randint(0, 256, (c.img_size, c.img_size), np.uint8)

    bb_dict = {}
    for frame in range(c.seq_size):
        # gem imgs and bbs
        bb_dict[frame] = []
        bb_dict[frame].append(BoundingBox(obj.anchor[1], obj.anchor[0], obj.shape[1], obj.shape[0]).scale((1/s[0], 1/s[1])))
        hm_gauss = generate_gauss(bg.shape, obj.center(), c.gauss_std)
        hm_circle = generate_circle(bg.shape, c.circle_radius, (obj.center()[1], obj.center()[0]))
        img = overlay(bg, obj.img, obj.anchor)
        obj.move()

        # save images
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_or.jpg"), img)
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_gauss.jpg"), hm_gauss)
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_circle.jpg"), hm_circle)

    save_bbs(bb_dict, os.path.join(out_dir, "groundtruth.txt"))


if __name__ == '__main__':
    c = conf.config
    s = (c.img_size, c.img_size)
    for seq in tqdm(range(c.num_seq)):
        path = join(c.out_dir, f'seq{str(seq).zfill(5)}_gt/')
        gen_seq(path)
