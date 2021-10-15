import gzip
import os
import random
from os.path import join
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

from gen_data.MovingObject import MovingObject
from gen_data.gen_mnist_data import mnist_config as conf
from utils.bb_utils import BoundingBox, save_bbs
from utils.fs_utils import ensure_dir
from utils.gauss_utils import generate_gauss, generate_circle
from utils.img_utils import add_and_clip


def create_digit(mnist_dict, possible_digits):
    if len(possible_digits) > 0:
        label = random.choice(possible_digits)
        scale = np.random.randint(c.digits_scale_interval[0], c.digits_scale_interval[1] + 1)
        img = cv2.resize(random.choice(mnist_dict[label]), (scale, scale))
        x = np.random.randint(0, c.img_size - scale)
        y = np.random.randint(0, c.img_size - scale)
        speed_x = np.random.randint(c.move_speed_interval[0], c.move_speed_interval[1] + 1)
        speed_y = np.random.randint(c.move_speed_interval[0], c.move_speed_interval[1] + 1)
        speed_x = speed_x if np.random.uniform(0, 1) < 0.5 else -speed_x
        speed_y = speed_y if np.random.uniform(0, 1) < 0.5 else -speed_y
        return MovingObject(img, [y, x], [speed_y, speed_x], [c.img_size, c.img_size])
    return None


def generate_heatmap_gauss(shape: Tuple[int, int], gauss: np.ndarray, bb: BoundingBox):
    hm = np.zeros(shape, dtype=np.uint8)
    r_gauss = cv2.resize(gauss, (bb.h, bb.w))
    hm = add_and_clip(hm, r_gauss, (bb.y, bb.x))
    return hm


def gen_seq(mnist_dict, out_dir, precalculated_gauss: np.ndarray):
    ensure_dir(out_dir)
    detectable_digits = []
    non_detectable_digits = []
    num_detectable_digits = np.random.randint(c.num_detectable_digits_interval[0],
                                              c.num_detectable_digits_interval[1] + 1)
    num_non_detectable_digits = np.random.randint(c.num_non_detectable_digits_interval[0],
                                                  c.num_non_detectable_digits_interval[1] + 1)
    for frame in range(num_detectable_digits):
        detectable_digits.append(create_digit(mnist_dict, c.detectable_digits))
    for frame in range(num_non_detectable_digits):
        non_detectable_digits.append(create_digit(mnist_dict, c.non_detectable_digits))

    img_shape = (c.img_size, c.img_size)
    bg_color = np.random.randint(0, 255, img_shape, np.uint8)

    bb_dict = {}
    for frame in range(c.seq_size):
        bb_dict[frame] = []
        img = np.full(s, c.bg_color, dtype=np.uint32) if c.bg_color is not None else bg_color
        hm_gauss, hm_circle = np.zeros(s, np.uint32), np.zeros(s, np.uint32)
        for dig in detectable_digits:
            bb_dict[frame].append(dig.bb().scale((1/c.img_size, 1/c.img_size)))
            gauss = generate_heatmap_gauss(img.shape, precalculated_gauss, dig.bb()).astype(np.uint32)
            circle = generate_circle(img.shape, c.circle_radius, (dig.center()[1], dig.center()[0])).astype(np.uint32)
            hm_gauss = add_and_clip(hm_gauss, gauss, [0, 0])
            hm_circle = add_and_clip(hm_circle, circle, [0, 0])
        for dig in detectable_digits + non_detectable_digits:
            img = add_and_clip(img, dig.img, dig.anchor)
            dig.move()
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_or.jpg"), img.astype(np.uint8))
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_gauss.jpg"), hm_gauss.astype(np.uint8))
        cv2.imwrite(join(out_dir, f"{str(frame).zfill(5)}_circle.jpg"), hm_circle.astype(np.uint8))

    save_bbs(bb_dict, os.path.join(out_dir, "groundtruth.txt"))


def extract_mnist():
    # extract images
    with gzip.open(c.mnist_images, 'r') as f:
        f.read(16)
        buffer = f.read(c.mnist_img_size * c.mnist_img_size * c.mnist_num_images)

    images = np.frombuffer(buffer, dtype=np.uint8)
    images = images.reshape(c.mnist_num_images, c.mnist_img_size, c.mnist_img_size, 1).squeeze()
    # extract labels
    with gzip.open(c.mnist_labels, 'r') as f:
        f.read(8)
        buffer = f.read(c.mnist_num_images)

    labels = np.frombuffer(buffer, dtype=np.uint8)

    # create dict
    mnist_dict = {}
    for label, img in zip(labels, images):
        if label not in mnist_dict:
            mnist_dict[label] = []
        mnist_dict[label].append(img)

    return mnist_dict


if __name__ == '__main__':
    c = conf.config
    s = (c.img_size, c.img_size)
    mnist_dictionary = extract_mnist()

    # do not change this... this one is configured such that it is exactly fitting the border of the image
    precalculated_gauss = generate_gauss((198, 198), (99, 99), (30, 30)).astype(np.uint8)

    for seq in tqdm(range(c.num_seq)):
        path = join(c.out_dir, f'seq{str(seq).zfill(5)}_gt/')
        gen_seq(mnist_dictionary, path, precalculated_gauss)
