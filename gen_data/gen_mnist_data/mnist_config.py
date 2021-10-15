from config.config import AbstractConfig

base_config = AbstractConfig({
    # number of sequences to be generated
    "num_seq": 1000,
    # path to mnist train images file (train-images-idx3-ubyte.gz)
    "mnist_images": "../../datasets/mnist/train-images-idx3-ubyte.gz",
    # path to mnist train labels file (train-labels-idx1-ubyte.gz)
    "mnist_labels": "../../datasets/mnist/train-labels-idx1-ubyte.gz",
    # number of images in mnist train dataset (changing not recommended)
    "mnist_num_images": 50_000,
    # image size of mnist sample (changing not recommended)
    "mnist_img_size": 28,
    # number of images to be generated per sequence
    "seq_size": 21,
    # image size of sequences to be generated
    "img_size": 224,
    # digit size in images to be generated (from, to)
    "digits_scale_interval": (28, 28),
    # move speed of digits in sequences to be generated (from, to)
    "move_speed_interval": (3, 7),
    # array holding all digits which should be annotated in the generated ground truth
    "detectable_digits": [0, 1, 2, 3, 4],
    # array holding all digits which should not be annotated in the generated ground truth
    "non_detectable_digits": [5, 6, 7, 8, 9],
    # number of annotated digits per image sequence to be generated (from, to)
    "num_detectable_digits_interval": (1, 1),
    # number of non annotated digits per image sequence to be generated (from, to)
    "num_non_detectable_digits_interval": (1, 1),
    # root directory which will hold all generated sequences
    "out_dir": None,
    # if None, background pixels will get assigned a random color, else the value of the property [0, 255]
    'bg_color': 0,
    # deviation of the gaussian curves, which will serve as labels in the heatmap
    "gauss_std": (5, 5),
    # radius of circle used in circle heatmap
    'circle_radius': 0.07
})

config = base_config.copy()


def set_cfg(config_name: str):
    global config
    config.replace(eval(config_name))
