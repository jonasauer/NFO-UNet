from config.config import AbstractConfig

base_config = AbstractConfig({
    # number of sequences to be generated
    "num_seq": 7000,
    # number of images to be generated per sequence
    "seq_size": 5,
    # image size of sequences to be generated
    "img_size": 224,
    # digit size in images to be generated (from, to)
    "obj_scale_interval": (20, 30),
    # move speed of digits in sequences to be generated (from, to)
    "move_speed_interval": (3, 7),
    # root directory which will hold all generated sequences
    "out_dir": None,
    # deviation of the gaussian curves, which will serve as labels in the heatmap
    "gauss_std": (5, 5),
    # radius of circle used in circle heatmap
    'circle_radius': 0.07
})

config = base_config.copy()


def set_cfg(config_name: str):
    global config
    config.replace(eval(config_name))
