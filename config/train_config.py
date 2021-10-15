import dill

from config.config import AbstractConfig
from utils.fs_utils import ensure_dir

base_config = AbstractConfig({
    # dataset
    # dataset type to use
    'dataset_type': None,
    # path to training data root
    'train_data': None,
    # path to evaluation data root
    'eval_data': None,
    # exclude certain folders from the dataset inside of the train and eval data folder
    'exclude_dirs': [],
    # search string specifying which heatmap type to use (either use _gauss or _circle)
    'hm_filter': None,
    # criterion to be used in training
    'criterion': None,
    # batch size for training and evaluation
    'batch_size': None,
    # number of frames in one sequence
    'seq_size': None,
    # every nth frame will be included in the sequences
    'nth_frame': 1,
    # shuffle training dataset
    'shuffle': True,
    # number of workers for loading data samples
    'num_workers': 0,
    # pin memory of the data loader
    'pin_memory': True,
    # transforms to use for training (order matters)
    'train_transforms': [],
    # transforms to use for evaluation (order matters)
    'eval_transforms': [],

    # training
    # number of epochs for training
    'num_epochs': 100,
    # learning rate for training
    'lr': None,
    # every nth minibatch will be printed to the console (running mean loss or evaluation)
    'print_ma': 100,
    # if true, early stopping will be enabled
    'enable_early_stopping': True,
    # how many times the network can stay without improvement before early stopping will trigger
    'early_stopping_patience': 15,
    # True for visualizing, False otherwise
    'visualize': False
})
config = base_config.copy()


def set_cfg(config_name: str):
    config.replace(config.copy(eval(config_name)))


def persist_cfg(file_path: str):
    ensure_dir(file_path)
    with open(file_path, 'wb') as output_file:
        dill.dump(config, output_file, dill.HIGHEST_PROTOCOL)


def load_cfg(file_path: str):
    with open(file_path, 'rb') as input_file:
        config.replace(dill.load(input_file))
