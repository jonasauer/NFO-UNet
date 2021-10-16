import dill

from config.config import AbstractConfig

base_config = AbstractConfig({
    # dataset (KthDataSet | MnistDataSet | TestingDataSet)
    'dataset_type': None,
    # path to test data root
    'test_data': None,
    # exclude certain folders from the dataset inside of the train and eval data folder
    'exclude_dirs': [],
    # search string specifying which heatmap type to use (either use _gauss or _circle)
    'hm_filter': None,
    # batch size for training and evaluation
    'batch_size': None,
    # number of frames in one sequence
    'seq_size': None,
    # every nth frame will be included in the sequences
    'nth_frame': 1,
    # number of workers for loading data samples
    'num_workers': 0,
    # shuffle the dataset
    'shuffle': True,
    # pin memory of the data loader
    'pin_memory': True,
    # transforms to use for testing (order matters)
    'test_transforms': [],
    # evaluation to use
    'eval_method': None
})
config = base_config.copy()


def set_cfg(config_name: str):
    config.replace(config.copy(eval(config_name)))


def load_cfg(file_path: str):
    with open(file_path, 'rb') as input_file:
        config.replace(dill.load(input_file))
