import logging
import os
import sys
from logging import handlers

from utils.fs_utils import ensure_dir


def init_logging(file: str = None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # add additional file handler
    if file is not None:
        ensure_dir(file)
        filename = os.path.join(file)
        fh = handlers.RotatingFileHandler(filename, maxBytes=1000_000, backupCount=3)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
