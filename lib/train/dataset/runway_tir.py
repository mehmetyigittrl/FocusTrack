import random

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from .lsotb_tir import LSOTB_TIR


class RunwayTIR(LSOTB_TIR):
    """ Private 1088x1088 single-channel thermal runway dataset (approaching aircraft).

    Same layout and split file as LSOTB-TIR: <root>/<sequence>/img/%06d.jpg with
    <root>/<sequence>/groundtruth.txt and split file <root>/train.txt.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        root = env_settings().runway_tir_dir if root is None else root
        BaseVideoDataset.__init__(self, 'RunwayTIR', root, image_loader)

        self.sequence_list = self._get_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def get_name(self):
        return 'runway_tir'
