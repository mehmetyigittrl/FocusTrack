import csv
import os
import os.path
import random
from collections import OrderedDict

import numpy as np
import pandas
import torch

from lib.train.admin import env_settings
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset


class LSOTB_TIR(BaseVideoDataset):
    """ LSOTB-TIR: a large-scale thermal infrared single object tracking dataset.

    Layout: <root>/<sequence>/img/%06d.jpg with <root>/<sequence>/groundtruth.txt; the split file
    <root>/train.txt lists one sequence name per line.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        """
        args:
            root - path to the LSOTB-TIR dataset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - If split='train', the <root>/train.txt split file is used.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lsotb_tir_dir if root is None else root
        super().__init__('LSOTB_TIR', root, image_loader)

        self.sequence_list = self._get_sequence_list(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _get_sequence_list(self, split=None):
        if split is not None and split != 'train':
            raise ValueError('Unknown split name.')
        with open(os.path.join(self.root, 'train.txt'), 'r') as f:
            sequence_list = [line.strip() for line in f if line.strip()]
        return sequence_list

    def get_name(self):
        return 'lsotb_tir'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "full_occlusion.txt")
        out_of_view_file = os.path.join(seq_path, "out_of_view.txt")

        try:
            with open(occlusion_file, 'r', newline='') as f:
                occlusion = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])
            with open(out_of_view_file, 'r') as f:
                out_of_view = torch.ByteTensor([int(v) for v in list(csv.reader(f))[0]])

            target_visible = ~occlusion & ~out_of_view
        except OSError:
            # visibility flags are optional in this layout; assume the target is always visible
            target_visible = torch.ones(self._read_bb_anno(seq_path).shape[0], dtype=torch.uint8)

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:06}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self.sequence_list[seq_id]
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': obj_class,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
