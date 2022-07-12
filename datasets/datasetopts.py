import os
import json


class DatasetArgs(object):
    def __init__(self, ek_version='ek55', mode="train", use_label_only=True,
                 base_path="./", batch_size=32, num_workers=0, modalities=['feat'],
                 fps=4, t_buffer=2.5):

        self.features_paths = {
            'ek55': os.path.join(base_path, 'data/ek55/feats'),
            'ek100': os.path.join(base_path, 'data/ek100/feats')}

        # generated data labels
        self.label_path = {
            'ek55': os.path.join(base_path, 'data/ek55'),
            'ek100': os.path.join(base_path, 'data/ek100')}

        # amazon-annotated eval labels
        self.eval_label_path = {
            'ek55': os.path.join(base_path, 'data/ek55/ek55_eval_labels.pkl'),
            'ek100': os.path.join(base_path, 'data/ek100/ek100_eval_labels.pkl')
        }

        self.annot_path = {
            'ek55': os.path.join(base_path, 'common/epic-kitchens-55-annotations'),
            'ek100': os.path.join(base_path, 'common/epic-kitchens-100-annotations')}

        self.rulstm_annot_path = {
            'ek55': os.path.join(base_path, 'common/rulstm/RULSTM/data/ek55'),
            'ek100': os.path.join(base_path, 'common/rulstm/RULSTM/data/ek100')}

        self.pretrained_backbone_path = {
            'ek55': os.path.join(base_path, 'common/rulstm/FEATEXT/models/ek55', 'TSN-rgb.pth.tar'),
            'ek100': os.path.join(base_path, 'common/rulstm/FEATEXT/models/ek100', 'TSN-rgb-ek100.pth.tar'),
        }

        # default settings, no need changes
        if fps is None:
            self.fps = 4
        else:
            self.fps = fps

        if t_buffer is None:
            self.t_buffer = 2.5
        else:
            self.t_buffer = t_buffer

        self.ori_fps = 60.0
        self.t_ant = 1.0

        self.validation_ratio = 0.2
        self.use_rulstm_splits = True

        # only preprocess uids that have corresponding labels, in "video_info.json"
        self.use_label_only = use_label_only

        self.task = 'anticipation'
        self.num_actions_prev = 1

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.modalities = modalities
        self.ek_version = ek_version # 'ek55' or 'ek100'
        self.mode = mode # 'train'

    def add_attr(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)

    def has_attr(self, attr_name):
        return hasattr(self, attr_name)

    def __repr__(self):
        return 'Input Args: ' + json.dumps(self.__dict__, indent=4)

