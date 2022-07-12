from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import numpy as np

from datasets.dataset_utils import get_ek55_annotation, get_ek100_annotation
from datasets.input_loaders import get_loaders


class EpicAction(object):
    def __init__(self, uid, participant_id, video_id, verb, verb_class,
                 noun, noun_class, all_nouns, all_noun_classes, start_frame,
                 stop_frame, start_time, stop_time, ori_fps, partition, action, action_class):
        self.uid = uid
        self.participant_id = participant_id
        self.video_id = video_id
        self.verb = verb
        self.verb_class = verb_class
        self.noun = noun
        self.noun_class = noun_class
        self.all_nouns = all_nouns
        self.all_noun_classes = all_noun_classes
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.start_time = start_time
        self.stop_time = stop_time
        self.ori_fps = ori_fps
        self.partition = partition
        self.action = action
        self.action_class = action_class

        self.duration = self.stop_time - self.start_time

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

    def set_previous_actions(self, actions):
        self.actions_prev = actions


class EpicVideo(object):
    def __init__(self, df_video, ori_fps, partition, t_ant=None):
        self.df = df_video
        self.ori_fps = ori_fps
        self.partition = partition
        self.t_ant = t_ant

        self.actions, self.actions_invalid = self._get_actions()
        self.duration = max([a.stop_time for a in self.actions])

    def _get_actions(self):
        actions = []
        _actions_all = []
        actions_invalid = []
        for _, row in self.df.iterrows():
            action_args = {
                'uid': row.uid,
                'participant_id': row.participant_id,
                'video_id': row.video_id,
                'verb': row.verb if 'test' not in self.partition else None,
                'verb_class': row.verb_class if 'test' not in self.partition else None,
                'noun': row.noun if 'test' not in self.partition else None,
                'noun_class': row.noun_class if 'test' not in self.partition else None,
                'all_nouns': row.all_nouns if 'test' not in self.partition else None,
                'all_noun_classes': row.all_noun_classes if 'test' not in self.partition else None,
                'start_frame': row.start_frame,
                'stop_frame': row.stop_time,
                'start_time': row.start_time,
                'stop_time': row.stop_time,
                'ori_fps': self.ori_fps,
                'partition': self.partition,
                'action': row.action if 'test' not in self.partition else None,
                'action_class': row.action_class if 'test' not in self.partition else None,
            }
            action = EpicAction(**action_args)
            action.set_previous_actions([aa for aa in _actions_all])
            assert self.t_ant is not None
            assert self.t_ant > 0.0
            if action.start_time - self.t_ant >= 0:
                actions += [action]
            else:
                actions_invalid += [action]
            _actions_all += [action]
        return actions, actions_invalid


class EpicDataset(Dataset):
    def __init__(self, df, partition, ori_fps=60.0, fps=4.0, loader=None, t_ant=None, transform=None,
                 num_actions_prev=None, label_path=None, eval_label_path=None,
                 annot_path=None, rulstm_annot_path=None, ek_version=None):
        super().__init__()
        self.partition = partition
        self.ori_fps = ori_fps
        self.fps = fps
        self.df = df
        self.loader = loader
        self.t_ant = t_ant
        self.transform = transform
        self.num_actions_prev = num_actions_prev

        self.videos = self._get_videos()
        self.actions, self.actions_invalid = self._get_actions()

    def _get_videos(self):
        video_ids = sorted(list(set(self.df['video_id'].values.tolist())))
        videos = []
        pbar = tqdm(desc=f'Loading {self.partition} samples', total=len(self.df))
        for video_id in video_ids:
            video_args = {
                'df_video': self.df[self.df['video_id'] == video_id].copy(),
                'ori_fps': self.ori_fps,
                'partition': self.partition,
                't_ant': self.t_ant
            }
            video = EpicVideo(**video_args)
            videos += [video]
            pbar.update(len(video.actions))
        pbar.close()
        return videos

    def _get_actions(self):
        actions = []
        actions_invalid = []
        for video in self.videos:
            actions += video.actions
            actions_invalid += video.actions_invalid
        return actions, actions_invalid

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        a = self.actions[idx]
        sample = {'uid': a.uid}

        inputs = self.loader(a)
        sample.update(inputs)

        if 'test' not in self.partition:
            sample['verb_class'] = a.verb_class
            sample['noun_class'] = a.noun_class
            sample['action_class'] = a.action_class

        actions_prev = [-1] + [aa.action_class for aa in a.actions_prev]
        actions_prev = actions_prev[-self.num_actions_prev:]
        if len(actions_prev) < self.num_actions_prev:
            actions_prev = actions_prev[0:1] * (self.num_actions_prev - len(actions_prev)) + actions_prev
        actions_prev = np.array(actions_prev, dtype=np.int64)
        sample['action_class_prev'] = actions_prev
        return sample


def get_datasets(args, epic_ds=None, featuresloader=None):
    loaders = get_loaders(args, featuresloader=featuresloader)

    annotation_args = {
        'annot_path': args.annot_path,
        'label_path': args.label_path,
        'eval_label_path': args.eval_label_path,
        'rulstm_annot_path': args.rulstm_annot_path,
        'validation_ratio': args.validation_ratio,
        'use_rulstm_splits': args.use_rulstm_splits,
        'use_label_only': args.use_label_only
    }

    if args.ek_version == 'ek55':
        dfs = {
            'train': get_ek55_annotation(partition='train', **annotation_args),
            'validation': get_ek55_annotation(partition='validation', **annotation_args),
            'eval': get_ek55_annotation(partition='eval', **annotation_args),
            'test_s1': get_ek55_annotation(partition='test_s1', **annotation_args),
            'test_s2': get_ek55_annotation(partition='test_s2', **annotation_args),
        }
    elif args.ek_version == 'ek100':
        dfs = {
            'train': get_ek100_annotation(partition='train', **annotation_args),
            'validation': get_ek100_annotation(partition='validation', **annotation_args),
            'eval': get_ek100_annotation(partition='eval', **annotation_args),
            'test': get_ek100_annotation(partition='test', **annotation_args),
        }
    else:
        raise Exception(f'Error. EPIC-Kitchens Version "{args.ek_version}" not supported.')

    ds_args = {
        'label_path': args.label_path[args.ek_version],
        'eval_label_path': args.eval_label_path[args.ek_version],
        'annot_path': args.annot_path,
        'rulstm_annot_path': args.rulstm_annot_path[args.ek_version],
        'ek_version': args.ek_version,
        'ori_fps': args.ori_fps,
        'fps': args.fps,
        't_ant': args.t_ant,
        'num_actions_prev': args.num_actions_prev if args.task in ['anticipation'] else None,
    }

    if epic_ds is None:
        epic_ds = EpicDataset

    if args.mode in ['train', 'training']:
        dss = {
            'train': epic_ds(df=dfs['train'], partition='train', loader=loaders['train'], **ds_args),
            'validation': epic_ds(df=dfs['validation'], partition='validation', loader=loaders['validation'],
                                  **ds_args),
            'eval': epic_ds(df=dfs['eval'], partition='eval', loader=loaders['validation'], **ds_args),
        }
    elif args.mode in ['validation', 'validating', 'validate']:
        dss = {
            'validation': epic_ds(df=dfs['validation'], partition='validation',
                                  loader=loaders['validation'], **ds_args),
            'eval': epic_ds(df=dfs['eval'], partition='eval', loader=loaders['validation'], **ds_args),
        }
    elif args.mode in ['test', 'testing']:

        if args.ek_version == "ek55":
            dss = {
                'test_s1': epic_ds(df=dfs['test_s1'], partition='test_s1', loader=loaders['test'], **ds_args),
                'test_s2': epic_ds(df=dfs['test_s2'], partition='test_s2', loader=loaders['test'], **ds_args),
            }
        elif args.ek_version == "ek100":
            dss = {
                'test': epic_ds(df=dfs['test'], partition='test', loader=loaders['test'], **ds_args),
            }
    else:
        raise Exception(f'Error. Mode "{args.mode}" not supported.')

    return dss


def get_dataloaders(args, epic_ds=None, featuresloader=None):
    dss = get_datasets(args, epic_ds=epic_ds,featuresloader=featuresloader)
    dl_args = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }
    if args.mode in ['train', 'training']:
        dls = {
            'train': DataLoader(dss['train'], shuffle=False, **dl_args),
            'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        }
    elif args.mode in ['validate', 'validation', 'validating']:
        dls = {
            'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        }
    elif args.mode == 'test':
        if args.ek_version == "ek55":
            dls = {
                'test_s1': DataLoader(dss['test_s1'], shuffle=False, **dl_args),
                'test_s2': DataLoader(dss['test_s2'], shuffle=False, **dl_args),
            }
        elif args.ek_version == "ek100":
            dls = {
                'test': DataLoader(dss['test'], shuffle=False, **dl_args),
            }
    else:
        raise Exception(f'Error. Mode "{args.mode}" not supported.')
    return dls
