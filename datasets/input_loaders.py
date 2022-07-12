import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import os
import lmdb


class ActionAnticipationSampler(object):
    def __init__(self, t_buffer, t_ant=1.0, fps=4.0, ori_fps=60.0):
        self.t_buffer = t_buffer
        self.t_ant = t_ant
        self.fps = fps
        self.ori_fps = ori_fps

    def __call__(self, action):
        times, frames_idxs = sample_history_frames(action.start_frame, self.t_buffer,
                                                   self.t_ant, fps=self.fps,
                                                   fps_init=self.ori_fps)
        return times, frames_idxs


def get_sampler(args):
    sampler = ActionAnticipationSampler(t_buffer=args.t_buffer, t_ant=args.t_ant,
                                        fps=args.fps, ori_fps=args.ori_fps)
    return sampler


def sample_history_frames(frame_start, t_buffer=2.5, t_ant=1.0, fps=4.0, fps_init=60.0):
    time_start = (frame_start - 1) / fps_init
    num_frames = int(np.floor(t_buffer * fps))
    time_ant = time_start - t_ant
    times = (np.arange(1, num_frames + 1) - num_frames) / fps + time_ant
    times = np.clip(times, 0, np.inf)
    times = times.astype(np.float32)
    frames_idxs = np.floor(times * fps_init).astype(np.int32) + 1
    times = (frames_idxs - 1) / fps_init
    return times, frames_idxs


def sample_future_frames(frame_start, t_buffer=1, fps=4.0, fps_init=60.0):
    time_start = (frame_start - 1) / fps_init
    num_frames = int(np.floor(t_buffer * fps))
    times = (np.arange(num_frames + 1) - num_frames) / fps + time_start
    times = np.clip(times, 0, np.inf)
    times = times.astype(np.float32)
    frames_idxs = np.floor(times * fps_init).astype(np.int32) + 1
    if frames_idxs.max() >= 1:
        frames_idxs[frames_idxs < 1] = frames_idxs[frames_idxs >= 1].min()
    return list(frames_idxs)


class FeaturesLoader(object):
    def __init__(self, sampler, feature_base_path, fps, input_name='rgb',
                 frame_tmpl='frame_{:010d}.jpg', transform_feat=None,
                 transform_video=None):
        self.feature_base_path = feature_base_path
        self.env = lmdb.open(os.path.join(self.feature_base_path, input_name), readonly=True, lock=False)
        self.fps = fps
        self.input_name = input_name
        self.frame_tmpl = frame_tmpl
        self.transform_feat = transform_feat
        self.transform_video = transform_video
        self.sampler = sampler

    def __call__(self, action):
        times, frames_idxs = self.sampler(action)
        frames_names = [self.frame_tmpl.format(action.video_id, i) for i in frames_idxs]
        feats = []
        with self.env.begin() as env:
            for f_name in frames_names:
                feat = env.get(f_name.strip().encode('utf-8'))
                if feat is None:
                    print(f_name)
                feat = np.frombuffer(feat, 'float32')

                if self.transform_feat is not None:
                    feat = self.transform_feat(feat)
                feats += [feat]

        if self.transform_video is not None:
            feats = self.transform_video(feats)
        out = {self.input_name: feats}
        out['times'] = times
        out['start_time'] = action.start_time
        out['frames_idxs'] = frames_idxs
        return out


class PipeLoaders(object):
    def __init__(self, loader_list):
        self.loader_list = loader_list

    def __call__(self, action):
        out = {}
        for loader in self.loader_list:
            out.update(loader(action))
        return out


def get_features_loader(args, featuresloader=None):
    sampler = get_sampler(args)
    feat_in_modalities = list({'feat'}.intersection(args.modalities))
    transform_feat = lambda x: torch.tensor(x.copy())
    transform_video = lambda x: torch.stack(x, 0)
    loader_args = {
        'feature_base_path': args.features_paths[args.ek_version],
        'fps': args.fps,
        'frame_tmpl': 'frame_{:010d}.jpg',
        'transform_feat': transform_feat,
        'transform_video': transform_video,
        'sampler': sampler}
    if featuresloader is None:
        featuresloader = FeaturesLoader
    feat_loader_list = []
    for modality in feat_in_modalities:
        feat_loader = featuresloader(input_name=modality, **loader_args)
        feat_loader_list += [feat_loader]
    feat_loaders = {
        'train': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
        'validation': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
        'test': PipeLoaders(feat_loader_list) if len(feat_loader_list) else None,
    }
    return feat_loaders


def get_loaders(args, featuresloader=None):
    loaders = {
        'train': [],
        'validation': [],
        'test': [],
    }

    if 'feat' in args.modalities:
        feat_loaders = get_features_loader(args, featuresloader=featuresloader)
        for k, l in feat_loaders.items():
            if l is not None:
                loaders[k] += [l]

    for k, l in loaders.items():
        loaders[k] = PipeLoaders(l)
    return loaders
