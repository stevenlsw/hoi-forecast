import os
import pickle
import numpy as np


def load_video_info(label_path, video_index):
    with open(os.path.join(label_path, "label_{}.pkl".format(video_index)), 'rb') as f:
        video_info = pickle.load(f)
    return video_info


def sample_hand_traj(meta, fps, t_ant, shape=(456, 256)):
    width, height = shape
    traj = meta["traj"]
    ori_fps = int((len(traj) - 1) / t_ant)
    gap = int(ori_fps // fps)
    stop_idx = len(traj)
    indices = [0] + list(range(gap, stop_idx, gap))
    hand_traj = []
    for idx in indices:
        x, y = traj[idx]
        x, y, = x / width, y / height
        hand_traj.append(np.array([x, y], dtype=np.float32))
    hand_traj = np.array(hand_traj, dtype=np.float32)
    return hand_traj, indices


def process_video_info(video_info, fps=4, t_ant=1.0, shape=(456, 256)):
    frames_idxs = video_info["frame_indices"]
    hand_trajs = video_info["hand_trajs"]
    obj_affordance = video_info['affordance']['select_points_homo']
    num_points = obj_affordance.shape[0]
    select_idx = np.random.choice(num_points, 1, replace=False)
    contact_point = obj_affordance[select_idx]
    cx, cy = contact_point[0]
    width, height = shape
    cx, cy = cx / width, cy/ height
    contact_point = np.array([cx, cy], dtype=np.float32)

    valid_mask = []
    if "RIGHT" in hand_trajs:
        meta = hand_trajs["RIGHT"]
        rhand_traj, indices = sample_hand_traj(meta, fps, t_ant, shape)
        valid_mask.append(1)
    else:
        length = int(fps * t_ant + 1)
        rhand_traj = np.repeat(np.array([[0.75, 1.5]], dtype=np.float32), length, axis=0)
        valid_mask.append(0)

    if "LEFT" in hand_trajs:
        meta = hand_trajs["LEFT"]
        lhand_traj, indices = sample_hand_traj(meta, fps, t_ant, shape)
        valid_mask.append(1)
    else:
        length = int(fps * t_ant + 1)
        lhand_traj = np.repeat(np.array([[0.25, 1.5]], dtype=np.float32), length, axis=0)
        valid_mask.append(0)

    future_hands = np.stack((rhand_traj, lhand_traj), axis=0)
    future_valid = np.array(valid_mask, dtype=np.int)

    last_frame_index = frames_idxs[0]
    return future_hands, contact_point, future_valid, last_frame_index


def process_eval_video_info(video_info, fps=4, t_ant=1.0):
    valid_mask = []
    if "RIGHT" in video_info:
        rhand_traj = video_info["RIGHT"]
        assert rhand_traj.shape[0] == int(fps * t_ant + 1)
        valid_mask.append(1)
    else:
        rhand_traj = np.repeat(np.array([[0.75, 1.5]], dtype=np.float32), int(fps * t_ant + 1), axis=0)
        valid_mask.append(0)

    if "LEFT" in video_info:
        lhand_traj = video_info['LEFT']
        assert lhand_traj.shape[0] == int(fps * t_ant + 1)
        valid_mask.append(1)
    else:
        lhand_traj = np.repeat(np.array([[0.25, 1.5]], dtype=np.float32), int(fps * t_ant + 1), axis=0)
        valid_mask.append(0)

    future_hands = np.stack((rhand_traj, lhand_traj), axis=0)
    future_valid = np.array(valid_mask, dtype=np.int)
    return future_hands, future_valid











