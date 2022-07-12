import os
import cv2
import numpy as np
from preprocess.ho_types import FrameDetections, HandDetection, HandSide, HandState, ObjectDetection


def sample_action_anticipation_frames(frame_start, t_buffer=1, fps=4.0, fps_init=60.0):
    time_start = (frame_start - 1) / fps_init
    num_frames = int(np.floor(t_buffer * fps))
    times = (np.arange(num_frames + 1) - num_frames) / fps + time_start
    times = np.clip(times, 0, np.inf)
    times = times.astype(np.float32)
    frames_idxs = np.floor(times * fps_init).astype(np.int32) + 1
    if frames_idxs.max() >= 1:
        frames_idxs[frames_idxs < 1] = frames_idxs[frames_idxs >= 1].min()
    return list(frames_idxs)


def load_ho_annot(video_detections, frame_index, imgW=456, imgH=256):
    annot = video_detections[frame_index-1] # frame_index start from 1
    assert annot.frame_number == frame_index, "wrong frame index"
    annot.scale(width_factor=imgW, height_factor=imgH)
    return annot


def load_img(frames_path, frame_index):
    frame = cv2.imread(os.path.join(frames_path, "frame_{:010d}.jpg".format(frame_index)))
    return frame


def get_mask(frame, annot, hand_threshold=0.1, obj_threshold=0.1):
    msk_img = np.ones((frame.shape[:2]), dtype=frame.dtype)
    hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
    objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
    for hand in hands:
        (x1, y1), (x2, y2) = hand.bbox.coords_int
        msk_img[y1:y2, x1:x2] = 0

    if len(objs) > 0:
        hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                             hand_threshold=hand_threshold)
        for hand_idx, object_idx in hand_object_idx_correspondences.items():
            hand = annot.hands[hand_idx]
            object = annot.objects[object_idx]
            if not hand.state.value == HandState.STATIONARY_OBJECT.value:
                (x1, y1), (x2, y2) = object.bbox.coords_int
                msk_img[y1:y2, x1:x2] = 0
    return msk_img


def bbox_inter(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return xA, yA, xB, yB, 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return xA, yA, xB, yB, iou


def compute_iou(boxA, boxB):
    boxA = np.array(boxA).reshape(-1)
    boxB = np.array(boxB).reshape(-1)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def points_in_bbox(point, bbox):
    (x1, y1), (x2, y2) = bbox
    (x, y) = point
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def valid_point(point, imgW=456, imgH=256):
    if point is None:
        return False
    else:
        x, y = point
        return (0 <= x < imgW) and (0 <=y < imgH)


def valid_traj(traj, imgW=456, imgH=256):
    if len(traj) > 0:
        num_outlier = np.sum([not valid_point(point, imgW=imgW, imgH=imgH)
                              for point in traj if point is not None])
        valid_ratio = np.sum([valid_point(point, imgW=imgW, imgH=imgH) for point in traj[1:]]) / len(traj[1:])
        valid_last = valid_point(traj[-1], imgW=imgW, imgH=imgH)
        if num_outlier > 1 or valid_ratio < 0.5 or not valid_last:
            traj = []
    return traj


def get_valid_traj(traj, imgW=456, imgH=256):
    try:
        traj[traj < 0] = traj[traj >= 0].min()
    except:
        traj[traj < 0] = 0
    try:
        traj[:, 0][traj[:, 0] >= imgW] = imgW - 1
    except:
        traj[:, 0][traj[:, 0] >= imgW] = imgW - 1
    try:
        traj[:, 1][traj[:, 1] >= imgH] = imgH - 1
    except:
        traj[:, 1][traj[:, 1] >= imgH] = imgH - 1
    return traj


def fetch_data(frames_path, video_detections, frames_idxs, hand_threshold=0.1, obj_threshold=0.1):
    tolerance = frames_idxs[1] - frames_idxs[0] # extend future act frame by tolerance to find ho interaction
    frames = []
    annots = []

    miss_hand = 0
    for frame_idx in frames_idxs[:-1]:
        frame = load_img(frames_path, frame_idx)
        annot = load_ho_annot(video_detections, frame_idx)
        hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
        if len(hands) == 0:
            miss_hand += 1
        frames.append(frame)
        annots.append(annot)
    if miss_hand == len(frames_idxs[:-1]):
        return None
    frame_idx = frames_idxs[-1]
    frames_idxs = frames_idxs[:-1]

    hand_sides = []
    idx = 0
    flag = False
    while idx < tolerance:
        annot = load_ho_annot(video_detections, frame_idx)
        hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
        objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
        if len(hands) > 0 and len(objs) > 0: # at least one hand is contact with obj
            hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                                 hand_threshold=hand_threshold)
            for hand_idx, object_idx in hand_object_idx_correspondences.items():
                hand_bbox = np.array(annot.hands[hand_idx].bbox.coords).reshape(-1)
                obj_bbox = np.array(annot.objects[object_idx].bbox.coords).reshape(-1)
                xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
                contact_state = annot.hands[hand_idx].state.value
                if iou > 0 and (contact_state == HandState.STATIONARY_OBJECT.value or
                                contact_state == HandState.PORTABLE_OBJECT.value):
                    hand_side = annot.hands[hand_idx].side.name
                    hand_sides.append(hand_side)
                    flag = True
            if flag:
                break
            else:
                idx += 1
                frame_idx += 1
        else:
            idx += 1
            frame_idx += 1
    if flag:
        frames_idxs.append(frame_idx)
        frames.append(load_img(frames_path, frame_idx))
        annots.append(annot)
        return frames_idxs, frames, annots, list(set(hand_sides)) # remove redundant hand sides
    else:
        return None


def save_video_info(save_path, video_index, frames_idxs, homography_stack, contacts,
               hand_trajs, obj_trajs, affordance_info):
    import pickle
    video_info = {"frame_indices": frames_idxs,
                  "homography": homography_stack,
                  "contact": contacts}
    video_info.update({"hand_trajs": hand_trajs})
    video_info.update({"obj_trajs": obj_trajs})
    video_info.update({"affordance": affordance_info})
    with open(os.path.join(save_path, "label_{}.pkl".format(video_index)), 'wb') as f:
        pickle.dump(video_info, f)


def load_video_info(save_path, video_index):
    import pickle
    with open(os.path.join(save_path, "label_{}.pkl".format(video_index)), 'rb') as f:
        video_info = pickle.load(f)
    return video_info
