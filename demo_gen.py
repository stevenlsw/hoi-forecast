import os
import pickle
import argparse
import pandas as pd
import cv2

from preprocess.traj_util import compute_hand_traj
from preprocess.dataset_util import FrameDetections, sample_action_anticipation_frames, fetch_data, save_video_info
from preprocess.obj_util import compute_obj_traj
from preprocess.affordance_util import compute_obj_affordance
from preprocess.vis_util import vis_affordance, vis_hand_traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', default="assets/EPIC_train_action_labels.csv", type=str, help="dataset annotation")
    parser.add_argument('--dataset_path', default="assets/EPIC-KITCHENS", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./figs", type=str, help="generated results save path")
    parser.add_argument('--fps', default=10, type=int, help="sample frames per second")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('--contact_ratio', default=0.4, type=float, help="active obj contact frames ratio")
    parser.add_argument('--num_sampling', default=20, type=int, help="sampling points for affordance")
    parser.add_argument('--num_points', default=5, type=int, help="selected points for affordance")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    uid = 52

    annotations = pd.read_csv(args.label_path)
    participant_id = annotations.loc[annotations['uid'] == uid].participant_id.item()
    video_id = annotations.loc[annotations['uid'] == uid].video_id.item()
    frames_path = os.path.join(args.dataset_path, participant_id, "rgb_frames", video_id)
    ho_path = os.path.join(args.dataset_path, participant_id, "hand-objects", "{}.pkl".format(video_id))
    start_act_frame = annotations.loc[annotations['uid'] == uid].start_frame.item()
    frames_idxs = sample_action_anticipation_frames(start_act_frame, fps=args.fps)

    with open(ho_path, "rb") as f:
        video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
    results = fetch_data(frames_path, video_detections, frames_idxs)
    if results is None:
        print("data fetch failed")
    else:
        frames_idxs, frames, annots, hand_sides = results

        results_hand = compute_hand_traj(frames, annots, hand_sides, hand_threshold=args.hand_threshold,
                                         obj_threshold=args.obj_threshold)
        if results_hand is None:
            print("compute traj failed")  # homography fails or not enough points
        else:
            homography_stack, hand_trajs = results_hand
            results_obj = compute_obj_traj(frames, annots, hand_sides, homography_stack,
                                           hand_threshold=args.hand_threshold,
                                           obj_threshold=args.obj_threshold,
                                           contact_ratio=args.contact_ratio)
            if results_obj is None:
                print("compute obj traj failed")
            else:
                contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj = results_obj
                frame, homography = frames[-1], homography_stack[-1]
                affordance_info = compute_obj_affordance(frame, annots[-1], active_obj, active_object_idx, homography,
                                                         active_obj_traj=obj_trajs['traj'], obj_bboxs_traj=obj_bboxs_traj,
                                                         num_points=args.num_points, num_sampling=args.num_sampling)
                if affordance_info is not None:
                    img_vis = vis_hand_traj(frames, hand_trajs)
                    img_vis = vis_affordance(img_vis, affordance_info)
                    img = cv2.hconcat([img_vis, frames[-1]])
                    cv2.imwrite(os.path.join(save_path, "demo_{}.jpg".format(uid)), img)
                    save_video_info(save_path, uid, frames_idxs, homography_stack, contacts, hand_trajs, obj_trajs, affordance_info)
    print(f"result stored at {save_path}")











