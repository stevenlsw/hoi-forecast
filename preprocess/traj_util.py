import numpy as np
import cv2
from preprocess.dataset_util import get_mask, valid_traj


def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.7, reprojThresh=4.0):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0]))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        matchesMask = status.ravel().tolist()
        return matches, H, matchesMask
    return None


def get_pair_homography(frame_1, frame_2, annot_1, annot_2, hand_threshold=0.1, obj_threshold=0.1):
    flag = True
    descriptor = cv2.xfeatures2d.SURF_create()
    msk_img_1 = get_mask(frame_1, annot_1, hand_threshold=hand_threshold, obj_threshold=obj_threshold)
    msk_img_2 = get_mask(frame_2, annot_2, hand_threshold=hand_threshold, obj_threshold=obj_threshold)
    (kpsA, featuresA) = descriptor.detectAndCompute(frame_1, mask=msk_img_1)
    (kpsB, featuresB) = descriptor.detectAndCompute(frame_2, mask=msk_img_2)
    matches, matchesMask = None, None
    try:
        (matches, H_BA, matchesMask) = match_keypoints(kpsB, kpsA, featuresB, featuresA)
    except Exception:
        print("compute homography failed!")
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False

    NoneType = type(None)
    if type(H_BA) == NoneType:
        print("compute homography failed!")
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    try:
        np.linalg.inv(H_BA)
    except Exception:
        print("compute homography failed!")
        H_BA = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3, 3)
        flag = False
    return matches, H_BA, matchesMask, flag


def get_homo_point(point, homography):
    cx, cy = point
    center = np.array((cx, cy, 1.0), dtype=np.float32)
    x, y, z = np.dot(homography, center)
    x, y = x / z, y / z
    point = np.array((x, y), dtype=np.float32)
    return point


def get_homo_bbox_point(bbox, homography):
    x1, y1, x2, y2 = np.array(bbox).reshape(-1)
    points = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
    points_homo = np.concatenate((points, np.ones((4, 1), dtype=np.float32)), axis=1)
    points_coord = np.dot(points_homo, homography.T)
    points_coord2d = points_coord[:, :2] / points_coord[:, None, 2]
    return points_coord2d


def get_hand_center(annot, hand_threshold=0.1):
    hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
    hands_center= {}
    hands_score = {}
    for hand in hands:
        side = hand.side.name
        score = hand.score
        if side not in hands_center or score > hands_score[side]:
            hands_center[side] = hand.bbox.center
            hands_score[side] = score
    return hands_center


def get_hand_point(hands_center, homography, side):
    point, homo_point = None, None
    if side in hands_center:
        point = hands_center[side]
        homo_point = get_homo_point(point, homography)
    return point, homo_point


def traj_compute(frames, annots, hand_sides, hand_threshold=0.1, obj_threshold=0.1):
    imgH, imgW = frames[0].shape[:2]
    left_traj, right_traj = [], []
    left_centers , right_centers= [], []
    homography_stack = [np.eye(3)]
    for idx in range(1, len(frames)):
        matches, H_BA, matchesMask, flag = get_pair_homography(frames[idx - 1], frames[idx],
                                                               annots[idx - 1], annots[idx],
                                                               hand_threshold=hand_threshold,
                                                               obj_threshold=obj_threshold)
        if not flag:
            return None
        else:
            homography_stack.append(np.dot(homography_stack[-1], H_BA))
    for idx in range(len(frames)):
        hands_center = get_hand_center(annots[idx], hand_threshold=hand_threshold)
        if "LEFT" in hand_sides:
            left_center, left_point = get_hand_point(hands_center, homography_stack[idx], "LEFT")
            left_centers.append(left_center)
            left_traj.append(left_point)
        if "RIGHT" in hand_sides:
            right_center, right_point = get_hand_point(hands_center, homography_stack[idx], "RIGHT")
            right_centers.append(right_center)
            right_traj.append(right_point)

    left_traj = valid_traj(left_traj, imgW=imgW, imgH=imgH)
    right_traj = valid_traj(right_traj, imgW=imgW, imgH=imgH)
    return left_traj, left_centers, right_traj, right_centers, homography_stack


def traj_completion(traj, side, imgW=456, imgH=256):
    from scipy.interpolate import CubicHermiteSpline

    def get_valid_traj(traj, imgW, imgH):
        traj[traj < 0] = traj[traj >= 0].min()
        traj[:, 0][traj[:, 0] > 1.5 * imgW] = 1.5 * imgW
        traj[:, 1][traj[:, 1] > 1.5 * imgH] = 1.5 * imgH
        return traj

    def spline_interpolation(axis):
        fill_times = np.array(fill_indices, dtype=np.float32)
        fill_traj = np.array([traj[idx][axis] for idx in fill_indices], dtype=np.float32)
        dt = fill_times[2:] - fill_times[:-2]
        dt = np.hstack([fill_times[1] - fill_times[0], dt, fill_times[-1] - fill_times[-2]])
        dx = fill_traj[2:] - fill_traj[:-2]
        dx = np.hstack([fill_traj[1] - fill_traj[0], dx, fill_traj[-1] - fill_traj[-2]])
        dxdt = dx / dt
        curve = CubicHermiteSpline(fill_times, fill_traj, dxdt)
        full_traj = curve(np.arange(len(traj), dtype=np.float32))
        return full_traj, curve

    fill_indices = [idx for idx, point in enumerate(traj) if point is not None]
    if 0 not in fill_indices:
        if side == "LEFT":
            traj[0] = np.array((0.25*imgW, 1.5*imgH), dtype=np.float32)
        else:
            traj[0] = np.array((0.75*imgW, 1.5*imgH), dtype=np.float32)
        fill_indices = np.insert(fill_indices, 0, 0).tolist()
    fill_indices.sort()
    full_traj_x, curve_x = spline_interpolation(axis=0)
    full_traj_y, curve_y = spline_interpolation(axis=1)
    full_traj = np.stack([full_traj_x, full_traj_y], axis=1)
    full_traj = get_valid_traj(full_traj, imgW=imgW, imgH=imgH)
    curve = [curve_x, curve_y]
    return full_traj, fill_indices, curve


def compute_hand_traj(frames, annots, hand_sides, hand_threshold=0.1, obj_threshold=0.1):
    imgH, imgW = frames[0].shape[:2]
    results = traj_compute(frames, annots, hand_sides,
                           hand_threshold=hand_threshold, obj_threshold=obj_threshold)
    if results is None:
        print("compute homography failed")
        return None
    else:
        left_traj, left_centers, right_traj, right_centers, homography_stack = results
        if len(left_traj) == 0 and len(right_traj) == 0:
            print("compute traj failed")
            return None
        hand_trajs = {}
        if len(left_traj) == 0:
            print("left traj filtered out")
        else:
            left_complete_traj, left_fill_indices, left_curve = traj_completion(left_traj, side="LEFT",
                                                                                imgW=imgW, imgH=imgH)
            hand_trajs["LEFT"] = {"traj": left_complete_traj, "fill_indices": left_fill_indices,
                                  "fit_curve": left_curve, "centers": left_centers}
        if len(right_traj) == 0:
            print("right traj filtered out")
        else:
            right_complete_traj, right_fill_indices, right_curve = traj_completion(right_traj, side="RIGHT",
                                                                                   imgW=imgW, imgH=imgH)
            hand_trajs["RIGHT"] = {"traj": right_complete_traj, "fill_indices": right_fill_indices,
                                   "fit_curve": right_curve, "centers": right_centers}
            return homography_stack, hand_trajs
