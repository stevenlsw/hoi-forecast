import cv2
import numpy as np
from preprocess.dataset_util import bbox_inter


def skin_extract(image):
    def color_segmentation():
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")
        mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        return binary_mask_image

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    binary_mask_image = color_segmentation()
    image_foreground = cv2.erode(binary_mask_image, None, iterations=3)
    dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=3)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

    image_marker = cv2.add(image_foreground, image_background)
    image_marker32 = np.int32(image_marker)
    cv2.watershed(image, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)
    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((20, 20), np.uint8)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
    return image_mask


def farthest_sampling(pcd, n_samples, init_pcd=None):
    def compute_distance(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)

    n_pts, dim = pcd.shape[0], pcd.shape[1]
    selected_pts_expanded = np.zeros(shape=(n_samples, 1, dim))
    remaining_pts = np.copy(pcd)

    if init_pcd is None:
        if n_pts > 1:
            start_idx = np.random.randint(low=0, high=n_pts - 1)
        else:
            start_idx = 0
        selected_pts_expanded[0] = remaining_pts[start_idx]
        n_selected_pts = 1
    else:
        num_points = min(init_pcd.shape[0], n_samples)
        selected_pts_expanded[:num_points] = init_pcd[:num_points, None, :]
        n_selected_pts = num_points

    for _ in range(1, n_samples):
        if n_selected_pts < n_samples:
            dist_pts_to_selected = compute_distance(remaining_pts, selected_pts_expanded[:n_selected_pts]).T
            dist_pts_to_selected_min = np.min(dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            selected_pts_expanded[n_selected_pts] = remaining_pts[res_selected_idx]
            n_selected_pts += 1

    selected_pts = np.squeeze(selected_pts_expanded, axis=1)
    return selected_pts


def compute_heatmap(points, image_size, k_ratio=3.0):
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        col = int(x)
        row = int(y)
        try:
            heatmap[col, row] += 1.0
        except:
            col = min(max(col, 0), image_size[0] - 1)
            row = min(max(row, 0), image_size[1] - 1)
            heatmap[col, row] += 1.0
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap.transpose()
    return heatmap


def select_points_bbox(bbox, points, tolerance=2):
    x1, y1, x2, y2 = bbox
    ind_x = np.logical_and(points[:, 0] > x1-tolerance, points[:, 0] < x2+tolerance)
    ind_y = np.logical_and(points[:, 1] > y1-tolerance, points[:, 1] < y2+tolerance)
    ind = np.logical_and(ind_x, ind_y)
    indices = np.where(ind == True)[0]
    return points[indices]


def find_contour_points(mask):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        c = c.squeeze(axis=1)
        return c
    else:
        return None


def get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj):
    # active_obj_traj: active obj traj in last observation frame
    # obj_bboxs_traj: active obj bbox traj in last observation frame
    select_points_homo = np.concatenate((select_points, np.ones((select_points.shape[0], 1), dtype=np.float32)), axis=1)
    select_points_homo = np.dot(select_points_homo, homography.T)
    select_points_homo = select_points_homo[:, :2] / select_points_homo[:, None, 2]

    obj_point_last_observe = np.array(active_obj_traj[0])
    obj_point_future_start = np.array(active_obj_traj[-1])

    future2last_trans = obj_point_last_observe - obj_point_future_start
    select_points_homo = select_points_homo + future2last_trans

    fill_indices = [idx for idx, points in enumerate(obj_bboxs_traj) if points is not None]
    contour_last_observe = obj_bboxs_traj[fill_indices[0]]
    contour_future_homo = obj_bboxs_traj[fill_indices[-1]] + future2last_trans
    contour_last_observe = contour_last_observe[:, None, :].astype(np.int)
    contour_future_homo = contour_future_homo[:, None, :].astype(np.int)
    filtered_points = []
    for point in select_points_homo:
        if cv2.pointPolygonTest(contour_last_observe, (point[0], point[1]), False) >= 0 \
                or cv2.pointPolygonTest(contour_future_homo, (point[0], point[1]), False) >= 0:
            filtered_points.append(point)
    filtered_points = np.array(filtered_points)
    return filtered_points


def compute_affordance(frame, active_hand, active_obj, num_points=5, num_sampling=20):
    skin_mask = skin_extract(frame)
    hand_bbox = np.array(active_hand.bbox.coords_int).reshape(-1)
    obj_bbox = np.array(active_obj.bbox.coords_int).reshape(-1)
    obj_center = active_obj.bbox.center
    xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
    if not iou > 0:
        return None
    x1, y1, x2, y2 = hand_bbox
    hand_mask = np.zeros_like(skin_mask, dtype=np.uint8)
    hand_mask[y1:y2, x1:x2] = 255
    hand_mask = cv2.bitwise_and(skin_mask, hand_mask)
    select_points, init_points = None, None
    contact_points = find_contour_points(hand_mask)

    if contact_points is not None and contact_points.shape[0] > 0:
        contact_points = select_points_bbox((xA, yA, xB, yB), contact_points)
        if contact_points.shape[0] >= num_points:
            if contact_points.shape[0] > num_sampling:
                contact_points = farthest_sampling(contact_points, n_samples=num_sampling)
            distance = np.linalg.norm(contact_points - obj_center, ord=2, axis=1)
            indices = np.argsort(distance)[:num_points]
            select_points = contact_points[indices]
        elif contact_points.shape[0] > 0:
            print("no enough boundary points detected, sampling points in interaction region")
            init_points = contact_points
        else:
            print("no boundary points detected, use farthest point sampling")
    else:
        print("no boundary points detected, use farthest point sampling")
    if select_points is None:
        ho_mask = np.zeros_like(skin_mask, dtype=np.uint8)
        ho_mask[yA:yB, xA:xB] = 255
        ho_mask = cv2.bitwise_and(skin_mask, ho_mask)
        points = np.array(np.where(ho_mask[yA:yB, xA:xB] > 0)).T
        if points.shape[0] == 0:
            xx, yy = np.meshgrid(np.arange(xB - xA), np.arange(yB - yA))
            xx += xA
            yy += yA
            points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        else:
            points = points[:, [1, 0]]
            points[:, 0] += xA
            points[:, 1] += yA
        if not points.shape[0] > 0:
            return None
        contact_points = farthest_sampling(points, n_samples=min(num_sampling, points.shape[0]), init_pcd=init_points)
        distance = np.linalg.norm(contact_points - obj_center, ord=2, axis=1)
        indices = np.argsort(distance)[:num_points]
        select_points = contact_points[indices]
    return select_points


def compute_obj_affordance(frame, annot, active_obj, active_obj_idx, homography,
                           active_obj_traj, obj_bboxs_traj,
                           num_points=5, num_sampling=20,
                           hand_threshold=0.1, obj_threshold=0.1):
    affordance_info = {}
    hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                         hand_threshold=hand_threshold)
    select_points = None
    for hand_idx, object_idx in hand_object_idx_correspondences.items():
        if object_idx == active_obj_idx:
            active_hand = annot.hands[hand_idx]
            affordance_info[active_hand.side.name] = np.array(active_hand.bbox.coords_int).reshape(-1)
            cmap_points = compute_affordance(frame, active_hand, active_obj, num_points=num_points, num_sampling=num_sampling)
            if select_points is None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = cmap_points
            elif select_points is not None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = np.concatenate((select_points, cmap_points), axis=0)
    if select_points is None:
        print("affordance contact points filtered out")
        return None
    select_points_homo = get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj)
    if len(select_points_homo) == 0:
        print("affordance contact points filtered out")
        return None
    else:
        affordance_info["select_points"] = select_points
        affordance_info["select_points_homo"] = select_points_homo

        obj_bbox = np.array(active_obj.bbox.coords_int).reshape(-1)
        affordance_info["obj_bbox"] = obj_bbox
        return affordance_info