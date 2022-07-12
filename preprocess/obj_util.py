import numpy as np
from preprocess.dataset_util import bbox_inter, HandState, compute_iou, \
    valid_traj, get_valid_traj, points_in_bbox
from preprocess.traj_util import get_homo_point, get_homo_bbox_point


def find_active_side(annots, hand_sides, hand_threshold=0.1, obj_threshold=0.1):
    if len(hand_sides) == 1:
        return hand_sides[0]
    else:
        hand_counter = {"LEFT": 0, "RIGHT": 0}
        for annot in annots:
            hands = [hand for hand in annot.hands if hand.score >= hand_threshold]
            objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
            if len(hands) > 0 and len(objs) > 0:
                hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                                     hand_threshold=hand_threshold)
                for hand_idx, object_idx in hand_object_idx_correspondences.items():
                    hand_bbox = np.array(annot.hands[hand_idx].bbox.coords_int).reshape(-1)
                    obj_bbox = np.array(annot.objects[object_idx].bbox.coords_int).reshape(-1)
                    xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
                    if iou > 0:
                        hand_side = annot.hands[hand_idx].side.name
                        if annot.hands[hand_idx].state.value == HandState.PORTABLE_OBJECT.value:
                            hand_counter[hand_side] += 1
                        elif annot.hands[hand_idx].state.value == HandState.STATIONARY_OBJECT.value:
                            hand_counter[hand_side] += 0.5
        if hand_counter["LEFT"] == hand_counter["RIGHT"]:
            return "RIGHT"
        else:
            return max(hand_counter, key=hand_counter.get)


def compute_contact(annots, hand_side, contact_state, hand_threshold=0.1):
    contacts = []
    for annot in annots:
        hands = [hand for hand in annot.hands if hand.score >= hand_threshold
                 and hand.side.name == hand_side and hand.state.value == contact_state]
        if len(hands) > 0:
            contacts.append(1)
        else:
            contacts.append(0)
    contacts = np.array(contacts)
    padding_contacts = np.pad(contacts, [1, 1], 'edge')
    contacts = np.convolve(padding_contacts, [1, 1, 1], 'same')
    contacts = contacts[1:-1] / 3
    contacts = contacts > 0.5
    indices = np.diff(contacts) != 0
    if indices.sum() == 0:
        return contacts
    else:
        split = np.where(indices)[0] + 1
        contacts_idx = split[-1]
        contacts[:contacts_idx] = False
        return contacts


def find_active_obj_side(annot, hand_side, return_hand=False, return_idx=False, hand_threshold=0.1, obj_threshold=0.1):
    objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
    if len(objs) == 0:
        return None
    else:
        hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                             hand_threshold=hand_threshold)
        for hand_idx, object_idx in hand_object_idx_correspondences.items():
            if annot.hands[hand_idx].side.name == hand_side:
                if return_hand and return_idx:
                    return annot.objects[object_idx], object_idx, annot.hands[hand_idx], hand_idx
                elif return_hand:
                    return annot.objects[object_idx], annot.hands[hand_idx]
                elif return_idx:
                    return annot.objects[object_idx], object_idx
                else:
                    return annot.objects[object_idx]
        return None


def find_active_obj_iou(objs, bbox):
    max_iou = 0
    active_obj = None
    for obj in objs:
        iou = compute_iou(obj.bbox.coords, bbox)
        if iou > max_iou:
            max_iou = iou
            active_obj = obj
    return active_obj, max_iou


def traj_compute(annots, hand_sides, homography_stack, hand_threshold=0.1, obj_threshold=0.1):
    annot = annots[-1]
    obj_traj = []
    obj_centers = []
    obj_bboxs =[]
    obj_bboxs_traj = []
    active_hand_side = find_active_side(annots, hand_sides, hand_threshold=hand_threshold,
                                        obj_threshold=obj_threshold)
    active_obj, active_object_idx, active_hand, active_hand_idx = find_active_obj_side(annot,
                                                                                       hand_side=active_hand_side,
                                                                                       return_hand=True, return_idx=True,
                                                                                       hand_threshold=hand_threshold,
                                                                                       obj_threshold=obj_threshold)
    contact_state = active_hand.state.value
    contacts = compute_contact(annots, active_hand_side, contact_state,
                               hand_threshold=hand_threshold)
    obj_center = active_obj.bbox.center
    obj_centers.append(obj_center)
    obj_point = get_homo_point(obj_center, homography_stack[-1])
    obj_bbox = active_obj.bbox.coords
    obj_traj.append(obj_point)
    obj_bboxs.append(obj_bbox)

    obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[-1])
    obj_bboxs_traj.append(obj_points2d)

    for idx in np.arange(len(annots)-2, -1, -1):
        annot = annots[idx]
        objs = [obj for obj in annot.objects if obj.score >= obj_threshold]
        contact = contacts[idx]
        if not contact:
            obj_centers.append(None)
            obj_traj.append(None)
            obj_bboxs_traj.append(None)
        else:
            if len(objs) >= 2:
                target_obj, max_iou = find_active_obj_iou(objs, obj_bboxs[-1])
                if target_obj is None:
                    target_obj = find_active_obj_side(annot, hand_side=active_hand_side,
                                                      hand_threshold=hand_threshold,
                                                      obj_threshold=obj_threshold)
                if target_obj is None:
                    obj_centers.append(None)
                    obj_traj.append(None)
                    obj_bboxs_traj.append(None)
                else:
                    obj_center = target_obj.bbox.center
                    obj_centers.append(obj_center)
                    obj_point = get_homo_point(obj_center, homography_stack[idx])
                    obj_bbox = target_obj.bbox.coords
                    obj_traj.append(obj_point)
                    obj_bboxs.append(obj_bbox)

                    obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[idx])
                    obj_bboxs_traj.append(obj_points2d)

            elif len(objs) > 0:
                target_obj = find_active_obj_side(annot, hand_side=active_hand_side,
                                                  hand_threshold=hand_threshold,
                                                  obj_threshold=obj_threshold)
                if target_obj is None:
                    obj_centers.append(None)
                    obj_traj.append(None)
                    obj_bboxs_traj.append(None)
                else:
                    obj_center = target_obj.bbox.center
                    obj_centers.append(obj_center)
                    obj_point = get_homo_point(obj_center, homography_stack[idx])
                    obj_bbox = target_obj.bbox.coords
                    obj_traj.append(obj_point)
                    obj_bboxs.append(obj_bbox)

                    obj_points2d = get_homo_bbox_point(obj_bbox, homography_stack[idx])
                    obj_bboxs_traj.append(obj_points2d)
            else:
                obj_centers.append(None)
                obj_traj.append(None)
                obj_bboxs_traj.append(None)
    obj_bboxs.reverse()
    obj_traj.reverse()
    obj_centers.reverse()
    obj_bboxs_traj.reverse()
    return obj_traj, obj_centers, obj_bboxs, contacts, active_obj, active_object_idx, obj_bboxs_traj


def traj_filter(obj_traj, obj_centers, obj_bbox, contacts, homography_stack, contact_ratio=0.4):
    assert len(obj_traj) == len(obj_centers), "traj length and center length not equal"
    assert len(obj_centers) == len(homography_stack), "center length and homography length not equal"
    homo_last2first = homography_stack[-1]
    homo_first2last = np.linalg.inv(homo_last2first)
    obj_points = []
    obj_inside, obj_detect = [], []
    for idx, obj_center in enumerate(obj_centers):
        if obj_center is not None:
            homo_current2first = homography_stack[idx]
            homo_current2last = homo_current2first.dot(homo_first2last)
            obj_point = get_homo_point(obj_center, homo_current2last)
            obj_points.append(obj_point)
            obj_inside.append(points_in_bbox(obj_point, obj_bbox))
            obj_detect.append(True)
        else:
            obj_detect.append(False)
    obj_inside = np.array(obj_inside)
    obj_detect = np.array(obj_detect)
    contacts = np.bitwise_and(obj_detect, contacts)
    if np.sum(obj_inside) == len(obj_inside) and np.sum(contacts) / len(contacts) < contact_ratio:
        obj_traj = np.tile(obj_traj[-1], (len(obj_traj), 1))
    return obj_traj, contacts


def traj_completion(traj, imgW=456, imgH=256):
    fill_indices = [idx for idx, point in enumerate(traj) if point is not None]
    full_traj = traj.copy()
    if len(fill_indices) == 1:
        point = traj[fill_indices[0]]
        full_traj = np.array([point] * len(traj), dtype=np.float32)
    else:
        contact_time = fill_indices[0]
        if contact_time > 0:
            full_traj[:contact_time] = [traj[contact_time]] * contact_time
        for previous_idx, current_idx in zip(fill_indices[:-1], fill_indices[1:]):
            start_point, end_point = traj[previous_idx], traj[current_idx]
            time_expand = current_idx - previous_idx
            for idx in range(previous_idx+1, current_idx):
                full_traj[idx] = (idx-previous_idx) / time_expand * end_point + (current_idx-idx) / time_expand * start_point
    full_traj = np.array(full_traj, dtype=np.float32)
    full_traj = get_valid_traj(full_traj, imgW=imgW, imgH=imgH)
    return full_traj, fill_indices


def compute_obj_traj(frames, annots, hand_sides, homography_stack, hand_threshold=0.1, obj_threshold=0.1,
                     contact_ratio=0.4):
    imgH, imgW = frames[0].shape[:2]
    obj_traj, obj_centers, obj_bboxs, contacts, active_obj, active_object_idx, obj_bboxs_traj = traj_compute(annots, hand_sides, homography_stack,
                                                                                                             hand_threshold=hand_threshold, obj_threshold=obj_threshold)
    obj_traj, contacts = traj_filter(obj_traj, obj_centers, obj_bboxs[-1], contacts, homography_stack,
                                     contact_ratio=contact_ratio)
    obj_traj = valid_traj(obj_traj, imgW=imgW, imgH=imgH)
    if len(obj_traj) == 0:
        print("object traj filtered out")
        return None
    else:
        complete_traj, fill_indices = traj_completion(obj_traj, imgW=imgW, imgH=imgH)
        obj_trajs = {"traj": complete_traj, "fill_indices": fill_indices, "centers": obj_centers}
        return contacts, obj_trajs, active_obj, active_object_idx, obj_bboxs_traj