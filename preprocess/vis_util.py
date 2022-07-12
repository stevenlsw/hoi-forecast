import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess.affordance_util import compute_heatmap

hand_rgb = {"LEFT": (0, 90, 181), "RIGHT": (220, 50, 32)}
object_rgb = (255, 194, 10)


def vis_traj(frame_vis, traj, fill_indices=None, side=None, circle_radis=4, circle_thickness=3, line_thickness=2, style='line', gap=5):
    for idx in range(len(traj)):
        x, y = traj[idx]
        if fill_indices is not None and idx in fill_indices:
            thickness = -1
        else:
            thickness = -1
        color = hand_rgb[side][::-1] if side is not None else (0, 255, 255)
        frame_vis = cv2.circle(frame_vis, (int(round(x)), int(round(y))), radius=circle_radis, color=color,
                               thickness=thickness)
        if idx > 0:
            pt1 = (int(round(traj[idx-1][0])), int(round(traj[idx-1][1])))
            pt2 = (int(round(traj[idx][0])), int(round(traj[idx][1])))
            dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
            pts = []
            for i in np.arange(0, dist, gap):
                r = i / dist
                x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
                y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
                p = (x, y)
                pts.append(p)
            if style == 'dotted':
                for p in pts:
                    cv2.circle(frame_vis, p, circle_thickness, color, -1)
            else:
                if len(pts) > 0:
                    s = pts[0]
                    e = pts[0]
                    i = 0
                    for p in pts:
                        s = e
                        e = p
                        if i % 2 == 1:
                            cv2.line(frame_vis, s, e, color, line_thickness)
                        i += 1
    return frame_vis


def vis_hand_traj(frames, hand_trajs):
    frame_vis = frames[0].copy()
    for side in hand_trajs:
        meta = hand_trajs[side]
        traj, fill_indices = meta["traj"], meta["fill_indices"]
        frame_vis = vis_traj(frame_vis, traj, fill_indices, side)
    return frame_vis


def vis_affordance(frame, affordance_info):
    select_points = affordance_info["select_points_homo"]
    hmap = compute_heatmap(select_points, (frame.shape[1], frame.shape[0]))
    hmap = (hmap * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(hmap, colormap=cv2.COLORMAP_JET)
    for idx in range((len(select_points))):
        point = select_points[idx].astype(np.int)
        frame_vis = cv2.circle(frame, (point[0], point[1]), radius=2, color=(255, 0, 255),
                               thickness=-1)
    overlay = (0.7 * frame + 0.3 * hmap).astype(np.uint8)
    return overlay
