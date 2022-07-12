import numpy as np
import torch
import cv2
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


def farthest_sampling(pcd, n_samples):
    def compute_distance(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)

    n_pts, dim = pcd.shape[0], pcd.shape[1]
    selected_pts_expanded = np.zeros(shape=(n_samples, 1, dim))
    remaining_pts = np.copy(pcd)

    if n_pts > 1:
        start_idx = np.random.randint(low=0, high=n_pts - 1)
    else:
        start_idx = 0
    selected_pts_expanded[0] = remaining_pts[start_idx]
    n_selected_pts = 1

    for _ in range(1, n_samples):
        if n_selected_pts < n_samples:
            dist_pts_to_selected = compute_distance(remaining_pts, selected_pts_expanded[:n_selected_pts]).T
            dist_pts_to_selected_min = np.min(dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            selected_pts_expanded[n_selected_pts] = remaining_pts[res_selected_idx]
            n_selected_pts += 1

    selected_pts = np.squeeze(selected_pts_expanded, axis=1)
    return selected_pts


def makeGaussian(size, fwhm=3., center=None):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def compute_heatmap(normalized_points, image_size, k_ratio=3.0, transpose=True,
                    fps=False, kmeans=False, n_pts=5, gaussian_sigma=0.):
    normalized_points = np.asarray(normalized_points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = normalized_points.shape[0]
    if n_points > n_pts and kmeans:
        kmeans = KMeans(n_clusters=n_pts, random_state=0).fit(normalized_points)
        normalized_points = kmeans.cluster_centers_
    elif n_points > n_pts and fps:
        normalized_points = farthest_sampling(normalized_points, n_samples=n_pts)
    n_points = normalized_points.shape[0]
    for i in range(n_points):
        x = normalized_points[i, 0] * image_size[0]
        y = normalized_points[i, 1] * image_size[1]
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
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), gaussian_sigma)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    if transpose:
        heatmap = heatmap.transpose()
    return heatmap


def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    if not np.any(fixation_map):
        return np.nan
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape, interpolation=cv2.INTER_AREA)
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-7
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0;
    tp[-1] = 1
    fp[0] = 0;
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k + 1] = (k + 1) / float(n_fix)
        fp[k + 1] = (above_th - (k + 1)) / float(n_pixels - n_fix)
    return np.trapz(tp, fp)


def NSS(saliency_map, fixation_map):
    MAP = (saliency_map - saliency_map.mean()) / (saliency_map.std())
    mask = fixation_map.astype(np.bool)
    score = MAP[mask].mean()
    return score


def compute_score(pred, gt, valid_thresh=0.001):
    if torch.is_tensor(pred):
        pred = pred.numpy()
    if torch.is_tensor(gt):
        gt = gt.numpy()

    pred = pred / (pred.max() + 1e-12)

    all_thresh = np.linspace(0.001, 1.0, 41)
    tp = np.zeros((all_thresh.shape[0],))
    fp = np.zeros((all_thresh.shape[0],))
    fn = np.zeros((all_thresh.shape[0],))
    tn = np.zeros((all_thresh.shape[0],))
    valid_gt = gt > valid_thresh
    for idx, thresh in enumerate(all_thresh):
        mask = (pred >= thresh)
        tp[idx] += np.sum(np.logical_and(mask == 1, valid_gt == 1))
        tn[idx] += np.sum(np.logical_and(mask == 0, valid_gt == 0))
        fp[idx] += np.sum(np.logical_and(mask == 1, valid_gt == 0))
        fn[idx] += np.sum(np.logical_and(mask == 0, valid_gt == 1))

    scores = {}
    gt_real = np.array(gt)
    if gt_real.sum() == 0:
        gt_real = np.ones(gt_real.shape) / np.product(gt_real.shape)

    score = SIM(pred, gt_real)
    scores['SIM'] = score if not np.isnan(score) else None

    gt_binary = np.array(gt)
    gt_binary = (gt_binary / gt_binary.max() + 1e-12) if gt_binary.max() > 0 else gt_binary
    gt_binary = np.where(gt_binary > 0.5, 1, 0)
    score = AUC_Judd(pred, gt_binary)
    scores['AUC-J'] = score if not np.isnan(score) else None

    score = NSS(pred, gt_binary)
    scores['NSS'] = score if not np.isnan(score) else None

    return dict(scores), tp, tn, fp, fn


def evaluate_affordance(preds_dict, gts_dict, val_log=None,
                        sz=32, fps=False, kmeans=False, n_pts=5,
                        gaussian_sigma=3., gaussian_k_ratio=3.):
    scores = []
    all_thresh = np.linspace(0.001, 1.0, 41)
    tp = np.zeros((all_thresh.shape[0],))
    fp = np.zeros((all_thresh.shape[0],))
    fn = np.zeros((all_thresh.shape[0],))
    tn = np.zeros((all_thresh.shape[0],))

    pred_hmaps = Parallel(n_jobs=16, verbose=0)(delayed(compute_heatmap)(norm_contacts, (sz, sz),
                                                                         fps=fps, kmeans=kmeans, n_pts=n_pts,
                                                                         gaussian_sigma=gaussian_sigma,
                                                                         k_ratio=gaussian_k_ratio)
                                                for (uid, norm_contacts) in preds_dict.items())
    gt_hmaps = Parallel(n_jobs=16, verbose=0)(delayed(compute_heatmap)(norm_contacts, (sz, sz),
                                                                       fps=fps, n_pts=n_pts,
                                                                       gaussian_sigma=0,
                                                                       k_ratio=3.)
                                              for (uid, norm_contacts) in gts_dict.items())

    for (pred_hmap, gt_hmap) in zip(pred_hmaps, gt_hmaps):
        score, ctp, ctn, cfp, cfn = compute_score(pred_hmap, gt_hmap)
        scores.append(score)
        tp = tp + ctp
        tn = tn + ctn
        fp = fp + cfp
        fn = fn + cfn

    write_out = []
    metrics = {}
    for key in ['SIM', 'AUC-J', 'NSS']:
        key_score = [s[key] for s in scores if s[key] is not None]
        mean, stderr = np.mean(key_score), np.std(key_score) / (np.sqrt(len(key_score)))
        log_str = '%s: %.3f ± %.3f (%d/%d)' % (key, mean, stderr, len(key_score), len(gts_dict))
        write_out.append(log_str)
        metrics[key] = mean
    write_out = '\n'.join(write_out)
    print(write_out)
    if val_log is not None:
        with open(val_log, "a") as log_file:
            log_file.write(write_out + '\n')

    write_out = []
    prec = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * prec * recall / (prec + recall + 1e-6)
    idx = np.argmax(f1)
    prec_score = prec[idx]
    f1_score = f1[idx]
    recall_score = recall[idx]

    log_str = 'Precision: {:.3f}'.format(prec_score)
    write_out.append(log_str)
    log_str = 'Recall: {:0.4f}'.format(recall_score)
    write_out.append(log_str)
    log_str = 'F1 Score: {:0.4f}'.format(f1_score)
    write_out.append(log_str)
    write_out = '\n'.join(write_out)
    print(write_out)
    if val_log is not None:
        with open(val_log, "a") as log_file:
            log_file.write(write_out + '\n')

    return metrics