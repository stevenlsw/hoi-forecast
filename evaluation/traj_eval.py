import torch
import numpy as np


def compute_ade(pred_traj, gt_traj, valid_traj=None, reduction=True):
    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 1] >= 0)  \
                 & (gt_traj[:, :, :, 0] < 1) & (gt_traj[:, :, :, 1] < 1)

    error = gt_traj - pred_traj
    error = error * valid_loc[:, :, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        ade = torch.sqrt(error.sum(dim=3)).mean(dim=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=3)
        ade = error.mean(axis=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return ade, valid_traj


def compute_fde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    pred_last = pred_traj[:, :, -1, :]
    gt_last = gt_traj[:, :, -1, :]

    valid_loc = (gt_last[:, :, 0] >= 0) & (gt_last[:, :, 1] >= 0) \
                & (gt_last[:, :, 0] < 1) & (gt_last[:, :, 1] < 1)

    error = gt_last - pred_last
    error = error * valid_loc[:, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        fde = torch.sqrt(error.sum(dim=2)) * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=2)
        fde = error * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return fde, valid_traj


def evaluate_traj_stochastic(preds, gts, valids):
    len_dataset, num_samples, num_obj = preds.shape[0], preds.shape[1], preds.shape[2]
    ade_list, fde_list = [], []
    for idx in range(num_samples):
        ade, _ = compute_fde(preds[:, idx, :, :, :], gts, valids, reduction=False)
        ade_list.append(ade)
        fde, _ = compute_ade(preds[:, idx, :, :, :], gts, valids, reduction=False)
        fde_list.append(fde)

    if torch.is_tensor(preds):
        ade_list = torch.stack(ade_list, dim=0)
        fde_list = torch.stack(fde_list, dim=0)

        ade_err_min, _ = torch.min(ade_list, dim=0)
        ade_err_min = ade_err_min * valids
        fde_err_min, _ = torch.min(fde_list, dim=0)
        fde_err_min = fde_err_min * valids

        ade_err_mean = torch.mean(ade_list, dim=0)
        ade_err_mean = ade_err_mean * valids
        fde_err_mean = torch.mean(fde_list, dim=0)
        fde_err_mean = fde_err_mean * valids

        ade_err_std = torch.std(ade_list, dim=0) * np.sqrt((ade_list.shape[0] - 1.) / ade_list.shape[0])
        ade_err_std = ade_err_std * valids
        fde_err_std = torch.std(fde_list, dim=0) * np.sqrt((fde_list.shape[0] - 1.) / fde_list.shape[0])
        fde_err_std = fde_err_std * valids

    else:
        ade_list = np.array(ade_list, dtype=np.float32)
        fde_list = np.array(fde_list, dtype=np.float32)

        ade_err_min = ade_list.min(axis=0) * valids
        fde_err_min = fde_list.min(axis=0) * valids

        ade_err_mean = ade_list.mean(axis=0) * valids
        fde_err_mean = fde_list.mean(axis=0) * valids

        ade_err_std = ade_list.std(axis=0) * valids
        fde_err_std = fde_list.std(axis=0) * valids

    ade_mean = ade_err_mean.sum() / valids.sum()
    fde_mean = fde_err_mean.sum() / valids.sum()

    ade_std = ade_err_std.sum() / valids.sum()
    fde_std = fde_err_std.sum() / valids.sum()
    ade_mean_info = 'ADE: %.3f ± %.3f (%d/%d)' % (ade_mean, ade_std, valids.sum(), len_dataset * num_obj)
    fde_mean_info = "FDE: %.3f ± %.3f (%d/%d)" % (fde_mean, fde_std, valids.sum(), len_dataset * num_obj)

    ade_min = ade_err_min.sum() / valids.sum()
    fde_min = fde_err_min.sum() / valids.sum()
    ade_min_info = 'min ADE: %.3f (%d/%d)' % (ade_min, valids.sum(), len_dataset * num_obj)
    fde_min_info = "min FDE: %.3f (%d/%d)" % (fde_min, valids.sum(), len_dataset * num_obj)

    print(ade_min_info)
    print(fde_min_info)
    print(ade_mean_info)
    print(fde_mean_info)

    return ade_mean, fde_mean




