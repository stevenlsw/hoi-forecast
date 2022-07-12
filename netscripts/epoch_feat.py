import time
import numpy as np
import torch

from netscripts.epoch_utils import progress_bar as bar, AverageMeters
from evaluation.traj_eval import evaluate_traj_stochastic
from evaluation.affordance_eval import evaluate_affordance


def epoch_pass(loader, model, epoch, phase, optimizer=None,
               train=True, use_cuda=False,
               num_samples=5, pred_len=4, num_points=5, gaussian_sigma=3., gaussian_k_ratio=3.,
               scheduler=None):
    time_meters = AverageMeters()

    if train:
        print(f"{phase} epoch: {epoch + 1}")
        loss_meters = AverageMeters()
        model.train()
    else:
        print(f"evaluate epoch {epoch}")
        preds_traj, gts_traj, valids_traj = [], [], []
        gts_affordance_dict, preds_affordance_dict = {}, {}
        model.eval()

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    end = time.time()
    for batch_idx, sample in enumerate(loader):
        if train:
            input = sample['feat'].float().to(device)
            bbox_feat = sample['bbox_feat'].float().to(device)
            valid_mask = sample['valid_mask'].float().to(device)
            future_hands = sample['future_hands'].float().to(device)
            contact_point = sample['contact_point'].float().to(device)
            future_valid = sample['future_valid'].float().to(device)
            time_meters.add_loss_value("data_time", time.time() - end)
            model_loss, model_losses = model(input, bbox_feat=bbox_feat,
                                             valid_mask=valid_mask, future_hands=future_hands,
                                             contact_point=contact_point, future_valid=future_valid)

            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

            for key, val in model_losses.items():
                if val is not None:
                    loss_meters.add_loss_value(key, val)

            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s " \
                     "| Hand Traj Loss: {traj_loss:.3f} " \
                     "| Hand Traj KL Loss: {traj_kl_loss:.3f} " \
                     "| Object Affordance Loss: {obj_loss:.3f} " \
                     "| Object Affordance KL Loss: {obj_kl_loss:.3f} " \
                     "| Total Loss: {total_loss:.3f} ".format(batch=batch_idx + 1, size=len(loader),
                                                              data=time_meters.average_meters["data_time"].val,
                                                              bt=time_meters.average_meters["batch_time"].avg,
                                                              traj_loss=loss_meters.average_meters["traj_loss"].avg,
                                                              traj_kl_loss=loss_meters.average_meters[
                                                                  "traj_kl_loss"].avg,
                                                              obj_loss=loss_meters.average_meters["obj_loss"].avg,
                                                              obj_kl_loss=loss_meters.average_meters["obj_kl_loss"].avg,
                                                              total_loss=loss_meters.average_meters[
                                                                  "total_loss"].avg)
            bar(suffix)
            end = time.time()
            if scheduler is not None:
                scheduler.step()
        else:
            input = sample['feat'].float().to(device)
            bbox_feat = sample['bbox_feat'].float().to(device)
            valid_mask = sample['valid_mask'].float().to(device)
            future_valid = sample['future_valid'].float().to(device)

            time_meters.add_loss_value("data_time", time.time() - end)

            with torch.no_grad():
                pred_future_hands, contact_points = model(input, bbox_feat, valid_mask,
                                                          num_samples=num_samples,
                                                          future_valid=future_valid,
                                                          pred_len=pred_len)

            uids = sample['uid'].numpy()
            future_hands = sample['future_hands'][:, :, 1:, :].float().numpy()
            future_valid = sample['future_valid'].float().numpy()

            gts_traj.append(future_hands)
            valids_traj.append(future_valid)

            pred_future_hands = pred_future_hands.cpu().numpy()
            preds_traj.append(pred_future_hands)

            if 'eval' in loader.dataset.partition:
                contact_points = contact_points.cpu().numpy()
                for idx, uid in enumerate(uids):
                    gts_affordance_dict[uid] = loader.dataset.eval_labels[uid]['norm_contacts']
                    preds_affordance_dict[uid] = contact_points[idx]

            time_meters.add_loss_value("batch_time", time.time() - end)

            suffix = "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s" \
                .format(batch=batch_idx + 1, size=len(loader),
                        data=time_meters.average_meters["data_time"].val,
                        bt=time_meters.average_meters["batch_time"].avg)

            bar(suffix)
            end = time.time()

    if train:
        return loss_meters
    else:
        val_info = {}
        if phase == "traj":
            gts_traj = np.concatenate(gts_traj)
            preds_traj = np.concatenate(preds_traj)
            valids_traj = np.concatenate(valids_traj)

            ade, fde = evaluate_traj_stochastic(preds_traj, gts_traj, valids_traj)
            val_info.update({"traj_ade": ade, "traj_fde": fde})

        if 'eval' in loader.dataset.partition and phase == "affordance":
            affordance_metrics = evaluate_affordance(preds_affordance_dict,
                                                     gts_affordance_dict,
                                                     n_pts=num_points,
                                                     gaussian_sigma=gaussian_sigma,
                                                     gaussian_k_ratio=gaussian_k_ratio)
            val_info.update(affordance_metrics)

        return val_info
