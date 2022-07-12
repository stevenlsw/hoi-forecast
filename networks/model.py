import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, net, lambda_obj=None, lambda_traj=None, lambda_obj_kl=None, lambda_traj_kl=None):
        super(Model, self).__init__()
        self.net = net
        self.lambda_obj = lambda_obj
        self.lambda_obj_kl = lambda_obj_kl
        self.lambda_traj = lambda_traj
        self.lambda_traj_kl = lambda_traj_kl

    def forward(self, feat, bbox_feat, valid_mask, future_hands=None, contact_point=None, future_valid=None,
                num_samples=5, pred_len=4):
        if self.training:
            losses = {}
            total_loss = 0
            traj_loss, traj_kl_loss, obj_loss, obj_kl_loss = self.net(feat, bbox_feat, valid_mask, future_hands,
                                                                      contact_point, future_valid)

            if self.lambda_traj is not None and traj_loss is not None:
                traj_loss = self.lambda_traj * traj_loss.sum()
                total_loss += traj_loss
                losses['traj_loss'] = traj_loss.detach().cpu()
            else:
                losses['traj_loss'] = 0.

            if self.lambda_traj_kl is not None and traj_kl_loss is not None:
                traj_kl_loss = self.lambda_traj_kl * traj_kl_loss.sum()
                total_loss += traj_kl_loss
                losses['traj_kl_loss'] = traj_kl_loss.detach().cpu()
            else:
                losses['traj_kl_loss'] = 0.

            if self.lambda_obj is not None and obj_loss is not None:
                obj_loss = self.lambda_obj * obj_loss.sum()
                total_loss += obj_loss
                losses['obj_loss'] = obj_loss.detach().cpu()
            else:
                losses['obj_loss'] = 0.

            if self.lambda_obj_kl is not None and obj_kl_loss is not None:
                obj_kl_loss = self.lambda_obj_kl * obj_kl_loss.sum()
                total_loss += obj_kl_loss
                losses['obj_kl_loss'] = obj_kl_loss.detach().cpu()
            else:
                losses['obj_kl_loss'] = 0.

            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu()
            else:
                losses["total_loss"] = 0.
            return total_loss, losses
        else:
            future_hands_list = []
            contact_points_list = []
            for i in range(num_samples):
                future_hands, contact_point = self.net.module.inference(feat, bbox_feat, valid_mask,
                                                                        future_valid=future_valid,
                                                                        pred_len=pred_len)
                future_hands_list.append(future_hands)
                contact_points_list.append(contact_point)

            contact_points = torch.stack(contact_points_list, dim=0)

            assert len(contact_points.shape) == 3
            contact_points = contact_points.transpose(0, 1)

            future_hands_list = torch.stack(future_hands_list, dim=0)
            future_hands_list = future_hands_list.transpose(0, 1)
            return future_hands_list, contact_points
