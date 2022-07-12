import torch
import torch.nn as nn
from networks.decoder_modules import VAE


class AffordanceCVAE(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 pred_len=4, condition_traj=True, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_traj = condition_traj
        self.z_scale = z_scale
        if self.condition_traj:
            if coord_dim is None:
                coord_dim = hidden_dim // 2
            self.coord_dim = coord_dim
            self.traj_to_feature = nn.Sequential(
                nn.Linear(2*(pred_len+1), coord_dim*(pred_len+1), bias=False),
                nn.ELU(inplace=True))
            self.traj_context_fusion = nn.Sequential(
                nn.Linear(condition_dim+coord_dim*(pred_len+1), condition_dim, bias=False),
                nn.ELU(inplace=True))

        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim)

    def forward(self, context, contact_point, hand_traj=None, return_pred=False):
        if self.condition_traj:
            assert hand_traj is not None
            batch_size = context.shape[0]
            hand_traj = hand_traj.reshape(batch_size, -1)
            traj_feat = self.traj_to_feature(hand_traj)
            fusion_feat = torch.cat([context, traj_feat], dim=1)
            condition_context = self.traj_context_fusion(fusion_feat)
        else:
            condition_context = context
        if not return_pred:
            recon_loss, KLD = self.cvae(contact_point, c=condition_context)
            return recon_loss, KLD
        else:
            pred_contact, recon_loss, KLD = self.cvae(contact_point, c=condition_context, return_pred=return_pred)
            return pred_contact, recon_loss, KLD

    def inference(self, context, hand_traj=None):
        if self.condition_traj:
            assert hand_traj is not None
            batch_size = context.shape[0]
            hand_traj = hand_traj.reshape(batch_size, -1)
            traj_feat = self.traj_to_feature(hand_traj)
            fusion_feat = torch.cat([context, traj_feat], dim=1)
            condition_context = self.traj_context_fusion(fusion_feat)
        else:
            condition_context = context
        z = self.z_scale * torch.randn([condition_context.shape[0], self.latent_dim], device=condition_context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x