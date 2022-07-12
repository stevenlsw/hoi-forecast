import torch
import torch.nn as nn
from einops import rearrange
from networks.embedding import PositionalEncoding, Encoder_PositionalEmbedding, Decoder_PositionalEmbedding
from networks.layer import EncoderBlock, DecoderBlock
from networks.net_utils import trunc_normal_, get_pad_mask, get_subsequent_mask, traj_affordance_dist


class Encoder(nn.Module):
    def __init__(self, num_patches=5, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 dropout=0., time_embed_type=None, num_frames=None):
        super().__init__()
        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        self.num_patches = num_patches  # (hand, object global feature patches, default: 5)
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Encoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, x, mask=None):
        B, T, N = x.shape[:3]

        x = rearrange(x, 'b t n m -> (b t) n m', b=B, t=T, n=N)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = self.time_embed(x)
        x = self.time_drop(x)
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        mask = mask.transpose(1, 2)
        for blk in self.encoder_blocks:
            x = blk(x, B, T, N, mask=mask)

        x = rearrange(x, 'b (n t) m -> b t n m', b=B, t=T, n=N)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_features, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 time_embed_type=None, num_frames=None):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_features = self.embed_dim = embed_dim

        self.trg_embedding = nn.Linear(in_features, embed_dim)

        if time_embed_type is None or num_frames is None:
            time_embed_type = 'sin'
        self.time_embed_type = time_embed_type
        if not self.time_embed_type == "sin" and num_frames is not None:
            self.time_embed = Decoder_PositionalEmbedding(embed_dim, seq_len=num_frames)
        else:
            self.time_embed = PositionalEncoding(embed_dim)
        self.time_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.decoder_blocks = nn.ModuleList([DecoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'time_embed'}

    def forward(self, trg, memory, memory_mask=None, trg_mask=None):
        trg = self.trg_embedding(trg)
        trg = self.time_embed(trg)
        trg = self.time_drop(trg)

        for blk in self.decoder_blocks:
            trg = blk(trg, memory, memory_mask=memory_mask, trg_mask=trg_mask)

        trg = self.norm(trg)
        return trg


class ObjectTransformer(nn.Module):

    def __init__(self, src_in_features, trg_in_features, num_patches,
                 hand_head, obj_head,
                 embed_dim=512, coord_dim=64, num_heads=8, enc_depth=6, dec_depth=4,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.,
                 encoder_time_embed_type='sin', decoder_time_embed_type='sin',
                 num_frames_input=None, num_frames_output=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.coord_dim = coord_dim
        self.downproject = nn.Linear(src_in_features, embed_dim)

        self.bbox_to_feature = nn.Sequential(
            nn.Linear(4, self.coord_dim // 2),
            nn.ELU(inplace=True),
            nn.Linear(self.coord_dim // 2, self.coord_dim),
            nn.ELU()
        )
        self.feat_fusion = nn.Sequential(
            nn.Linear(self.embed_dim + self.coord_dim, self.embed_dim),
            nn.ELU(inplace=True))

        self.encoder = Encoder(num_patches=num_patches,
                               embed_dim=embed_dim, depth=enc_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                               time_embed_type=encoder_time_embed_type, num_frames=num_frames_input)

        self.decoder = Decoder(in_features=trg_in_features, embed_dim=embed_dim,
                               depth=dec_depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                               attn_drop_rate=attn_drop_rate,
                               drop_path_rate=drop_path_rate, norm_layer=norm_layer, dropout=dropout,
                               time_embed_type=decoder_time_embed_type, num_frames=num_frames_output)

        self.hand_head = hand_head
        self.object_head = obj_head
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def encoder_input(self, feat, bbox_feat):
        B, T = feat.shape[0], feat.shape[2]
        feat = self.downproject(feat)
        bbox_feat = bbox_feat.view(-1, 4)
        bbox_feat = self.bbox_to_feature(bbox_feat)
        bbox_feat = bbox_feat.view(B, -1, T, self.coord_dim)
        ho_feat = feat[:, 1:, :, :]
        global_feat = feat[:, 0:1, :, :]
        feat = torch.cat((ho_feat, bbox_feat), dim=-1)
        feat = feat.view(-1, self.embed_dim + self.coord_dim)
        feat = self.feat_fusion(feat)
        feat = feat.view(B, -1, T, self.embed_dim)
        feat = torch.cat((global_feat, feat), dim=1)
        feat = feat.transpose(1, 2)
        return feat

    def forward(self, feat, bbox_feat, valid_mask, future_hands, contact_point, future_valid):
        # feat: (B, 5, T, src_in_features), global, hand & obj, T=10
        # bbox_feat: (B, 4, T, 4), hand & obj
        # valid_mask: (B, 4, T), hand & obj / (B, 5, T), hand & obj & global
        # future_hands: (B, 2, T, 2) right & left, T=5 (contain last observation frame)
        # contact_points: (B, 2)
        # future_valid: (B, 2), right & left traj valid
        # return: traj_loss: (B), obj_loss (B)
        if not valid_mask.shape[1] == feat.shape[1]:
            src_mask = torch.cat(
                (torch.ones_like(valid_mask[:, 0:1, :], dtype=valid_mask.dtype, device=valid_mask.device),
                 valid_mask), dim=1).transpose(1, 2)
        else:
            src_mask = valid_mask.transpose(1, 2)
        feat = self.encoder_input(feat, bbox_feat)
        x = self.encoder(feat, mask=src_mask)

        memory = x[:, -1, :, :]
        memory_mask = get_pad_mask(src_mask[:, -1, :], pad_idx=0)

        future_rhand, future_lhand = future_hands[:, 0, :, :], future_hands[:, 1, :, :]

        rhand_input = future_rhand[:, :-1, :]
        lhand_input = future_lhand[:, :-1, :]
        trg_mask = torch.ones_like(rhand_input[:, :, 0])
        trg_mask = get_subsequent_mask(trg_mask)

        x_rhand = self.decoder(rhand_input, memory,
                               memory_mask=memory_mask, trg_mask=trg_mask)
        x_lhand = self.decoder(lhand_input, memory,
                               memory_mask=memory_mask, trg_mask=trg_mask)

        x_hand = torch.cat((x_rhand, x_lhand), dim=1)
        x_hand = x_hand.reshape(-1, self.embed_dim)

        target_hand = future_hands[:, :, 1:, :]
        target_hand = target_hand.reshape(-1, target_hand.shape[-1])

        pred_hand, traj_loss, traj_kl_loss = self.hand_head(x_hand, target_hand, future_valid, contact=None,
                                                            return_pred=True)

        r_pred_contact, r_obj_loss, r_obj_kl_loss = self.object_head(memory[:, 0, :], contact_point, future_rhand,
                                                                     return_pred=True)
        l_pred_contact, l_obj_loss, l_obj_kl_loss = self.object_head(memory[:, 0, :], contact_point, future_lhand,
                                                                     return_pred=True)

        obj_loss = torch.stack([r_obj_loss, l_obj_loss], dim=1)
        obj_kl_loss = torch.stack([r_obj_kl_loss, l_obj_kl_loss], dim=1)

        obj_loss[~(future_valid > 0)] = 1e9
        selected_obj_loss, selected_idx = obj_loss.min(dim=1)
        selected_valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_obj_kl_loss = torch.gather(obj_kl_loss, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        obj_loss = selected_obj_loss * selected_valid
        obj_kl_loss = selected_obj_kl_loss * selected_valid

        return traj_loss, traj_kl_loss, obj_loss, obj_kl_loss

    def inference(self, feat, bbox_feat, valid_mask, future_valid=None, pred_len=4):
        # feat: (B, 5, T, src_in_features), hand & obj, T=10
        # bbox_feat: (B, 4, T, 4), hand & obj
        # valid_mask: (B, 4, T), hand & obj
        # future_valid: (B, 2), right & left traj valid
        # return: future_hand (B, 2, T, 2), not include last observation frame
        # return: contact_point (B, 2)
        B, T = feat.shape[0], feat.shape[2]
        if not valid_mask.shape[1] == feat.shape[1]:
            src_mask = torch.cat(
                (torch.ones_like(valid_mask[:, 0:1, :], dtype=valid_mask.dtype, device=valid_mask.device),
                 valid_mask), dim=1).transpose(1, 2)
        else:
            src_mask = valid_mask.transpose(1, 2)
        feat = self.encoder_input(feat, bbox_feat)
        x = self.encoder(feat, mask=src_mask)

        memory = x[:, -1, :, :]
        memory_mask = get_pad_mask(src_mask[:, -1, :], pad_idx=0)

        observe_bbox = bbox_feat[:, :2, -1, :]
        observe_rhand, observe_lhand = observe_bbox[:, 0, :], observe_bbox[:, 1, :]
        future_rhand = (observe_rhand[:, :2] + observe_rhand[:, 2:]) / 2
        future_lhand = (observe_lhand[:, :2] + observe_lhand[:, 2:]) / 2
        future_rhand = future_rhand.unsqueeze(dim=1)
        future_lhand = future_lhand.unsqueeze(dim=1)

        pred_contact = None
        for i in range(pred_len):
            trg_mask = torch.ones_like(future_rhand[:, :, 0])
            trg_mask = get_subsequent_mask(trg_mask)

            x_rhand = self.decoder(future_rhand, memory,
                                   memory_mask=memory_mask, trg_mask=trg_mask)
            x_hand = x_rhand.reshape(-1, self.embed_dim)
            loc_rhand = self.hand_head.inference(x_hand, pred_contact)
            loc_rhand = loc_rhand.reshape(B, -1, 2)
            pred_rhand = loc_rhand[:, -1:, :]
            future_rhand = torch.cat((future_rhand, pred_rhand), dim=1)

        for i in range(pred_len):
            trg_mask = torch.ones_like(future_lhand[:, :, 0])
            trg_mask = get_subsequent_mask(trg_mask)

            x_lhand = self.decoder(future_lhand, memory,
                                   memory_mask=memory_mask, trg_mask=trg_mask)
            x_hand = x_lhand.reshape(-1, self.embed_dim)
            loc_lhand = self.hand_head.inference(x_hand, pred_contact)
            loc_lhand = loc_lhand.reshape(B, -1, 2)
            pred_lhand = loc_lhand[:, -1:, :]
            future_lhand = torch.cat((future_lhand, pred_lhand), dim=1)

        future_hands = torch.stack((future_rhand[:, 1:, :], future_lhand[:, 1:, :]), dim=1)

        r_pred_contact = self.object_head.inference(memory[:, 0, :], future_rhand)
        l_pred_contact = self.object_head.inference(memory[:, 0, :], future_lhand)
        pred_contact = torch.stack([r_pred_contact, l_pred_contact], dim=1)

        if future_valid is not None and torch.all(future_valid.sum(dim=1) >= 1):
            r_pred_contact_dist = traj_affordance_dist(future_hands.reshape(-1, 2), r_pred_contact,
                                                       future_valid)
            l_pred_contact_dist = traj_affordance_dist(future_hands.reshape(-1, 2), l_pred_contact,
                                                       future_valid)
            pred_contact_dist = torch.stack((r_pred_contact_dist, l_pred_contact_dist), dim=1)
            _, selected_idx = pred_contact_dist.min(dim=1)
            selected_idx = selected_idx.unsqueeze(dim=1).unsqueeze(dim=2).expand(pred_contact.shape[0], 1,
                                                                                 pred_contact.shape[2])
            pred_contact = torch.gather(pred_contact, dim=1, index=selected_idx).squeeze(dim=1)

        return future_hands, pred_contact
