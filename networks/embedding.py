import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class Encoder_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(Encoder_PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        B, T = x.shape[:2]
        if T != self.position_embedding.size(1):
            position_embedding = self.position_embedding.transpose(1, 2)
            new_position_embedding = F.interpolate(position_embedding, size=(T), mode='nearest')
            new_position_embedding = new_position_embedding.transpose(1, 2)
            x = x + new_position_embedding
        else:
            x = x + self.position_embedding
        return x


class Decoder_PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super(Decoder_PositionalEmbedding, self).__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

    def forward(self, x):
        x = x + self.position_embedding[:, :x.shape[1], :]
        return x

