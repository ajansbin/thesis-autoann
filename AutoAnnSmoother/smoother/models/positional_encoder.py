import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=180):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model  # feature dimension after first encoding
        self.max_len = max_len  # window_length

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,F)
                padding_mask    (B,W)
        output: x               (B,W,F)
        """

        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]

        padding_mask = padding_mask.unsqueeze(-1)
        pe = pe * (1 - padding_mask)

        x = x + pe
        return x
