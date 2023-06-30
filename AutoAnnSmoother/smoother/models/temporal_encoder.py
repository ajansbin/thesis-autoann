import torch
from torch import nn
from torch.nn import functional as F


### TEMPORAL ENCODER ###


class PoolTempEnc(nn.Module):
    def __init__(self, input_size):
        super(PoolTempEnc, self).__init__()

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,F)
                padding_mask    (B,W)
        output: x               (B,W,F)
        """

        x = x.masked_fill_(padding_mask.bool().unsqueeze(-1), float("-inf"))
        x_pooled, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, F)
        x_expanded = x_pooled.expand((-1, x.size(1), -1))  # (B, W, F)

        return x_expanded


class LSTMTempEnc(nn.Module):
    def __init__(self, input_size):
        super(LSTMTempEnc, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,F)
                padding_mask    (B,W)
        output: x               (B,W,F)
        """

        x, _ = self.lstm(x)  # Shape: (B, W, lstm_hidden_size)

        return x


class TransformerTempEnc(nn.Module):
    def __init__(self, input_size):
        super(TransformerTempEnc, self).__init__()

        t_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=4,
            dim_feedforward=input_size * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=t_encoder_layer, num_layers=1
        )

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,F)
                padding_mask    (B,W)
        output: x               (B,W,F)
        """
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        return x


class FullTransformerTempEnc(nn.Module):
    def __init__(self, input_size):
        super(FullTransformerTempEnc, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=input_size * 4,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self, x_enc, x_dec, src_key_padding_mask=None, tgt_key_padding_mask=None
    ):
        """
        input:  x_enc                   (B,W,F_enc)
                x_dec                   (B,W,F_dec)
                src_key_padding_mask    (B,W)
                tgt_key_padding_mask    (B,W)
        output: x               (B,W,F)
        """
        x = self.transformer(
            src=x_enc,
            tgt=x_dec,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return x
