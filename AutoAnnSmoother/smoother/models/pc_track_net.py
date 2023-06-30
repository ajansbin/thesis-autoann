import torch
from torch import nn
from torch.nn import functional
from smoother.models.positional_encoder import SinusoidalPositionalEncoding
from smoother.models.temporal_encoder import (
    PoolTempEnc,
    LSTMTempEnc,
    TransformerTempEnc,
    FullTransformerTempEnc,
)
from smoother.models.backbone_encoders import TNet, PCEncoder, FuseEncoder, TrackEncoder


### NETS ###
class PCTrackEarlyFusionNet(nn.Module):
    def __init__(
        self,
        fuse_encoder_name: str,
        encoder_out_size: str,
        temporal_encoder_name: str,
        dec_out_size,
        track_feat_dim,
        pc_feat_dim,
        window_size=180,
    ):
        super(PCTrackEarlyFusionNet, self).__init__()

        feat_dim = pc_feat_dim + track_feat_dim - 2

        self.fuse_encoder = FuseEncoder(input_dim=feat_dim, out_size=encoder_out_size)

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=encoder_out_size, max_len=window_size
        )

        self.temporal_encoder = get_temporal_encoder(
            encoder_name=temporal_encoder_name, in_size=encoder_out_size
        )

        self.heads = DecoderHeads(encoder_out_size, dec_out_size)

    def forward(self, tracks, pcs):
        """
        Inputs: pcs     (B x W x N x 4)
                tracks  (B x W x 8)
        Outputs:
                tensor  (B x 7)
        """
        B, W, N, Fp = pcs.shape

        # Mask out padded
        s = torch.sum(tracks, dim=-1)
        padding_mask = (s == 0).long()  # (B, W)

        tracks_expanded = tracks.unsqueeze(2).expand((-1, -1, N, -1))

        # For now, remove temporal encoding. Add it in temporal encoding instead
        tp = torch.cat((pcs[:, :, :, :3], tracks_expanded[:, :, :, :7]), dim=-1)
        # (B, W, N, 10)

        tp_enc = self.fuse_encoder(tp, padding_mask)  # (B, W, encoder_out_size)

        # Add sinusodial temporal encoding
        tp_enc = self.pos_enc(tp_enc, padding_mask)

        # OBS: Not implemented for full-transformer yet
        tp_temp_enc = self.temporal_encoder(tp_enc, padding_mask)

        center_out, size_out, rotation_out, score_out = self.heads(tp_temp_enc)

        return center_out, size_out, rotation_out, score_out


class PCTrackNet(nn.Module):
    def __init__(
        self,
        track_encoder_name: str,
        pc_encoder_name: str,
        temporal_encoder_name: str,
        track_feat_dim=8,
        pc_feat_dim=3,
        track_out=64,
        pc_out=256,
        dec_out_size=16,
        window_size=180,
    ):
        super(PCTrackNet, self).__init__()

        self.pc_encoder = PCEncoder(
            input_dim=pc_feat_dim - 1, out_size=pc_out, dropout_rate=0.0
        )

        self.track_encoder = TrackEncoder(
            input_dim=track_feat_dim - 1, out_size=track_out, dropout_rate=0.0
        )

        encoder_out_size = track_out + pc_out

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=encoder_out_size, max_len=window_size
        )

        self.temporal_encoder = get_temporal_encoder(
            encoder_name=temporal_encoder_name, in_size=encoder_out_size
        )

        self.heads = DecoderHeads(encoder_out_size, dec_out_size)

    def forward(self, tracks, pcs):
        """
        Inputs: pcs     (B x W x N x 4)
                tracks  (B x W x 8)
        Outputs:
                tensor  (B x 7)
        """

        # Mask out padded
        s = torch.sum(tracks, dim=-1)
        padding_mask = (s == 0).long()  # (B, W)

        # out shape: (B x W x pc_out)
        # Remove temporal encoding
        pcs_enc = self.pc_encoder(pcs[:, :, :, :-1], padding_mask)

        # out shape: (B x W x track_out)
        # Remove temporal encoding
        tracks_enc = self.track_encoder(tracks[:, :, :-1], padding_mask)

        # out shape: (B x W x pc_out+track_out)
        tp_enc = torch.cat((pcs_enc, tracks_enc), dim=-1)

        # Add sinusodial temporal encoding
        tp_enc = self.pos_enc(tp_enc, padding_mask)

        # OBS: Not implemented for full-transformer yet
        tp_temp_enc = self.temporal_encoder(tp_enc, padding_mask)

        center_out, size_out, rotation_out, score_out = self.heads(tp_temp_enc)

        return center_out, size_out, rotation_out, score_out


class PCNet(nn.Module):
    def __init__(
        self,
        track_encoder_name: str,
        pc_encoder_name: str,
        temporal_encoder_name: str,
        track_feat_dim=8,
        pc_feat_dim=3,
        track_out=64,
        pc_out=256,
        dec_out_size=16,
        window_size=180,
    ):
        super(PCNet, self).__init__()

        self.pc_encoder = PCEncoder(
            input_dim=pc_feat_dim - 1, out_size=pc_out, dropout_rate=0.0
        )

        encoder_out_size = pc_out

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=encoder_out_size, max_len=window_size
        )

        self.temporal_encoder = get_temporal_encoder(
            encoder_name=temporal_encoder_name, in_size=encoder_out_size
        )

        self.heads = DecoderHeads(encoder_out_size, dec_out_size)

    def forward(self, tracks, pcs):
        """
        Inputs: pcs     (B x W x N x 4)
                tracks  (B x W x 8)
        Outputs:
                tensor  (B x 7)
        """

        # Mask out padded
        s = torch.sum(tracks, dim=-1)
        padding_mask = (s == 0).long()  # (B, W)

        # out shape: (B x W x pc_out)
        # Remove temporal encoding
        pcs_enc = self.pc_encoder(pcs[:, :, :, :-1], padding_mask)

        # Add sinusodial temporal encoding
        pcs_enc = self.pos_enc(pcs_enc, padding_mask)

        # OBS: Not implemented for full-transformer yet
        pcs_temp_enc = self.temporal_encoder(pcs_enc, padding_mask)

        center_out, size_out, rotation_out, score_out = self.heads(pcs_temp_enc)

        return center_out, size_out, rotation_out, score_out


class TrackNet(nn.Module):
    def __init__(
        self,
        track_encoder_name: str,
        pc_encoder_name: str,
        temporal_encoder_name: str,
        track_feat_dim=8,
        pc_feat_dim=3,
        track_out=64,
        pc_out=256,
        dec_out_size=16,
        window_size=180,
    ):
        super(TrackNet, self).__init__()

        self.track_encoder = TrackEncoder(
            input_dim=track_feat_dim - 1, out_size=track_out, dropout_rate=0.0
        )

        encoder_out_size = track_out

        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=encoder_out_size, max_len=window_size
        )

        self.temporal_encoder = get_temporal_encoder(
            encoder_name=temporal_encoder_name, in_size=encoder_out_size
        )

        self.heads = DecoderHeads(encoder_out_size, dec_out_size)

    def forward(self, tracks, pcs):
        """
        Inputs: pcs     (B x W x N x 4)
                tracks  (B x W x 8)
        Outputs:
                tensor  (B x 7)
        """

        # Mask out padded
        s = torch.sum(tracks, dim=-1)
        padding_mask = (s == 0).long()  # (B, W)

        # out shape: (B x W x track_out)
        # Remove temporal encoding
        tracks_enc = self.track_encoder(tracks[:, :, :-1], padding_mask)

        # Add sinusodial temporal encoding
        tracks_enc = self.pos_enc(tracks_enc, padding_mask)

        # OBS: Not implemented for full-transformer yet
        tracks_temp_enc = self.temporal_encoder(tracks_enc, padding_mask)

        center_out, size_out, rotation_out, score_out = self.heads(tracks_temp_enc)

        return center_out, size_out, rotation_out, score_out


### DECODERS ###
class DecoderHeads(nn.Module):
    def __init__(self, in_size, out_size):
        super(DecoderHeads, self).__init__()
        self.fc_center = nn.Sequential(
            nn.Linear(in_size, out_size), nn.ReLU(), nn.Linear(out_size, 3)
        )

        self.fc_size = nn.Sequential(
            nn.Linear(in_size, out_size), nn.ReLU(), nn.Linear(out_size, 3)
        )
        self.fc_rotation = nn.Sequential(
            nn.Linear(in_size, out_size), nn.ReLU(), nn.Linear(out_size, 1)
        )
        self.fc_gt = nn.Sequential(
            nn.Linear(in_size, out_size), nn.ReLU(), nn.Linear(out_size, 1)
        )

    def forward(self, x):
        """
        Inputs: x (B,W,F)
        """

        # Per frame center/rotation/score
        center_out = self.fc_center(x)  # (B, W, 3)
        rotation_out = self.fc_rotation(x)  # (B, W, 1)
        score_out = torch.sigmoid(self.fc_gt(x))  # (B, W, 1)
        # All frame max pool size
        x_pooled = torch.max(x, 1)[0]  # (B, F)
        size_out = self.fc_size(x_pooled)  # (B, 3)

        return center_out, size_out, rotation_out, score_out


### UTILS ###
def get_encoder(
    encoder_name: str, data_type: str, in_size: int, out_size: int
) -> nn.Module:
    if encoder_name == "pointnet":
        if data_type == "pc":
            return PCEncoder(in_size, out_size)
        elif data_type == "track":
            return TrackEncoder(in_size, out_size)
    raise NotImplementedError(
        f"Encoder of type {encoder_name} with data of type {data_type} is not implemented."
    )


def get_temporal_encoder(encoder_name: str, in_size):
    if encoder_name == "pool":
        return PoolTempEnc(in_size)
    if encoder_name == "lstm":
        return LSTMTempEnc(in_size)
    if encoder_name == "transformer":
        return TransformerTempEnc(in_size)
    if encoder_name == "full-transformer":
        return FullTransformerTempEnc(in_size)
    raise NotImplementedError(
        f"Temporal encoder of type {encoder_name} is not implemented."
    )
