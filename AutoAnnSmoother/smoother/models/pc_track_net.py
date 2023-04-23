import torch
from torch import nn
from torch.nn import functional

### NETS ###

class PCTrackNet(nn.Module):

    def __init__(self, track_encoder:str, pc_encoder:str, decoder:str, pc_feat_dim = 3, track_feat_dim=9, pc_out=256, track_out=64, dec_out=16):
        super(PCTrackNet, self).__init__()

        self.pc_encoder = get_encoder(encoder_name=pc_encoder, 
                                      data_type='pc', 
                                      in_size=pc_feat_dim, 
                                      out_size=pc_out)
        
        self.track_encoder = get_encoder(encoder_name=track_encoder, 
                                      data_type='track', 
                                      in_size=track_feat_dim, 
                                      out_size=track_out)

        dec_in_size = pc_out+track_out
        self.temporal_decoder = get_decoder(decoder_name=decoder,
                                            in_size=dec_in_size,
                                            out_size=dec_out)

    def forward(self, tracks, pcs):
        '''
        Inputs: pcs     (B x W x N x 3)
                tracks  (B x W x 9)
        Outputs:
                tensor  (B x 8)
        '''

        pcs_enc = self.pc_encoder(pcs) # Shape: (B x W x pc_out)
        tracks_enc = self.track_encoder(tracks) # Shape: (B x W x track_out)

        x = torch.cat((pcs_enc, tracks_enc), dim=-1)  # Shape: (B x W x pc_out+track_out)

        x_dec = self.temporal_decoder(x) # Shape (B x 8)

        return x_dec

class PCNet(nn.Module):

    def __init__(self, track_encoder:str, pc_encoder:str, decoder:str, pc_feat_dim = 3, track_feat_dim=9, pc_out=256, track_out=64, dec_out=16):
        super(PCNet, self).__init__()

        self.pc_encoder = get_encoder(encoder_name=pc_encoder, 
                                      data_type='pc', 
                                      in_size=pc_feat_dim, 
                                      out_size=pc_out)
        

        dec_in_size = pc_out
        self.temporal_decoder = get_decoder(decoder_name=decoder,
                                            in_size=dec_in_size,
                                            out_size=dec_out)
    def forward(self, tracks, pcs):
        '''
        Inputs:     pcs  (B x W x N x 9)
        
        Outputs:    tensor  (B x 8)
        '''

        pcs_enc = self.pc_encoder(pcs) # Shape: (B x W x pc_out)

        pcs_dec = self.temporal_decoder(pcs_enc) # Shape (B x 8)

        return pcs_dec

class TrackNet(nn.Module):

    def __init__(self, track_encoder:str, pc_encoder:str, decoder:str, pc_feat_dim = 3, track_feat_dim=9, pc_out=256, track_out=64, dec_out=16):
        super(TrackNet, self).__init__()

        self.track_encoder = get_encoder(encoder_name=track_encoder, 
                                         data_type='track', 
                                         in_size=track_feat_dim, 
                                         out_size=track_out)

        dec_in_size = track_out
        self.temporal_decoder = get_decoder(decoder_name=decoder,
                                            in_size=dec_in_size,
                                            out_size=dec_out)

    def forward(self, tracks, pcs):
        '''
        Inputs:     tracks  (B x W x 9)

        Outputs:    tensor  (B x 8)
        '''

        tracks_enc = self.track_encoder(tracks) # Shape: (B x W x track_out)

        tracks_dec = self.temporal_decoder(tracks_enc) # Shape (B x 8)

        return tracks_dec   
    

### ENCODERS ###

class TNet(nn.Module):
    def __init__(self, k, out_size=256):
        super(TNet, self).__init__()
        self.k = k
        self.out_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, out_size),
            nn.ReLU(),
        )
        self.fc = nn.Linear(out_size, k * k)
        self.register_buffer('identity', torch.eye(k))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k) + self.identity
        return x


class PCEncoder(nn.Module):
    def __init__(self, input_dim=4, out_size=256):
        super(PCEncoder, self).__init__()
        self.tnet4 = TNet(input_dim, out_size)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, out_size, 1)

    def forward(self, x):
        B, W, N, F = x.shape
        x = x.view(B * W, N, F) # B*W, N,F
        tnet4 = self.tnet4(x)
        x = torch.bmm(x, tnet4).transpose(1,2)
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)
        return x

class TrackEncoder(nn.Module):
    def __init__(self, input_dim=9, out_size=64):
        super(TrackEncoder, self).__init__()
        self.tnet9 = TNet(input_dim, out_size)
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 32, 1)
        self.conv3 = nn.Conv1d(32, 64, 1)
        self.conv4 = nn.Conv1d(64, out_size, 1)

    def forward(self, x):
        B, W, F = x.shape
        x = x.view(B * W, 1, F)
        tnet9 = self.tnet9(x)
        x = torch.bmm(x, tnet9).transpose(1, 2)
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)
        return x
    
### DECODERS ###
    
class DecoderHeads(nn.Module):
    def __init__(self, in_size, out_size):
        super(DecoderHeads, self).__init__()
        self.fc_center = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 3)
        )

        self.fc_size = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 3)
        )
        self.fc_rotation = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 2)
        )
        self.fc_gt = nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, 1)
        )

    def forward(self, x):
        '''
        Inputs: x (B,W,F)
        '''

        # Per frame center/rotation/score
        center_out = self.fc_center(x)              # (B, W, 3)
        rotation_out = self.fc_rotation(x)          # (B, W, 2)
        score_out = torch.sigmoid(self.fc_gt(x))    # (B, W, 1)

        # All frame max pool size
        x_pooled = torch.max(x, 1)[0] # (B, F)
        size_out = self.fc_size(x_pooled) # (B, 3)

        return center_out, size_out, rotation_out, score_out

class PoolDecoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(PoolDecoder, self).__init__()
        mlp_hidden_size = 128
        self.mlp = nn.Sequential(
            nn.Linear(in_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
        )
        self.heads = DecoderHeads(mlp_hidden_size, out_size)

    def forward(self, x):
        x = self.mlp(x)  # Pass the input through the MLP: (B, W, F)
        center_out, size_out, rotation_out, score_out = self.heads(x)
        return center_out, size_out, rotation_out, score_out
        
class LSTMDecoder(nn.Module):
    def __init__(self, input_size, output_head_size):
        super(LSTMDecoder, self).__init__()
        lstm_hidden_size = 32
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)
        self.heads = DecoderHeads(lstm_hidden_size, output_head_size)

    def forward(self, x):
        x, _ = self.lstm(x)  # Shape: (B, W, lstm_hidden_size)
        center_out, size_out, rotation_out, score_out = self.heads(x)
        return center_out, size_out, rotation_out, score_out

class TransformerDecoder(nn.Module):
    def __init__(self, in_size, out_size):
        super(TransformerDecoder, self).__init__()

        t_encoder_layer = nn.TransformerEncoderLayer(d_model=in_size, nhead=8, dim_feedforward=in_size*4)
        self.transformer = nn.TransformerEncoder(encoder_layer=t_encoder_layer, num_layers=2)
        self.heads = DecoderHeads(in_size, out_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # Transformer requires input shape: (W, B, F)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        center_out, size_out, rotation_out, score_out = self.heads(x)
        return center_out, size_out, rotation_out, score_out


### UTILS ###
def get_encoder(encoder_name:str, data_type:str, in_size:int, out_size:int) -> nn.Module:
    if encoder_name == 'pointnet':
        if data_type == 'pc':
            return PCEncoder(in_size, out_size)
        elif data_type == 'track':
            return TrackEncoder(in_size, out_size)
    raise NotImplementedError(f"Encoder of type {encoder_name} with data of type {data_type} is not implemented.")

def get_decoder(decoder_name:str, in_size:int, out_size:int) -> nn.Module:
    if decoder_name == 'pool':
        return PoolDecoder(in_size, out_size)
    elif decoder_name.lower() == 'lstm':
        return LSTMDecoder(in_size, out_size)
    elif decoder_name == 'transformer':
        return TransformerDecoder(in_size, out_size)
    raise NotImplementedError(f"Decoder of type {decoder_name} is not implemented.")
