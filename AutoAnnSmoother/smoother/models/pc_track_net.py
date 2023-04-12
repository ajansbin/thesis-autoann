import torch
from torch import nn
from torch.nn import functional

class PCTrackNet(nn.Module):

    def __init__(self, pc_feat_dim = 3, track_feat_dim=9):
        super(PCTrackNet, self).__init__()
        out_size = 1024

        self.pc_encoder = PCEncoder(pc_feat_dim, out_size)
        self.track_encoder = TrackEncoder(track_feat_dim, out_size)

        self.temporal_decoder = TemporalDecoder(out_size*2, out_size, 32)

    def forward(self, tracks, pcs):
        '''
        Inputs: pcs     (B x W x N x 3)
                tracks  (B x W x 9)
        Outputs:
                tensor  (B x 8)
        '''

        pcs_enc = self.pc_encoder(pcs) # Shape: (B x W x 1024)
        tracks_enc = self.track_encoder(tracks) # Shape: (B x W x 1024)

        x = torch.cat((pcs_enc, tracks_enc), dim=-1)  # Shape: (B x W x 2048)

        x_dec = self.temporal_decoder(x) # Shape (B x 8)

        return x_dec

class PCNet(nn.Module):

    def __init__(self, pc_feat_dim = 3):
        super(PCNet, self).__init__()
        out_size = 1024

        self.pc_encoder = PCEncoder(pc_feat_dim, out_size)

        self.temporal_decoder = TemporalDecoder(out_size, out_size, 32)

    def forward(self, tracks, pcs):
        '''
        Inputs:     pcs  (B x W x N x 9)
        
        Outputs:    tensor  (B x 8)
        '''

        pcs_enc = self.pc_encoder(pcs) # Shape: (B x W x 1024)

        pcs_dec = self.temporal_decoder(pcs_enc) # Shape (B x 8)

        return pcs_dec

class TrackNet(nn.Module):

    def __init__(self, track_feat_dim=9):
        super(TrackNet, self).__init__()
        out_size = 1024

        self.track_encoder = TrackEncoder(track_feat_dim, out_size)

        self.temporal_decoder = TemporalDecoder(out_size, out_size, 32)

    def forward(self, tracks, pcs):
        '''
        Inputs:     tracks  (B x W x 9)

        Outputs:    tensor  (B x 8)
        '''

        tracks_enc = self.track_encoder(tracks) # Shape: (B x W x 1024)

        tracks_dec = self.temporal_decoder(tracks_enc) # Shape (B x 8)

        return tracks_dec   
    

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(k, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU(),
        )
        self.fc = nn.Linear(1024, k * k)
        self.register_buffer('identity', torch.eye(k))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k) + self.identity
        return x

class PCEncoder(nn.Module):
    def __init__(self, input_dim=3, out_size=1024):
        super(PCEncoder, self).__init__()
        self.tnet3 = TNet(input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, out_size, 1)

    def forward(self, x):
        B, W, N, F = x.shape
        x = x.view(B * W, N, F) # B*W, N,F
        tnet3 = self.tnet3(x)
        x = torch.bmm(x, tnet3).transpose(1,2)
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)
        return x

class TrackEncoder(nn.Module):
    def __init__(self, input_dim=9, out_size=1024):
        super(TrackEncoder, self).__init__()
        self.tnet9 = TNet(input_dim)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, out_size, 1)

    def forward(self, x):
        B, W, F = x.shape
        x = x.view(B * W, F, 1).transpose(1, 2)
        tnet9 = self.tnet9(x)
        x = torch.bmm(x, tnet9).transpose(1, 2)
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = self.conv4(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)
        return x

class TemporalDecoder(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, output_head_size):
        super(TemporalDecoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        self.fc_center = nn.Sequential(
            nn.Linear(lstm_hidden_size, output_head_size),
            nn.ReLU(),
            nn.Linear(output_head_size, 3)
        )

        self.fc_size = nn.Sequential(
            nn.Linear(lstm_hidden_size, output_head_size),
            nn.ReLU(),
            nn.Linear(output_head_size, 3)
        )

        self.fc_rotation = nn.Sequential(
            nn.Linear(lstm_hidden_size, output_head_size),
            nn.ReLU(),
            nn.Linear(output_head_size, 2)
        )

    def forward(self, x):
        x, _ = self.lstm(x)  # Shape: (B, W, lstm_hidden_size)

        x = x[:, -1, :]  # Take the last output of LSTM for each batch: (B, lstm_hidden_size)

        center = self.fc_center(x)
        size = self.fc_size(x)
        rotation = self.fc_rotation(x)

        output = torch.cat((center, size, rotation), dim=-1)  # Shape: (B, 8)

        return output
    