import torch
from torch import nn
from torch.nn import functional

### ENCODERS ###


class TNet(nn.Module):
    def __init__(self, k, out_size=256):
        super(TNet, self).__init__()
        self.k = k
        self.out_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(k, 32),
            nn.ReLU(),
            # nn.Linear(32, 64),
            # nn.ReLU(),
            nn.Linear(32, out_size),
            nn.ReLU(),
        )
        self.fc = nn.Linear(out_size, k * k)
        self.register_buffer("identity", torch.eye(k))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp(x)
        x = torch.max(x, 1, keepdim=True)[0]
        x = x.view(-1, self.out_size)
        x = self.fc(x)
        x = x.view(batch_size, self.k, self.k) + self.identity
        return x


class FuseEncoder(nn.Module):
    def __init__(self, input_dim=10, out_size=256, dropout_rate=0.0):
        super(FuseEncoder, self).__init__()
        self.tnet = TNet(input_dim, out_size)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, out_size, 1)
        # self.conv3 = nn.Conv1d(128, out_size, 1)
        # self.conv4 = nn.Conv1d(256, out_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,N,10)
                padding_mask    (B,W)
        output: x               (B,W,out_size)
        """
        B, W, N, F = x.shape
        x = x.view(B * W, N, F)  # B*W, N,F
        tnet = self.tnet(x)
        x = torch.bmm(x, tnet).transpose(1, 2)
        x = functional.relu(self.dropout(self.conv1(x)))
        # x = functional.relu(self.dropout(self.conv2(x)))
        x = self.dropout(self.conv2(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)

        # Apply the padding mask to exclude padded tensors from being updated
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x * (1 - padding_mask)

        return x


class PCEncoder(nn.Module):
    def __init__(self, input_dim=3, out_size=256, dropout_rate=0.0):
        super(PCEncoder, self).__init__()
        self.tnet = TNet(input_dim, out_size)
        self.conv1 = nn.Conv1d(input_dim, out_size, 1)
        # self.conv2 = nn.Conv1d(64, out_size, 1)
        # self.conv3 = nn.Conv1d(128, 256, 1)
        # self.conv4 = nn.Conv1d(256, out_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,N,10)
                padding_mask    (B,W)
        output: x               (B,W,out_size)
        """
        B, W, N, F = x.shape
        x = x.view(B * W, N, F)  # B*W, N,F
        tnet = self.tnet(x)
        x = torch.bmm(x, tnet).transpose(1, 2)
        # x = functional.relu(self.dropout(self.conv1(x)))
        # x = functional.relu(self.dropout(self.conv2(x)))
        # x = functional.relu(self.dropout(self.conv3(x)))
        x = self.dropout(self.conv1(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)

        # Apply the padding mask to exclude padded tensors from being updated
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x * (1 - padding_mask)

        return x


class TrackEncoder(nn.Module):
    def __init__(self, input_dim=7, out_size=64, dropout_rate=0.0):
        super(TrackEncoder, self).__init__()
        self.tnet = TNet(input_dim, out_size)
        self.conv1 = nn.Conv1d(input_dim, out_size, 1)
        # self.conv2 = nn.Conv1d(32, out_size, 1)
        # self.conv3 = nn.Conv1d(64, out_size, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask):
        """
        input:  x               (B,W,N,10)
                padding_mask    (B,W)
        output: x               (B,W,out_size)
        """
        B, W, F = x.shape
        x = x.view(B * W, 1, F)
        tnet = self.tnet(x)
        x = torch.bmm(x, tnet).transpose(1, 2)
        # x = functional.relu(self.dropout(self.conv1(x)))
        # x = functional.relu(self.dropout(self.conv2(x)))
        x = self.dropout(self.conv1(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, W, -1)

        # Apply the padding mask to exclude padded tensors from being updated
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = x * (1 - padding_mask)

        return x
