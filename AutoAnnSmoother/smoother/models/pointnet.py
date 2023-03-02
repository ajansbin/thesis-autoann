from torch import nn

class WaymoPointNet(nn.Module):

    def __init__(self, input_dim=11, out_size=11, mlp1_sizes=[64,64,128,512], mlp2_sizes=[128,128], window_size=5):
        super(WaymoPointNet, self).__init__()
        self.input_dim = input_dim

        self.mlp1 = nn.Sequential(
                        nn.Linear(self.input_dim, mlp1_sizes[0]),
                        nn.Linear(mlp1_sizes[0], mlp1_sizes[1]),
                        nn.Linear(mlp1_sizes[1], mlp1_sizes[2]),
                        nn.Linear(mlp1_sizes[2], mlp1_sizes[3])
                    )
        kernel_size = (1,16)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        
        self.mlp2 = nn.Sequential(
                        nn.Linear(mlp1_sizes[3]-15, mlp2_sizes[0]),
                        nn.Linear(mlp2_sizes[0], mlp2_sizes[1])
                    )

        self.decoder = nn.Sequential(
                        nn.Conv1d(window_size,16,window_size),
                        nn.MaxPool2d((16,1))
                        )

        self.fc_out = nn.Linear(mlp2_sizes[1]-window_size+1, out_size)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.max_pool(x)
        x = self.mlp2(x)
        x = self.decoder(x)
        x = x.squeeze()
        x = self.fc_out(x)
        return x