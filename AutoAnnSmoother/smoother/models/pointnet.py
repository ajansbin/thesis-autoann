from torch import nn

class PointNet(nn.Module):

    def __init__(self, input_dim=11, out_size=11, mlp1_sizes=[64,64,128,512], mlp2_sizes=[128,128], window_size=5):
        super(PointNet, self).__init__()
        self.input_dim = input_dim

        self.mlp1 = nn.Sequential(
                        nn.Linear(self.input_dim, mlp1_sizes[0]),
                        nn.Linear(mlp1_sizes[0], mlp1_sizes[1]),
                        nn.Linear(mlp1_sizes[1], mlp1_sizes[2]),
                        nn.Linear(mlp1_sizes[2], mlp1_sizes[3])
                    )
        kernel_size = (window_size,1)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        
        self.mlp2 = nn.Sequential(
                        nn.Linear(mlp1_sizes[3], mlp2_sizes[0]),
                        nn.Linear(mlp2_sizes[0], mlp2_sizes[1])
                    )

        self.fc_out = nn.Linear(mlp2_sizes[1], out_size)

    def forward(self, x):
        #print("x_in", x.shape)
        x = self.mlp1(x)
        #print("x_enc", x.shape)
        x = self.max_pool(x)
        #print("x_max_pooled", x.shape)
        x = x.squeeze()
        #print("x_squeezed", x.shape)
        x = self.mlp2(x)
        #print("x_dec", x.shape)
        x = self.fc_out(x)
        #print("x_out", x.shape)
        return x