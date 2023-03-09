from torch import nn
import torch

class PointNet(nn.Module):

    def __init__(self, input_dim=11, out_size=11, mlp1_sizes=[64,64], mlp2_sizes=[64,128,1024], mlp3_sizes=[512,256], window_size=5):
        super(PointNet, self).__init__()
        self.input_dim = input_dim

        self.mlp1 = MLP(input_dim, mlp1_sizes)
        self.mlp2 = MLP(mlp1_sizes[-1], mlp2_sizes)
        self.mlp3 = MLP(mlp1_sizes[-2], mlp3_sizes)

        self.t_net1 = TNet(window_size, input_dim, [64,128,1024], [512,256])
        self.t_net2 = TNet(window_size, mlp1_sizes[-1], [64,128,1024], [512,256])


        kernel_size = (window_size,1)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        

        self.fc_out = nn.Linear(mlp3_sizes[-1], out_size)

    def forward(self, x):
        x_t_net1 = self.t_net1(x)
        x = torch.matmul(x, x_t_net1)
        x = self.mlp1(x)
        x_t_net2 = self.t_net2(x)
        x = torch.matmul(x, x_t_net2)
        x = self.mlp2(x)
        x = self.max_pool(x)
        x = self.mlp3(x)
        x = self.fc_out(x)

        return x
    

class MLP(nn.Module):

    def __init__(self, input_size=16, layer_sizes=[64,128,256]):
        super(MLP, self).__init__()

        layer_sizes.insert(0,input_size)
        mlp_modules = []
        for i in range(len(layer_sizes)-1):
            mlp_modules.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x):
        x = self.mlp(x)
        return x


class TNet(nn.Module):
    
    def __init__(self, window_size, input_size, mlp1_sizes=[64,128,1024],  mlp2_sizes=[512,256]):
        super(TNet, self).__init__()

        self.mlp1 = MLP(input_size, mlp1_sizes)

        kernel_size = (window_size,1)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        self.mlp2 = MLP(mlp1_sizes[-1], mlp2_sizes)

        self.lin = nn.Linear(mlp2_sizes[-1], input_size**2)

    def forward(self, x):
            B, W, F = x.shape
            x = self.mlp1(x)
            x = self.max_pool(x)
            x = self.mlp2(x)
            x = self.lin(x)
            x = x.reshape(B,F,F)
            return x

    
class PointNet2(nn.Module):

    def __init__(self, input_dim=11, out_size=11, mlp1_sizes=[64,64,128,512], mlp2_sizes=[128,64,32], window_size=5):
        super(PointNet2, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        mlp1_sizes.insert(0, input_dim)
        mlp_modules = []
        for i in range(len(mlp1_sizes)-1):
            mlp_modules.append(nn.Linear(mlp1_sizes[i], mlp1_sizes[i+1]))
        self.mlp = nn.Sequential(*mlp_modules)

        kernel_size = (self.window_size,1)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)

        mlp2_sizes.insert(0, input_dim*2)
        mlp2_modules = []
        for i in range(len(mlp2_sizes)-1):
            pass
        mlp2_modules = [nn.Conv2d(mlp2_sizes[i], mlp2_sizes[i+1], kernel_size=[1,1]) for i in range(len(mlp2_sizes)-1)]
        self.mlp2 = nn.Sequential(*mlp2_modules)
        
        in_channels = mlp1_sizes[-1]*2
        out_channels = 128
        kernel_size=(1,self.window_size)
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size, stride=1, padding=0)

    def forward(self, x):
        B,W,F = x.shape
        print("x_in", x.shape)
        x_enc = self.mlp(x)
        print("x_enc", x.shape)
        x_mp = self.max_pool(x_enc)
        print("x_max_pooled", x.shape)
        x_mpr = x_mp.repeat(1,W,1)
        print("x_repeated", x.shape)
        x_concat = torch.cat((x_enc,x_mpr), -1)
        
        x_dec = self.mlp2(x_concat)
        print("x_dec", x_dec.shape)
        assert False
        print("x_concat", x_concat.shape)
        x_unsq = x_concat.unsqueeze(-1).permute(0,2,3,1)
        print("x_unsqueezed", x_unsq.shape)
        x_conv = self.conv(x_unsq).squeeze(-1)
        print("x_conv", x_conv.shape)


        assert False

        return x