from smoother.models.attentions import MultiHeadAttention
from torch import nn


class PointTransformer(nn.Module):

    def __init__(self, input_dim=11, out_size=11, mlp_sizes=[64,64,128,512], num_heads=5, window_size=5):
        super(PointTransformer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.embed_dim = mlp_sizes[-1]
        self.head_dim = self.embed_dim // self.num_heads
        d_model = mlp_sizes[-1]

        # PointNet MLP
        mlp_sizes.insert(0, input_dim)
        mlp_sizes[-1] = mlp_sizes[-1]*3 # to divide into q,k,v later
        mlp_modules = []
        for i in range(len(mlp_sizes)-1):
            mlp_modules.append(nn.Linear(mlp_sizes[i], mlp_sizes[i+1]))
        self.mlp = nn.Sequential(*mlp_modules)

        self.multihead_attn = MultiHeadAttention(d_model, self.num_heads)

        kernel_size = (window_size,1)
        stride = 1
        padding = 0
        self.max_pool = nn.MaxPool2d(kernel_size, stride, padding)
        
        mlp2_sizes = [512,128]
        mlp2_sizes.insert(0, mlp_sizes[-1])
        mlp2_modules = []
        for i in range(len(mlp2_sizes)-1):
            mlp2_modules.append(nn.Linear(mlp2_sizes[i], mlp2_sizes[i+1]))
        self.mlp2 = nn.Sequential(*mlp2_modules)

        self.fc_out = nn.Linear(mlp2_sizes[-1], out_size)

    def forward(self, x):
        print("x_in", x.shape)
        B, W, F = x.size()
        print("sizes", B,W,F)
        qkv = self.mlp(x)
        print("qkv_enc", qkv.shape)
        #qkv = qkv.reshape(B, W, self.num_heads, 3*self.head_dim)
        print("qkv_reshaped", qkv.shape)
        #qkv = qkv.permute(0, 2, 1, 3) # [B, H, W, F]
        print("qkv_permuted", qkv.shape)
        q, k, v = qkv.chunk(3, dim=-1)
        print("qkv chunked", q.shape,k.shape,v.shape)
        attn_out, attn_out_weights = self.multihead_attn(q,k,v)
        print("attn_out", attn_out.shape)
        print("attn_out_weights", attn_out_weights.shape)

        # Decoder
        x_dec = self.mlp2(attn_out)
        print("x_dec1", x_dec.shape)
        x_out = self.fc_out(x_dec)
        print("x_out", x_out)


        return x_out