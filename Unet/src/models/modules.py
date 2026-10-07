#%%
import torch 
import torch.nn as nn
import math

#%%
# =========================================================
# 1. Sinusoidal positional / time embedding
# =========================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self,num_steps,embedding_dim):
        super().__init__()
        assert embedding_dim % 2 == 0,'output_dim should be even for sinusoidal embedding.'
        self.embedding_dim = embedding_dim
        # [num_steps,1]
        timesteps = torch.arange(num_steps).unsqueeze(1)
        # [embedding_dim/2]
        frequencies = 10000 ** ( -torch.arange(0, embedding_dim, 2) / embedding_dim) 
        # [num_steps,embdeeing_dim]
        embeddings = torch.zeros(num_steps, embedding_dim)

        embeddings[:, 0::2] = torch.sin(timesteps * frequencies)
        embeddings[:, 1::2] = torch.cos(timesteps * frequencies)

        #是不可训练参数，但是会跟随模型一起保存，加载，移动设备,不会出现在model.paramaters()里面，默认不会被优化器更新
        #会出现在state_dict()里面，model.cuda/model.device会一起移动，保存模型的时候会一起保存
        self.register_buffer('embeddings', embeddings)

    def forward(self, timesteps):
        ''' 
        输入：
            timesteps:[B],每个Batch中样本对应的时间步
        输出：
            embedded_time: [B,emdedding_dim]
        '''
        return self.embeddings[timesteps]






#%%
# =========================================================
# 2. Time embedding projection:embedding 再做一次非线性变换 + 线性投影，转换成网络某一层需要的维度。
# =========================================================
class TimeProjection(nn.Module):
    def __init__(self,time_embedding_dim,output_dim):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim,output_dim)
        )
    def forward(self,time_emb):
        ''' 
            输入维度：[B,embdedding_dim]
            输出维度：[B,output_dim]
        '''
        return self.time_proj(time_emb)




# =========================================================
# 3. Multi-head self-attention
# 标准的pre-Norm: norm1-attention-residual-norm2-forward-residual
# 一般来说，标准的self-attention的输入输出的整体形状是不会发生改变的
# =========================================================
class AttentionBlock(nn.Module):
    def __init__(self,num_heads,input_dim):
        super().__init__()
        assert input_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads 
        self.attention_dim = input_dim
        self.input_dim = input_dim

        # Q,K,V projection
        # n = h*w
        # [b,n,input_dim] -> [b,n,emb_dim]
        self.q_proj = nn.Linear(input_dim,input_dim)
        self.k_proj = nn.Linear(input_dim,input_dim)
        self.v_proj = nn.Linear(input_dim,input_dim)

        # PRE-norm
        self.norm1 = nn.LayerNorm(input_dim) # 对注意力（Attention）分支做归一化

        # ffn
        self.norm2 = nn.LayerNorm(input_dim) #对前馈网络（FFN）分支做归一化
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

        # output projection
        # [B, N, input_dim] -> [B, N, input_dim]
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self,x):
        '''
        输入：[B,C,H,W]
        输出: [B,C,H,W]
        '''
        B,C,H,W = x.shape

        # -------------------------------------------------
        # 1. image feature map -> sequence
        # [B, C, H, W] -> [B, H*W, C]
        # -------------------------------------------------
        x = x.permute(0,2,3,1).reshape(B,H*W,C)
        # -------------------------------------------------
        # 2. Pre-Norm
        # [B, H*W, C] -> [B, H*W, C]
        residual = x
        x_norm = self.norm1(x)
        # -------------------------------------------------
        # 3. Q, K, V projection
        # [B, H*W, C]->[B, H*W, emb_dim]
        # ------------------------------------------------
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        # -------------------------------------------------
        # 4. split into multiple heads
        # [B, H*W, emb_dim] -> [B, H*W, n_heads, head_dim] -> [B,n_heads,H*W,head_dim]
        # -------------------------------------------------
        q = q.view(B,H*W,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(B,H*W,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,H*W,self.num_heads,self.head_dim).transpose(1,2)
        # --------------------------------------------------
        # [B,n_heads,H*W,head_dim] -> [B,n_heads,H*W,H*W] 
        attn_scores = torch.matmul(q,k.transpose(-2,-1))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores,dim=-1)
        # -------------------------------------------------
        # 5. attention @ V
        # [B, n_heads, H*W, H*W] @ [B, n_heads, H*W, head_dim] -> [B, n_heads, H*W , head_dim]
        # -------------------------------------------------
        out = torch.matmul(attn_weights,v)
        # -------------------------------------------------
        # 6. merge multiple heads
        # [B, n_heads, H*W, head_dim] -> [B, H*W, n_heads, head_dim] -> [B, H*W, emb_dim]
        # -------------------------------------------------
        out = out.transpose(1,2).reshape(B,H*W, self.num_heads*self.head_dim)
        out = self.out_proj(out)
        out = residual + out 
        # [B, H*W, C] + [B, H*W, C] -> [B, H*W, C]
        out = (out + self.mlp(self.norm2(out)))
        out = out.reshape(B,H,W,C).permute(0,3,1,2)
        return out



# =========================================================
# 4. ResNet Block
# =========================================================
class ResNetBlock(nn.Module):
    ''' 
    x: [B,C_in,H,W]
    h: [B,C_out,H,W]
    '''
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8,in_channels), #对每个样本单独处理,把in_channels个通道分成8组做归一化,但是shape不变
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        self.block2 = nn.Sequential(
                    nn.GroupNorm(8,out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        self.time_projection = TimeProjection(
            time_embedding_dim = time_embedding_dim,
            output_dim=out_channels
        )
        # Residual shortcut
        # 如果输入输出通道相同，直接保留原输入
        # 如果不同，用 1×1 Conv 调整通道数
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self,x,time_emb):
        '''
        U-Net 层里的图像特征图 x: [B,in_channels,H,W]
        time_emb : [B, time_embedding_dim]
        '''

        residual = self.skip(x)
        h = self.block1(x)
        time_emb = self.time_projection(time_emb)
        # [B,C] -> [B,C,1,1]
        time_emb = time_emb[:,:,None,None]
        h = h + time_emb
        h = self.block2(h)
        h = residual + h

        return h


# =========================================================
# 5. Downsample
# =========================================================
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.downscale = nn.Conv2d(
            in_channels= in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
    def forward(self,x):
        return self.downscale(x)


# =========================================================
# 6. Upsample
# =========================================================
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding = 1
        )
    def forward(self, x):
        return self.upscale(x)



