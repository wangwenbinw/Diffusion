import torch
import torch.nn as nn
from src.models.modules import (
    SinusoidalTimeEmbedding,
    ResNetBlock,
    AttentionBlock,
    DownBlock,
    UpBlock
)


# %%
class UNet(nn.Module):
    def __init__(self,num_steps=1000,time_embedding_dim=512,base_channels=64,num_heads=4):
        super().__init__()

        # ============================================
        # Time embedding
        # ============================================
        self.time_embedding = SinusoidalTimeEmbedding(num_steps=num_steps,embedding_dim=time_embedding_dim)


        # =============================================
        # Input convolution [B,3,32,32]->[B,64,32,32]
        # =============================================
        self.input_conv = nn.Conv2d(in_channels=3,out_channels=base_channels,kernel_size=3,padding=1)

        # -------------------------------------------------
        # Channel configuration
        # -------------------------------------------------
        c1 = base_channels          # 64
        c2 = base_channels * 2      # 128
        c3 = base_channels * 4      # 256
        c4 = base_channels * 8      # 512

        # =================================================
        # Encoder Stage 1: 32×32
        # =================================================
        self.enc1_res1 = ResNetBlock(c1,c1,time_embedding_dim)
        self.enc1_res2 = ResNetBlock(c1,c1,time_embedding_dim)
        self.down1 = DownBlock(c1,c2)

        # =================================================
        # Encoder Stage 2: 16×16
        # ================================================== 
        self.enc2_res1 = ResNetBlock(c2,c2,time_embedding_dim)
        self.enc2_res2 = ResNetBlock(c2,c2,time_embedding_dim)
        self.down2 = DownBlock(c2,c3)

        # =================================================
        # Encoder Stage 3: 8×8
        # ================================================== 
        self.enc3_res1 = ResNetBlock(c3,c3,time_embedding_dim)
        self.enc3_res2 = ResNetBlock(c3,c3,time_embedding_dim)
        self.enc3_attn = AttentionBlock(num_heads=num_heads,input_dim=c3)
        self.down3 = DownBlock(c3,c4)

        # =================================================
        # Encoder Stage 4: 4x4
        # ================================================== 
        self.enc4_res1 = ResNetBlock(c4,c4,time_embedding_dim)
        self.enc4_res2 = ResNetBlock(c4,c4,time_embedding_dim)
        self.enc4_attn = AttentionBlock(num_heads=num_heads,input_dim=c4)


        # =================================================
        # Middle / Bottleneck 4×4
        # ResBlock->Attention-> ResBlock
        # =================================================
        self.mid_res1 = ResNetBlock(c4,c4,time_embedding_dim)
        self.mid_attn = AttentionBlock(num_heads=num_heads,input_dim=c4)
        self.mid_res2 = ResNetBlock(c4,c4,time_embedding_dim)


        # =================================================
        # Decoder Stage 4: 4×4
        # =================================================
        self.dec4_res1 = ResNetBlock(c4,c4,time_embedding_dim)
        self.dec4_res2 = ResNetBlock(c4,c4,time_embedding_dim)
        self.dec4_attn = AttentionBlock(num_heads=num_heads,input_dim=c4)
        self.up4 = UpBlock(c4,c3)

        # =================================================
        # Decoder Stage 3: 8×8
        # =================================================
        self.dec3_res1 = ResNetBlock(c3+c3,c3,time_embedding_dim)
        self.dec3_res2 = ResNetBlock(c3,c3,time_embedding_dim)
        self.dec3_attn = AttentionBlock(num_heads=num_heads,input_dim=c3)
        self.up3 = UpBlock(c3,c2)


        # =================================================
        # Decoder Stage 2: 16×16
        # =================================================
        self.dec2_res1 = ResNetBlock(c2+c2,c2,time_embedding_dim)
        self.dec2_res2 = ResNetBlock(c2,c2,time_embedding_dim)
        self.up2 = UpBlock(c2,c1)


        # =================================================
        # Decoder Stage 1: 32x32
        # =================================================
        self.dec1_res1 = ResNetBlock(c1+c1,c1,time_embedding_dim)
        self.dec1_res2 = ResNetBlock(c1,c1,time_embedding_dim)

        # =================================================
        # Output head
        # [B,c1,32,32] -> [B,3,32,32]
        # 预测噪声 epsilon
        # =================================================
        self.output = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=c1,
                out_channels=3,
                kernel_size=3,
                padding=1
            )
        )


    def forward(self,x,timesteps):
        '''
        输入：
            x: [B,3,32,32]
            timesteps: [B]
        输出：
            predicted_noise: [B,3,32,32]
        '''
        # =================================================
        # 1. Time embedding
        # [B] -> [B,time_embedding_dim]
        # =================================================
        time_emb = self.time_embedding(timesteps)
        return self.forward_with_embedding(x, time_emb)

    def forward_with_embedding(self,x,time_emb):
        # =================================================
        # 2. Input convolution
        # [B,3,32,32] -> [B,c1,32,32]
        # =================================================
        x = self.input_conv(x)

        # =================================================
        # Encoder Stage 1: 32×32
        # =================================================
        x = self.enc1_res1(x, time_emb)
        x = self.enc1_res2(x, time_emb)
        # 保存 skip connection
        skip1 = x
        x = self.down1(x)

        # =================================================
        # Encoder Stage 2: 16×16
        # =================================================
        x = self.enc2_res1(x, time_emb)
        x = self.enc2_res2(x, time_emb)
        skip2 = x
        x = self.down2(x)


        # =================================================
        # Encoder Stage 3: 8×8
        # =================================================
        x = self.enc3_res1(x, time_emb)
        x = self.enc3_res2(x, time_emb)
        x = self.enc3_attn(x)
        skip3 = x
        x = self.down3(x)

        # =================================================
        # Encoder Stage 4: 4×4
        # =================================================
        x = self.enc4_res1(x, time_emb)
        x = self.enc4_res2(x, time_emb)
        x = self.enc4_attn(x)


        # =================================================
        # Middle / Bottleneck: 4×4
        # =================================================
        x = self.mid_res1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, time_emb)


        # =================================================
        # Decoder Stage 4: 4×4
        # 不做 skip concat
        # =================================================
        x = self.dec4_res1(x, time_emb)
        x = self.dec4_res2(x, time_emb)
        x = self.dec4_attn(x)
        # 4×4 -> 8×8
        x = self.up4(x)

        # =================================================
        # Decoder Stage 3: 8×8
        # 与 Encoder Stage 3 拼接
        # =================================================
        x = torch.cat([x, skip3],dim=1)
        x = self.dec3_res1(x, time_emb)
        x = self.dec3_res2(x, time_emb)
        x = self.dec3_attn(x)
        # 8×8 -> 16×16
        x = self.up3(x)

        # =================================================
        # Decoder Stage 2: 16×16
        # 与 Encoder Stage 2 拼接
        # =================================================
        x = torch.cat([x, skip2],dim=1)
        x = self.dec2_res1(x, time_emb)
        x = self.dec2_res2(x, time_emb)
        # 16×16 -> 32×32
        x = self.up2(x)

        # =================================================
        # Decoder Stage 1: 32×32
        # 与 Encoder Stage 1 拼接
        # =================================================
        x = torch.cat([x, skip1],dim=1)
        x = self.dec1_res1(x, time_emb)
        x = self.dec1_res2(x, time_emb)

        # =================================================
        # Output
        # [B,c1,32,32] -> [B,3,32,32]
        # =================================================
        x = self.output(x)

        return x



        



        
        





