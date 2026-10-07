import torch.nn as nn
from src.models.unet import UNet 

class ConditionalUNet(UNet):
    def __init__(self, num_steps=1000, time_embedding_dim=512, base_channels=64, num_heads=4,num_classes=10):
        super().__init__(num_steps=num_steps, time_embedding_dim=time_embedding_dim, base_channels=base_channels, num_heads=num_heads)
        self.num_classes = num_classes
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=time_embedding_dim)
    
    def forward(self,x, timesteps,labels,conditional_mask=None):
        # [B] -> [B,D]
        time_emb = self.time_embedding(timesteps)
        class_emb = self.class_embedding(labels)
        # 为后续CFG提供移除类别条件的能力
        if conditional_mask is not None:
            # 把掩码[B]扩展为[B,1]，然后广播到类别嵌入的维度，这样每个样本的掩码就能乘到它的整个类别嵌入上。
            mask = conditional_mask.to(device=class_emb.device,dtype=class_emb.dtype).reshape(-1,1)
            class_emb = class_emb * mask

        # 时间和类别信息共同传递给所有的残差块
        combined_emb = time_emb + class_emb

        return self.forward_with_embedding(x, combined_emb)