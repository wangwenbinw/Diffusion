import torch
from src.diffusion.ddpm import DDPM

class ConditionalDDPM(DDPM):
    def __init__(self,num_steps,eps_model,condition_drop_prob):
        super().__init__(num_steps=num_steps, eps_model=eps_model)

        if not 0<= condition_drop_prob <= 1:
            raise ValueError("condition_drop_prob must be in [0,1]")

        self.condition_drop_prob = condition_drop_prob

    def forward(self,imgs,labels):
        batch_size = imgs.shape[0]
        labels = labels.to(device=imgs.device,dtype=torch.long)

        # 1.随机选择时间步,生成一维张量, t.shape = (batch_size,)  
        t = torch.randint(0,self.num_steps,(batch_size,),device=imgs.device)
        # 2.根据时间步t,对图像进行前向扩散,得到噪声图像x_t和噪声噪声eps，和普通DDPM一样
        noise = torch.randn_like(imgs)
        sqrt_alpha_bar_t = (self.sqrt_alpha_bar_schedule[t].view(batch_size, 1, 1, 1))
        sqrt_one_minus_alpha_bar_t = (self.sqrt_one_minus_alpha_bar_schedule[t].view(batch_size, 1, 1, 1))
        noisy_imgs = (sqrt_alpha_bar_t * imgs+ sqrt_one_minus_alpha_bar_t * noise)
        # 3. 训练时随机移除部分样本的类别条件
        if self.training:
            # 0 到 1 之间的均匀分布随机数。
            condition_mask = (torch.rand(batch_size, device=imgs.device) >= self.condition_drop_prob).to(imgs.dtype)
        else:
            condition_mask = torch.ones(batch_size, device=imgs.device, dtype=imgs.dtype)
        # 4. 将标签和掩码传给条件 U-Net
        pred_noise = self.eps_model(noisy_imgs, t, labels, conditional_mask=condition_mask)
        # 5. 目标仍然是真实噪声
        return self.criterion(pred_noise, noise)

    @torch.no_grad()
    def sample(self, labels, batch_size, image_shape, guidance_scale=1.0):
        device = self.beta_schedule.device
        labels = labels.to(device=device,dtype=torch.long)
        if labels.shape != (batch_size,):
            raise ValueError("labels 的形状必须是 [batch_size]")
        was_training = self.training
        self.eval()
        try:
            # 从标准的高斯噪声开始
            x_t = torch.randn(batch_size, *image_shape, device=device)
            condition_ones = torch.ones(batch_size, device=device)
            condition_zeros = torch.zeros(batch_size, device=device)
            for t in range(self.num_steps - 1, -1, -1):
                # 创建一个长度为 batch_size 的一维张量，并且里面所有元素都等于 t
                t_tensor = torch.full((batch_size,),t,device=device,dtype=torch.long)
                # 有条件噪声预测
                pred_cond = self.eps_model(x_t,t_tensor,labels,conditional_mask=condition_ones)
                if guidance_scale == 1.0:
                    pred_noise = pred_cond
                else:
                    # 无条件噪声预测
                    pred_uncond = self.eps_model(x_t,t_tensor,labels,conditional_mask=condition_zeros)
                    # Classifier-Free Guidance
                    pred_noise = (pred_uncond+ guidance_scale * (pred_cond - pred_uncond))

                # 反向采样均值，与普通 DDPM 相同
                mean = self.sqrt_recip_alpha_schedule[t] * (x_t - self.beta_schedule[t]/ self.sqrt_one_minus_alpha_bar_schedule[t]* pred_noise)
                if t > 0:
                    x_t = (mean+ torch.sqrt(self.posterior_variance_schedule[t])* torch.randn_like(x_t))
                else:
                    # 最后一步不添加随机噪声
                    x_t = mean
            return x_t
        finally:
            # 采样结束后恢复之前的训练/评估模式
            self.train(was_training)



    


