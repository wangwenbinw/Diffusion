import torch.nn as nn
import torch 

#%%
class DDPM(nn.Module):
    def __init__(self,num_steps,eps_model):
        super().__init__()
        self.num_steps = num_steps
        self.eps_model = eps_model
        beta_schedule = torch.linspace(1e-4,0.02,num_steps) 
        alpha_schedule= 1 - beta_schedule
        alpha_bar_schedule = torch.cumprod(alpha_schedule,dim=0)
        alpha_bar_prev_schedule = torch.cat([torch.ones(1),alpha_bar_schedule[:-1]])
        sqrt_alpha_bar_schedule = torch.sqrt(alpha_bar_schedule)
        sqrt_one_minus_alpha_bar_schedule = torch.sqrt(1- alpha_bar_schedule)
        sqrt_recip_alpha_schedule = torch.sqrt(1/alpha_schedule)
        posterior_variance_schedule = ( beta_schedule * (1-alpha_bar_prev_schedule) / (1-alpha_bar_schedule))
        self.register_buffer('beta_schedule',beta_schedule)
        self.register_buffer('alpha_schedule',alpha_schedule)
        self.register_buffer('alpha_bar_schedule',alpha_bar_schedule)
        self.register_buffer('alpha_bar_prev_schedule',alpha_bar_prev_schedule)
        self.register_buffer('sqrt_alpha_bar_schedule',sqrt_alpha_bar_schedule)
        self.register_buffer('sqrt_one_minus_alpha_bar_schedule',sqrt_one_minus_alpha_bar_schedule)
        self.register_buffer('sqrt_recip_alpha_schedule',sqrt_recip_alpha_schedule)
        self.register_buffer('posterior_variance_schedule',posterior_variance_schedule)
        self.criterion = nn.MSELoss()

    def forward(self,imgs):
        # random choose some time steps
        t = torch.randint(0,self.num_steps,size=(imgs.shape[0],),device=imgs.device)

        # get random noise and add to images
        noise = torch.randn_like(imgs)
        batch_size,_,_,_  = imgs.shape 
        sqrt_alpha_bar_t = self.sqrt_alpha_bar_schedule[t].view(batch_size, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar_schedule[t].view(batch_size, 1, 1, 1)
        noise_imgs = sqrt_alpha_bar_t * imgs + sqrt_one_minus_alpha_bar_t * noise

        # get predict noise from model
        pred_noise = self.eps_model(noise_imgs,t)

        return self.criterion(pred_noise,noise)

    def sample(self,batch_size,image_shape):
        ''' 
        Args:
            batch_size:一次生成多少张图片
            image_shape: C,H,W
        return:
            x_t:B,C,H,W
        '''
        self.eval()
        with torch.no_grad():
            # get normal noise
            x_t = torch.randn(batch_size,*image_shape,device=self.beta_schedule.device)
            # calculate x_(t-1) on every iteration
            for t in range(self.num_steps-1,-1,-1):
                t_tensor = torch.full((x_t.shape[0],),t,device=x_t.device)
                # get predicted noise from model
                pred_noise = self.eps_model(x_t,t_tensor)
                z = torch.randn_like(x_t) if t > 0 else 0 
                x_t = self.sqrt_recip_alpha_schedule[t] * (x_t - pred_noise * self.beta_schedule[t] / self.sqrt_one_minus_alpha_bar_schedule[t]) + torch.sqrt(self.posterior_variance_schedule[t]) * z 
            return x_t
















        