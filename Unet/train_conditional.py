import os
import csv
import torch
import copy

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import get_dataloaders
from src.models.conditional_unet import ConditionalUNet
from src.diffusion.conditional_ddpm import ConditionalDDPM

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay

        # 复制一份模型作为 EMA 模型
        self.ema_model = copy.deepcopy(model)

        # EMA 模型不参与梯度计算
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            model.parameters()
        ):
            ema_param.data.mul_(self.decay)
            ema_param.data.add_(
                model_param.data,
                alpha=1 - self.decay
            )

# =========================================================
# 1. Train one epoch
# =========================================================
def train_one_epoch(model,ema,train_dataloader, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")

    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device,non_blocking=True)

        optimizer.zero_grad()
        loss = model(imgs,labels)
        loss.backward()
        optimizer.step()
        global_step = (
            (epoch - 1) * len(train_dataloader)
            + batch_idx
            + 1
        )

        if global_step <= 1000:
            # 第 1～1000 步：直接同步训练模型，避免保留随机初始参数
            ema.ema_model.load_state_dict(
                model.state_dict()
            )
        else:
            # 从第 1001 步开始使用指数滑动平均
            ema.update(model)

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    return total_loss / len(train_dataloader)


# =========================================================
# 2. Validation
# =========================================================
@torch.no_grad()
def validate(model, val_dataloader, device):
    model.eval()
    total_loss = 0.0

    for imgs, labels in val_dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device,non_blocking=True)
        loss = model(imgs,labels)
        total_loss += loss.item()

    return total_loss / len(val_dataloader)


# =========================================================
# 3. Generate samples
# =========================================================
@torch.no_grad()
def generate_samples(model,batch_size,image_shape,save_path,labels,guidance_scale=3.0):
    model.eval()

    samples = model.sample(labels=labels,batch_size=batch_size,image_shape=image_shape,guidance_scale=guidance_scale)
    # 把模型输出重新变成正常图像
    # Model output is in [-1, 1] -> Convert it back to image range [0, 1]
    # how to realize it? x_new = 2x - 1 -> x = (x_new+1)/2
    samples = (samples.clamp(-1, 1) + 1) / 2

    # save_image会把[0,1]范围内的浮点张量转换成8-bit图像所需要的[0.255]整像素值再保存
    save_image(samples,save_path,nrow=4)


# =========================================================
# 4. Save checkpoint
# =========================================================
def save_checkpoint(model,ema,optimizer,scheduler,epoch,train_loss,val_loss,save_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
             "ema_model_state_dict": ema.ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss
        },
        save_path
    )


# =========================================================
# 5. Training
# =========================================================
def train(
    model,
    ema,
    optimizer,
    scheduler,
    train_dataloader,
    val_dataloader,
    device,
    epochs,
    output_dir="./outputs",
    save_every=10,
    sample_every=10,
    sample_batch_size=16,
    image_shape=(3, 32, 32),
    guidance_scale=3.0
):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    sample_dir = os.path.join(output_dir, "samples")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(1, epochs + 1):

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model,
            ema,
            train_dataloader,
            optimizer,
            device,
            epoch,
            epochs
        )

        val_loss = validate(
            model,
            val_dataloader,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        learning_rates.append(current_lr)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        scheduler.step()

        # =====================================================
        # Save checkpoint
        # =====================================================
        if epoch % save_every == 0:

            save_path = os.path.join(
                checkpoint_dir,
                f"conditional_ddpm_epoch_{epoch}.pt"
            )

            save_checkpoint(
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                val_loss,
                save_path
            )

        # =====================================================
        # Generate conditional samples
        # =====================================================
        if epoch % sample_every == 0:

            sample_path = os.path.join(
                sample_dir,
                f"samples_epoch_{epoch}.png"
            )

            # 16 张图：
            # 0 = airplane
            # 3 = cat
            # 8 = ship
            # 9 = truck
            sample_labels = torch.tensor(
                [
                    0, 0, 0, 0,
                    3, 3, 3, 3,
                    8, 8, 8, 8,
                    9, 9, 9, 9
                ],
                device=device,
                dtype=torch.long
            )

            generate_samples(
                model=ema.ema_model,
                batch_size=sample_batch_size,
                image_shape=image_shape,
                save_path=sample_path,
                labels=sample_labels,
                guidance_scale=guidance_scale
            )

    # =========================================================
    # Save training history
    # =========================================================
    history_path = os.path.join(
        output_dir,
        "training_history.csv"
    )

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "learning_rate"
            ]
        )

        for epoch, (train_loss, val_loss, lr) in enumerate(
            zip(
                train_losses,
                val_losses,
                learning_rates
            ),
            start=1
        ):
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_loss,
                    lr
                ]
            )

    # =========================================================
    # Save final checkpoint
    # =========================================================
    final_path = os.path.join(
        checkpoint_dir,
        "conditional_ddpm_final.pt"
    )

    save_checkpoint(
        model,
        ema,
        optimizer,
        scheduler,
        epochs,
        train_losses[-1],
        val_losses[-1],
        final_path
    )

    return (
        train_losses,
        val_losses,
        learning_rates
    )


# =========================================================
# 6. Main
# =========================================================
def main():

    # -------------------------
    # Configuration
    # -------------------------
    seed = 42

    image_size = 32
    batch_size = 128
    val_ratio = 0.1
    num_workers = 4

    num_steps = 1000

    base_channels = 128
    time_embedding_dim = 512
    num_heads = 4

    # Conditional DDPM
    num_classes = 10
    condition_drop_prob = 0.1
    guidance_scale = 1.0

    # EMA
    ema_decay = 0.9999

    epochs = 80
    learning_rate = 2e-4
    weight_decay = 1e-4

    save_every = 10
    sample_every = 10
    sample_batch_size = 16

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    output_dir = (
        "./outputs/conditional_ddpm_with_attention_ema"
    )

    # -------------------------
    # Device
    # -------------------------
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------
    # Data
    # -------------------------
    train_dataloader, val_dataloader = get_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=num_workers
    )

    # -------------------------
    # Conditional UNet
    # -------------------------
    eps_model = ConditionalUNet(
        num_steps=num_steps,
        time_embedding_dim=time_embedding_dim,
        num_classes=num_classes,
        base_channels=base_channels,
        num_heads=num_heads
    )

    # -------------------------
    # Conditional DDPM
    # -------------------------
    model = ConditionalDDPM(
        num_steps=num_steps,
        eps_model=eps_model,
        condition_drop_prob=condition_drop_prob
    ).to(device)

    # -------------------------
    # EMA
    # -------------------------
    ema = EMA(
        model,
        decay=ema_decay
    )

    # -------------------------
    # Number of parameters
    # -------------------------
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: {num_params:,}"
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )

    # -------------------------
    # Learning rate scheduler
    # -------------------------
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    # -------------------------
    # Train
    # -------------------------
    train(
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=epochs,
        output_dir=output_dir,
        save_every=save_every,
        sample_every=sample_every,
        sample_batch_size=sample_batch_size,
        image_shape=(
            3,
            image_size,
            image_size
        ),
        guidance_scale=guidance_scale
    )


if __name__ == "__main__":
    main()
