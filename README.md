# Diffusion

PyTorch implementations of an unconditional DDPM and a conditional DDPM with
UNet attention and exponential moving average (EMA), trained on CIFAR-10.

## Install and train

Install PyTorch and torchvision for your CUDA environment, then install the
remaining dependencies:

```bash
pip install -r requirements.txt
cd Unet
python train.py
python train_conditional.py
```

Run the training scripts separately. They load CIFAR-10 from `Unet/data`,
resolved relative to the script location. If the dataset is missing,
torchvision downloads it automatically. Training creates checkpoints,
sample images, and loss history under `Unet/outputs`.

## Repository layout

- `Unet/train.py`: unconditional DDPM training.
- `Unet/train_conditional.py`: conditional DDPM training.
- `Unet/src/data.py`: CIFAR-10 preprocessing and dataloaders.
- `Unet/src/models/`: UNet architectures and shared modules.
- `Unet/src/diffusion/`: diffusion training and sampling.

GitHub hosts code and documentation. Hugging Face hosts the dataset and only
the two final checkpoints. Data and training outputs are excluded from Git by
`.gitignore`.

Public Hugging Face repositories:

- Dataset: https://huggingface.co/datasets/Wenbinwang/cifar10-data
- Final checkpoints: https://huggingface.co/Wenbinwang/diffusion-outputs

## Final checkpoint policy

Only these final checkpoints are selected for publication on Hugging Face:

| Model | Checkpoint path in the Hugging Face repository |
| --- | --- |
| Unconditional DDPM | `ddpm_with_attention_ema/checkpoints/ddpm_final.pt` |
| Conditional DDPM | `conditional_ddpm_with_attention_ema/checkpoints/conditional_ddpm_final.pt` |

Each file is about 3.11 GiB (about 6.2 GiB combined). Intermediate epoch
checkpoints, sample images, and training-history CSV files are not published.
The final files retain the original full checkpoint format: model and EMA
weights, optimizer and scheduler state, epoch, and losses. Use
`ema_model_state_dict` for EMA inference weights or `model_state_dict` for the
training model weights, with the matching architectures in this repository.

To restore them, install the Hugging Face CLI and run these commands from the
project root:

```bash
pip install huggingface_hub
hf download Wenbinwang/cifar10-data --repo-type dataset --local-dir Unet/data
hf download Wenbinwang/diffusion-outputs ddpm_with_attention_ema/checkpoints/ddpm_final.pt conditional_ddpm_with_attention_ema/checkpoints/conditional_ddpm_final.pt --local-dir Unet/outputs
```
