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

Code is intended for GitHub. Datasets and training outputs are intended for
Hugging Face and are excluded from Git by `.gitignore`.

Public Hugging Face repositories:

- Dataset: https://huggingface.co/datasets/Wenbinwang/cifar10-data
- Training outputs: https://huggingface.co/Wenbinwang/diffusion-outputs

To restore them, install the Hugging Face CLI and run these commands from the
project root:

```bash
pip install huggingface_hub
hf download Wenbinwang/cifar10-data --repo-type dataset --local-dir Unet/data
hf download Wenbinwang/diffusion-outputs --local-dir Unet/outputs
```

The training outputs include optimizer, scheduler, model, and EMA state in
PyTorch `.pt` checkpoints. They use the custom architectures in this repository.
