# SliceMamba

This repository contains code for training segmentation models on skin lesion and polyp datasets.

## Environments
- conda creative -n slicemamba python=3.8
- conda activate slicemamba
- pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
- pip install packaging
- pip install timm==0.4.12
- pip install pytest chardst yacs termcolor
- pip install submitit tensorboardX
- pip install triton==2.0.0
- pip install causal_conv1d==1.0.0
- pip install mamba_ssm==1.0.1
- pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs

## Download skin lesion datasets and polyp datasets

## prepare the pre-trained weights

## Train slicemamba on skin lesion datasets
- cd SliceMamba
- python train.py

## Train slicemamba on polyp datasets
- python train_polyp.py
  
