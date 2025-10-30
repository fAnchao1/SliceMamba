# SliceMamba

Official code for paper: SliceMamba with Neural Architecture Search for Medical Image Segmentation

#### Abstract
Despite the progress made in Mamba-based medical image segmentation models, existing methods utilizing unidirectional or multi-directional feature scanning
mechanisms struggle to effectively capture dependencies between neighboring positions, limiting the discriminant representation learning of local features. These local features are crucial for medical image segmentation as they provide critical structural information about lesions and
organs. To address this limitation, we propose SliceMamba, a simple yet effective locally sensitive Mamba-based medical image segmentation model. SliceMamba features an efficient Bidirectional Slicing and Scanning (BSS) module, which performs bidirectional feature slicing and employs
varied scanning mechanisms for sliced features with distinct shapes. This design keeps spatially adjacent features close in the scan sequence, preserving the local structure of the image and enhancing segmentation performance. Additionally, to fit the varying sizes and shapes of lesions and organs, we introduce an Adaptive Slicing Search method that automatically identifies the optimal feature slicing method based on the characteristics of the target
data. Extensive experiments on two skin lesion datasets (ISIC2017 and ISIC2018), two polyp segmentation datasets (Kvasir and ClinicDB), one ultra-wide field retinal hemorrhage segmentation dataset (UWF-RHS), and one multiorgan segmentation dataset (Synapse) demonstrate the effectiveness of our method.
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

## Download skin lesion, polyp, and synapse datasets

You can follow [VM-UNet](https://github.com/JCruan519/VM-UNet?tab=readme-ov-file) to download the dataset.

## Introduce the Workflow Using Skin Lesion as an Example

### First, train the parameters of the supernet
- python train.py
- After training is complete, the supernet parameters will be saved in the "./results" folder.
### Subsequently, perform the search based on the supernet parameters
- python search.py
- After the search is complete, the relevant results will be saved in the "./log_nas" folder. At the same time, the optimal architecture will be pretrained on ImageNet,with the pretraining method following [VMamba](https://github.com/MzeroMiko/VMamba).
### Finally, retrain and test from scratch based on the obtained search results
- cd Evaluation
- python eval.py
- You will get a dst_folder in evaluation, then
- cd data/**
- python train.py

