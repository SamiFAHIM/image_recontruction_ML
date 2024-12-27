import os
from pathlib import Path

# Dynamically set the project root based on the file's location
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]  # Adjust the number to match your directory structure

# Set the working directory to the project root
os.chdir(project_root)

# Add project root to sys.path for module imports
import sys
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Confirm
print(f"Project root set to: {project_root}")
print(f"Current working directory: {os.getcwd()}")

from pathlib import Path

import math
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import spyrit.core.meas as meas
import spyrit.core.noise as noise
import spyrit.core.prep as prep
import spyrit.misc.statistics as stats
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet

from spyrit.misc.sampling import sort_by_significance
from spyrit.misc.metrics import psnr_,ssim
from src.pattern_order import choose_pattern_order

def test_model_on_data(model_name=None, pattern_order=None,alpha=10,img_size=64,verbose=False,model_path=None):
    if (model_name is None) or (pattern_order is None):
        print("Please provide model name and pattern order")
        return
    image_folder = 'data/images/'       # images for simulated measurements
    image_folder_full = Path.cwd() / Path(image_folder)
    if model_path is None:
        model_folder = 'model/'             # reconstruction models
        model_folder_full = Path.cwd() / Path(model_folder)
    else:
        model_folder_full = Path(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model_unet_path = os.path.join(model_folder_full, model_name)

    transform = stats.transform_gray_norm(img_size)
    # define dataset and dataloader. `image_folder_full` should contain
    # a class folder with the images
    dataset = torchvision.datasets.ImageFolder(
        image_folder_full, 
        transform=transform
        )

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=False
        )


    x, _ = next(iter(dataloader))
    print(f"Shape of input images: {x.shape}")