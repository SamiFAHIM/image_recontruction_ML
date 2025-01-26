#!/usr/bin/env python
# coding: utf-8

# In[1]:

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
import spyrit.core.recon as recon
import spyrit.core.nnet as nnet
import spyrit.core.train as train
import spyrit.misc.statistics as stats
import spyrit.external.drunet as drunet
from spyrit.core.recon import PinvNet


from spyrit.misc.disp import add_colorbar, noaxis
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.sampling import sort_by_significance
from spyrit.misc.metrics import psnr_,ssim
from misc.pattern_order import choose_pattern_order


# In[2]: Defining paths, and the device
image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)





# In[3]:
#Defining the image size and the noise constants and the subsampling factor
img_size = 64
alpha = 10 # Poisson law parameter for noisy image acquisitions
subsampling_factor = 4
M = img_size ** 2 // subsampling_factor  # Number of measurements (1/4 of the pixels)






# In[3]:
# Load images
# --------------------------------------------------------------------

print("Loading image...")
# crop to desired size, set to black and white, normalize
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

# select the image
x, _ = next(iter(dataloader))
x = x[1].unsqueeze(0)
b, c, h, w = x.shape # batch size, channels, height, width
print("Image shape:", x.shape)

x_plot = x.view(-1, h, h).cpu().numpy()# reshape to h x h

plt.imshow(x_plot.squeeze(), cmap="gray")
print("value of the image",x_plot[0,40,40])


# In[4]:


# Choose the pattern order


#order_name = 'low_freq'
#order_name = 'naive'
#order_name = 'high_freq'
#order_name = 'variance'
#order_name = 'random'
# order_name = 'random_variance'
# order_name = 'random_variance_2'
# order_name = 'random_variance_3'
order_name='70_lf'


# In[5]:


Ord_rec = choose_pattern_order(order_name, img_size)
count_ones= np.sum(Ord_rec)
print("number of ones in Ord_rec=",count_ones,"\nM=",(int(0.8*M)))

# Mask of order
mask_basis = np.zeros((h, h))
mask_basis.flat[:M] = 1 # M valeurs qui sont égales à 1
print("size of the variance vector",Ord_rec.shape)
mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")

im = plt.imshow(mask)
plt.title("Acquisition in " + order_name + " order", fontsize=20)
add_colorbar(im, "bottom", size="20%")


# In[6]:

torch.manual_seed(0)    # for reproducibility
# Measurement and noise operators
meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op,alpha=alpha)
prep_op = prep.SplitPoisson(alpha, meas_op)

 
# Measurement vectors

print ("shape of x", x.shape)
print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse



y = noise_op(x)
print(y.shape)

# %% STATIC RECO sans Pinv classe
from spyrit.core.nnet import Unet, ConvNet
from spyrit.core.train import load_net
import os

with torch.no_grad():
    m = prep_op(y)
    f_stat = meas_op.pinv(m)

    plt.imshow(f_stat.view(h, w).cpu().numpy(), cmap='gray')
    plt.title('Static reconstruction,fig i')
    f_stat_plot = f_stat.view(h, w).cpu().numpy()# reshape to h x h
    print("value of the image",np.min(f_stat_plot),np.max(f_stat_plot))
    


