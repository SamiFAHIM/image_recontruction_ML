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


from spyrit.misc.disp import add_colorbar, noaxis, imagesc
from spyrit.misc.statistics import Cov2Var
from spyrit.misc.sampling import sort_by_significance
from spyrit.misc.metrics import psnr_,ssim
from pattern_order import choose_pattern_order

# %% Order of measurements

# In[2]:


# General
# --------------------------------------------------------------------
# Experimental data
image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics

#
"""working_directory= 'C:/Users/marti/OneDrive - INSA Lyon/5GE/TDSI/Projet/fork_sami/image_recontruction_ML-1/2024_tdsi/'
# Full paths
image_folder_full = Path(working_directory) / Path(image_folder)
model_folder_full = Path(working_directory) / Path(model_folder)
stat_folder_full  = Path(working_directory)/ Path(stat_folder)
"""
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[3]:


# Load images
# --------------------------------------------------------------------

img_size = 64 # image size
i = 1
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


x, _ = next(iter(dataloader))
print(f"Shape of input images: {x.shape}")

# Select image
x = x[i : i + 1, :, :, :]
x = x.detach().clone()
print(f"Shape of selected image: {x.shape}")
b, c, h, w = x.shape

# plot
imagesc(x[0, 0, :, :], r"$x$ in [-1, 1]")

# In[3]:


# Simulate measurements for three image intensities
# --------------------------------------------------------------------
# Measurement parameters
# alpha_list = [2, 10, 50] # Poisson law parameter for noisy image acquisitions
alpha = 10 # Poisson law parameter for noisy image acquisitions
img_size = 64
h=img_size
und = 4
M = img_size ** 2 // und  # Number of measurements (here, 1/4 of the pixels)

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

# Mask of order
mask_basis = np.zeros((h, h))
mask_basis.flat[:M] = 1
mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")

im = plt.imshow(mask)
plt.title("Acquisition in " + order_name + " order", fontsize=20)
add_colorbar(im, "bottom", size="20%")


# %% 
# Measurement and noise operators
meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op, alpha)
prep_op = prep.SplitPoisson(alpha, meas_op)
#%%
 
# Measurement vectors
torch.manual_seed(0)    # for reproducibility
noise_op.alpha = alpha
print ("shape of x", x.shape)
print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse



y = noise_op(x)

# %% Reco with Pinv and a Unet Denoiser
#Load the pretrained weights of the Unet
from spyrit.core.nnet import ConvNet, Unet
from spyrit.core.recon import PinvNet
from spyrit.core.train import load_net
import os


denoi_net = Unet ()
full_op = PinvNet ( noise_op , prep_op, denoi_net)
data_name = "noise_set_pinv-net_Unet_stl10_N0_10_N_64_M_1024_epo_5_lr_0.001_sss_10_sdr_0.5_bs_256.pth"
#entrainé sur un ordre des basse freq et sur des images 128
# recontruire avec un ordre de subsampling HF et réentrainé un modèle sur ça
# prendre un peut de BF et un de 
model_unet_path = os.path.join(model_folder_full, data_name)
load_net(model_unet_path, full_op, device, False)

    

    

# %%
B, C, H, M = b, c, h, M
expected_shape = (B * C, 2 * M)

# Vérification initiale
print("Shape of y before reconstruct:", y.shape)

# Ajustement si nécessaire
if y.shape != expected_shape:
    y = y.view(B * C, -1)
    print("unexpected shape ,Shape of y after reshaping:", y.shape)

if y.dim() == 2:
    print ( 'yo')
    y = y.unsqueeze(1)  # Ajoute une dimension pour les canaux
# Appel de la reconstruction
with torch.no_grad():
    x_rec_2 = full_op.reconstruct(y)
print("Reconstructed shape:", x_rec_2.shape)
# %% Plot
fig,axis = plt.subplots(1,2)

axis[1].imshow(x_rec_2.view(h, w).cpu().numpy(), cmap='gray')
axis[1].set_title(' PinvNEt  Unet Denoiser')
    
axis[0].imshow(x.view(h,w).cpu().numpy(), cmap='gray')
axis[0].set_title('original image')
plt.show()
print(x.shape)
print(x_rec_2.shape)
print("PSNR Low_freq=", psnr_(x.view(h, h).cpu().numpy(), x_rec_2.view(h, h).cpu().numpy()))
# %%
