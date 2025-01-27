#%% IMports
"""This script performs image reconstruction simulations using Hadamard basis 
# sampling and various measurement patterns. The main steps include:
# 1. Setting up the project environment, paths, and device configuration.
# 2. Loading a grayscale image dataset and selecting a single image for testing.
# 3. Defining measurement operators, noise models, and reconstruction algorithms.
# 4. Generating measurements with different sampling patterns (e.g., low-frequency, 
#    random, variance-based) and reconstructing the image using static methods.
# 5. Visualizing and comparing the reconstructed images and Hadamard patterns.
# 6. Computing performance metrics (PSNR, SSIM) to evaluate reconstruction quality.
"""
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



#%%
figure, axis = plt.subplots(2, 2)
with torch.no_grad():
    axis[0,0].imshow(x_plot.squeeze(), cmap="gray")
    order_name = "random"
    Ord_rec = choose_pattern_order(order_name, img_size)
    print ("taille du mask", Ord_rec.shape)
    # Mask of order
    mask_basis = np.zeros((h, h))
    mask_basis.flat[:M] = 1
    mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")
    plt.figure(figsize=(10, 10))  # Ajuster la taille selon vos besoins
    im = plt.imshow(mask)
    plt.title("Acquisition in " + order_name + " order", fontsize=20)
    add_colorbar(im, "bottom", size="20%")
    plt.show()
    # Measurement and noise operators
    meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
    noise_op = noise.Poisson(meas_op)
    prep_op = prep.SplitPoisson(alpha, meas_op)
    # Extraire l'image à afficher
    hadamard_pattern = meas_op.H.numpy()[300:301, :].reshape(64, 64)

    # Créer une nouvelle figure avec une taille plus grande
    plt.figure(figsize=(10, 10))  # Ajuster la taille selon vos besoins
    plt.imshow(hadamard_pattern)
    plt.title("Une Matrice d'Hadamard", fontsize=20)
    plt.axis("off")  # Masquer les axes pour une meilleure lisibilité
    plt.colorbar()  # Ajouter une barre colorée si nécessaire
    plt.show()
   
    # Measurement vectors
    torch.manual_seed(0)    # for reproducibility
    noise_op.alpha = alpha
    y = noise_op(x)#1
    m = prep_op(y)
    f_stat1 = meas_op.pinv(m)
    im1=f_stat1.view(h, w).cpu().numpy()
    axis[0,1].imshow(im1, cmap='gray')
    axis[0,1].set_title('Static reconstruction low_freq')


    meas_op = meas.HadamSplit(M, h, torch.from_numpy(choose_pattern_order("70_lf",img_size)))
    noise_op = noise.Poisson(meas_op)
    prep_op = prep.SplitPoisson(alpha, meas_op)
    # Measurement vectors
    torch.manual_seed(0)    # for reproducibility
    noise_op.alpha = alpha
    # print ("shape of x", x.shape)
    # print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse
    y = noise_op(x)
    m = prep_op(y)
    f_stat2 = meas_op.pinv(m)
    im2=f_stat2.view(h, w).cpu().numpy()
    axis[1,1].imshow(im2, cmap='gray')
    axis[1,1].set_title('Static reconstruction 70_lf')
    
# plt.imshow ( meas_op.H[1023,:].reshape(img_size, img_size))
# print ("taille de H", meas_op.H.shape)
# %% IMAGE 3 and 4
meas_op = meas.HadamSplit(M, h, torch.from_numpy(choose_pattern_order("random",img_size)))
noise_op = noise.Poisson(meas_op)
prep_op = prep.SplitPoisson(alpha, meas_op)
# Measurement vectors
torch.manual_seed(0)    # for reproducibility
noise_op.alpha = alpha
# print ("shape of x", x.shape)
# print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse
y = noise_op(x)
m = prep_op(y)
f_stat3 = meas_op.pinv(m)
im3=f_stat3.view(h, w).cpu().numpy()
print("stat folder",stat_folder_full)
meas_op = meas.HadamSplit(M, h, torch.from_numpy(choose_pattern_order("variance",img_size)))
noise_op = noise.Poisson(meas_op)
prep_op = prep.SplitPoisson(alpha, meas_op)
# Measurement vectors
torch.manual_seed(0)    # for reproducibility
noise_op.alpha = alpha
# print ("shape of x", x.shape)
# print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse
y = noise_op(x)
m = prep_op(y)
f_stat4 = meas_op.pinv(m)
im4=f_stat4.view(h, w).cpu().numpy()
# %% Metrics
print("\n")
print("PSNR Low_freq=", psnr_(x_plot.squeeze(), im1))
print("PSNR variance=", psnr_(x_plot.squeeze(), im4))
print("PSNR 70_lf=", psnr_(x_plot.squeeze(), im2))
print("PSNR random=", psnr_(x_plot.squeeze(), im3))
print("\n")
print("SSIM Low_freq=", ssim(x_plot.squeeze(), im1))
print("SSIM variance=", ssim(x_plot.squeeze(), im4))
print("SSIM 70_lf=", ssim(x_plot.squeeze(), im2))
print("SSIM random=", ssim(x_plot.squeeze(), im3))