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
from src.pattern_order import choose_pattern_order


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


# Measurement and noise operators
meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op)
prep_op = prep.SplitPoisson(alpha, meas_op)

 
# Measurement vectors
torch.manual_seed(0)    # for reproducibility
noise_op.alpha = alpha
print ("shape of x", x.shape)
print ("measurment operator" , meas_op.H_pinv.shape)# c'est la pseudo inverse



y = noise_op(x)


# In[7]:

print(x.shape)
# PLOTTING THE MEAURED IMAGE
figure, axis = plt.subplots(2, 2)
with torch.no_grad():
    axis[0,0].imshow(x_plot.squeeze(), cmap="gray")
    axis[1,0].imshow(x_plot.squeeze(), cmap="gray")
    # Measurement and noise operators
    meas_op = meas.HadamSplit(M, h, torch.from_numpy(choose_pattern_order("low_freq",img_size)))
    noise_op = noise.Poisson(meas_op)
    prep_op = prep.SplitPoisson(alpha, meas_op)
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



# In[29]:


#training the denoiser a simple CNN





# In[30]:


# %% STATIC RECO sans Pinv classe
from spyrit.core.nnet import Unet, ConvNet
from spyrit.core.train import load_net
import os

figure, axis = plt.subplots(2, 2)
with torch.no_grad():
    m = prep_op(y)
    f_stat = meas_op.pinv(m)

    axis[0,0].imshow(f_stat.view(h, w).cpu().numpy(), cmap='gray')
    axis[0,0].set_title('Static reconstruction,fig i')
    
    



# %% Reco with Pinv

"""
denoi_net_conv = ConvNet()
full_op = PinvNet ( noise_op , prep_op,denoi_net_conv )
#Load the pretrained w
with torch.no_grad():
    x_rec = full_op.reconstruct ( y )
    axis[0,1].imshow(x_rec.view(h, w).cpu().numpy(), cmap='gray')
    axis[0,1].set_title('PinvNet, no denoiser, fig j')
"""    


# %% Reco with Pinv and a Unet Denoiser
#Load the pretrained weights of the Unet
from spyrit.core.nnet import ConvNet, Unet
from spyrit.core.recon import PinvNet
from spyrit.core.train import load_net


denoi_net = Unet ()
meas_op = meas.HadamSplit(M, h, torch.from_numpy(choose_pattern_order("random",img_size)))
noise_op = noise.Poisson(meas_op)
prep_op = prep.SplitPoisson(alpha, meas_op)
full_op = PinvNet ( noise_op , prep_op, denoi_net)
data_name = "pinv-net_unet_stl10_N0_10_N_64_M_1024_epo_5_lr_0.001_sss_10_sdr_0.5_bs_256.pth"
#entrainé sur un ordre des basse freq et sur des images 128
# recontruire avec un ordre de subsampling HF et réentrainé un modèle sur ça
# prendre un peut de BF et un de 
model_unet_path = os.path.join(model_folder, data_name)
load_net(model_unet_path, full_op, device, False)
with torch.no_grad():
    
    x_rec_2 = full_op . reconstruct ( y )
    axis[1,1].imshow(x_rec_2.view(h, w).cpu().numpy(), cmap='gray')
    axis[1,1].set_title(' PinvNEt  Unet Denoiser')
    
axis[1,0].imshow(x.view(h,w).cpu().numpy(), cmap='gray')
axis[1,0].set_title('original image')
plt.show()


# In[8]:


#just a test if the GPU is working
import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # Simple tensor computation on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    z = x + y
    print(f"Computation result on GPU: {z}")
else:
    print("CUDA is not available. Please check your setup.")


# In[ ]:


from spyrit.misc.statistics import data_loaders_stl10
from pathlib import Path

# Parameters
h = 64  # image size hxh
data_root = Path("./data_model_training")  # path to data folder (where the dataset is stored)
batch_size = 256

# Dataloader for STL-10 dataset
mode_run = True
if mode_run:
    dataloaders = data_loaders_stl10(
        data_root,
        img_size=h,
        batch_size=batch_size,
        seed=7,
        shuffle=True,
        download=False,
    )




# In[6]: Defining hadamart matrix and the measurement operator and the dataloader


from spyrit.core.meas import Linear
from spyrit.core.noise import NoNoise
from spyrit.core.prep import DirectPoisson


img_size=64 # à check
h = img_size
und = 4
M = img_size ** 2 // und 

Ord_rec = choose_pattern_order(order_name, img_size)

# Mask of order
mask_basis = np.zeros((h, h))
mask_basis.flat[:M] = 1 # M valeurs qui sont égales à 1
print("size of the variance vector",Ord_rec.shape)
mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")

im = plt.imshow(mask)
plt.title("Acquisition in " + order_name + " order", fontsize=20)
add_colorbar(im, "bottom", size="20%")

alpha = 10  # Mean maximum total number of photons

meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op)
prep_op = prep.SplitPoisson(alpha, meas_op)


# In[7]:

torch.cuda.empty_cache()
from spyrit.core.nnet import ConvNet, Unet
from spyrit.core.recon import PinvNet

denoiser = Unet()
model = PinvNet(noise_op, prep_op, denoi=denoiser)

# Send to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)




# In[8]: defining the model and the training parameters and training the model


import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from spyrit.core.train import save_net, Weight_Decay_Loss , load_net

# Parameters
lr = 1e-3
step_size = 10
gamma = 0.5
N0=alpha

loss = nn.MSELoss()
criterion = Weight_Decay_Loss(loss)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)# We train for one epoch only to check that everything works fine.

from spyrit.core.train import train_model
from datetime import datetime

# Parameters
model_root = Path("./model")  # path to model saving files
num_epochs = 5  # number of training epochs (num_epochs = 30)
checkpoint_interval = 2  # interval between saving model checkpoints
tb_freq = (
    50  # interval between logging to Tensorboard (iterations through the dataloader)
)

# Path for Tensorboard experiment tracking logs
name_run = "stdl10_hadampos"
now = datetime.now().strftime("%Y-%m-%d_%H-%M")
tb_path = f"runs/runs_{name_run}_n{int(N0)}_m{M}/{now}"
# regarder comment load un modele
#checkpoint_path = "./model/model_epoch_0.pth"
#load_net(checkpoint_path, model, device, False)
# Train the network
if mode_run:
    model, train_info = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        device,
        model_root,
        num_epochs=num_epochs,#-1, # because already trained for 1 epoch
        disp=True,
        do_checkpoint=checkpoint_interval,
        tb_path=tb_path,
        tb_freq=tb_freq,
    )
else:
    train_info = {}


# In[ ]:


from spyrit.core.train import save_net

# Training parameters
train_type = "N0_{:g}".format(N0)
arch = "pinv-net"
denoi = "cnn"
data = "stl10"
reg = 1e-7  # Default value
suffix = "N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}".format(
    h, M, num_epochs, lr, step_size, gamma, batch_size
)
title = model_root / f"{arch}_{denoi}_{data}_{train_type}_{suffix}"
print(title)

Path(model_root).mkdir(parents=True, exist_ok=True)

if checkpoint_interval:
    Path(title).mkdir(parents=True, exist_ok=True)

save_net(str(title) + ".pth", model)

# Save training history
import pickle

if mode_run:
    from spyrit.core.train import Train_par

    params = Train_par(batch_size, lr, h, reg=reg)
    params.set_loss(train_info)

    train_path = model_root / f"TRAIN_{arch}_{denoi}_{data}_{train_type}_{suffix}.pkl"

    with open(train_path, "wb") as param_file:
        pickle.dump(params, param_file)
    torch.cuda.empty_cache()

else:
    from spyrit.misc.load_data import download_girder

    url = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
    dataID = "667ebfe4baa5a90007058964"  # unique ID of the file
    data_name = "tuto4_TRAIN_pinv-net_cnn_stl10_N0_1_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07.pkl"
    train_path = os.path.join(model_root, data_name)
    # download girder file
    download_girder(url, dataID, model_root, data_name)

    with open(train_path, "rb") as param_file:
        params = pickle.load(param_file)
    train_info["train"] = params.train_loss
    train_info["val"] = params.val_loss


# In[ ]:


# Plot
# sphinx_gallery_thumbnail_number = 2

fig = plt.figure()
plt.plot(train_info["train"], label="train")
plt.plot(train_info["val"], label="val")
plt.xlabel("Epochs", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.legend(fontsize=20)
plt.show()

