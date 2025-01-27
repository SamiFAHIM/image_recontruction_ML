#%%
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
h = img_size
alpha = 10 # Poisson law parameter for noisy image acquisitions
subsampling_factor = 4
M = img_size ** 2 // subsampling_factor  # Number of measurements (1/4 of the pixels)

# In[4]:


# Choose the pattern order

order_name = 'low_freq'
#order_name = 'naive'
#order_name = 'high_freq'
#order_name = 'variance'
#order_name = 'random'
# order_name = 'random_variance'
# order_name = 'random_variance_2'
# order_name = 'random_variance_3'
#order_name='70_lf'


# In[ ]:


from spyrit.misc.statistics import data_loaders_stl10
from pathlib import Path

# Parameters
data_root = Path("./data_model_training")  # path to data folder (where the dataset is stored)
batch_size = 256

# Dataloader for STL-10 dataset
mode_run = False # Set to True to run the training
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
from spyrit.core.nnet import ConvNet, Unet
from spyrit.core.recon import PinvNet
from spyrit.core.train import train_model
from datetime import datetime


def train_for_order(order_name, model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    Ord_rec = choose_pattern_order(order_name, img_size)

    # Mask of order
    mask_basis = np.zeros((h, h))
    mask_basis.flat[:M] = 1 # M valeurs qui sont égales à 1
    print("size of the variance vector",Ord_rec.shape)
    mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")

    im = plt.imshow(mask)
    plt.title("Acquisition in " + order_name + " order", fontsize=20)
    add_colorbar(im, "bottom", size="20%")

    meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
    noise_op = noise.Poisson(meas_op,alpha=alpha)
    prep_op = prep.SplitPoisson(alpha, meas_op)
    
    model.noise_op = noise_op
    model.prep_op = prep_op
    

    # Send to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # Use multiple GPUs if available

    model = model.to(device)
    name_run = "stdl10_hadampos"
    now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    tb_path = f"runs/runs_{name_run}_n{int(N0)}_m{M}/{now}"
    # regarder comment load un modele
    #checkpoint_path = "./model/model_epoch_0.pth"
    #load_net(checkpoint_path, model, device, False)
    # Train the network
    
    model_root = Path(model_folder)  # path to model saving files
    checkpoint_interval = 2  # interval between saving model checkpoints
    tb_freq = (
        50  # interval between logging to Tensorboard (iterations through the dataloader)
    )
    
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

    return model, train_info
# In[7]:
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from spyrit.core.train import save_net , load_net
from misc.Weight_Decay_Loss import Weight_Decay_Loss
mask_basis = np.zeros((h, h))
mask_basis.flat[:M] = 1 # M valeurs qui sont égales à 1
Ord_rec = choose_pattern_order(order_name, img_size)
mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")

im = plt.imshow(mask)
plt.title("Acquisition in " + order_name + " order", fontsize=20)
add_colorbar(im, "bottom", size="20%")

meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
noise_op = noise.Poisson(meas_op,alpha=alpha)
prep_op = prep.SplitPoisson(alpha, meas_op)
denoiser = Unet()
model = PinvNet(noise_op, prep_op, denoi=denoiser)  # None for noise_op and prep_op, will update later

# Define training parameters
num_epochs_per_order = 10  # Number of epochs for each acquisition order
lr = 1e-3
step_size = 10
gamma = 0.5
N0=alpha
loss = nn.MSELoss()
criterion = Weight_Decay_Loss(loss)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Train the model for each order
if mode_run:
    # Define the orders and the number of epochs per segment
    base_orders = ["low_freq", "70_lf", "random"]
    epochs_per_segment = 2  # Number of epochs per segment
    total_epochs = 30  # Total epochs for training (adjust as needed)

    # Create the shuffled order list
    orders = []
    for i in range(total_epochs // epochs_per_segment):
        for order_name in base_orders:
            orders.append((order_name, epochs_per_segment))

    # Shuffle the orders for randomness
    #np.random.seed(42)  # Ensure reproducibility
    #np.random.shuffle(orders)

    # Train the model for each shuffled order
    for order_name, epochs in orders:
        print(f"Training with order '{order_name}' for {epochs} epochs.")
        model, train_info = train_for_order(order_name, model, dataloaders, criterion, optimizer, scheduler, epochs)
else:
    print("Mode run is disabled. No training performed.")





# In[8]: defining the model and the training parameters and training the model
"""

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from spyrit.core.train import save_net , load_net
from src.Weight_Decay_Loss import Weight_Decay_Loss

# Parameters
lr = 1e-3
step_size = 10
gamma = 0.5
N0=alpha

loss = nn.MSELoss()
criterion = Weight_Decay_Loss(loss)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

from spyrit.core.train import train_model
from datetime import datetime

# Parameters
model_root = Path(model_folder)  # path to model saving files
num_epochs = 50  # number of training epochs (num_epochs = 30)
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

"""
# In[ ]:


from spyrit.core.train import save_net
model_root = Path(model_folder)
checkpoint_interval = 2  # interval between saving model checkpoints


if mode_run:
    # Training parameters
    train_type = "N0_{:g}".format(N0)
    arch = "pinv-net_mult_acq"
    denoi = "Unet_weight_decay"
    data = "stl10"
    reg = 1e-7  # Default value
    suffix = "N_{}_M_{}_epo_{}_lr_{}_sss_{}_sdr_{}_bs_{}".format(
        h, M, total_epochs, lr, step_size, gamma, batch_size
    )
    title = model_root / f"{arch}_{denoi}_{data}_{train_type}_{suffix}"
    print(title)

    Path(model_root).mkdir(parents=True, exist_ok=True)

    if checkpoint_interval:
        Path(title).mkdir(parents=True, exist_ok=True)

    save_net(str(title) + ".pth", model)

    # Save training history
    import pickle


    from spyrit.core.train import Train_par

    params = Train_par(batch_size, lr, h, reg=reg)
    params.set_loss(train_info)

    train_path = model_root / f"TRAIN_{arch}_{denoi}_{data}_{train_type}_{suffix}.pkl"

    with open(train_path, "wb") as param_file:
        pickle.dump(params, param_file)
    torch.cuda.empty_cache()

else:
    print("the model was not trained, no need to save the training history, set mode_run to True to train the model")


# In[ ]:


# Plot
# sphinx_gallery_thumbnail_number = 2
if mode_run :
    fig = plt.figure()
    plt.plot(train_info["train"], label="train")
    plt.plot(train_info["val"], label="val")
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.legend(fontsize=20)
    plt.show()
else : 
    print("the model was not trained, no need to plot the training history, set mode_run to True to train the model")