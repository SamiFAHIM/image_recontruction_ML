# %% 
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
import spyrit.core.train as train

from spyrit.misc.disp import add_colorbar, noaxis, imagesc
from spyrit.misc.sampling import sort_by_significance
from spyrit.misc.metrics import psnr_,ssim
from src.pattern_order import choose_pattern_order
 #%% 
def test_model_on_data(model_name=None,model_type=nnet.Unet, pattern_order=None,alpha=10,und=4,img_size=64,verbose=False,model_path=None):
    """
    Test a denoising model on a set of images.
    NEED TO ADD STAT to USE COV order
    Parameters
    ----------
    model_name : str
        Name of the model to test.
    pattern_order : str
        Pattern order to use for the measurements.
    alpha : int
        Strength of the noise.
    img_size : int
        Size of the images to test on.
    verbose : bool
        Print out the shape of the input images.
    model_path : str
        Path to the model folder.

    Returns
    -------
    None
    """
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
    if(verbose):
        print("Images loaded")
        print(f"Shape of input images: {x.shape}")


    # GETTING PATTERN ORDER
    M=img_size**2//und
    Ord_rec = choose_pattern_order(pattern_order, img_size)
    # Mask of order
    mask_basis = np.zeros((img_size, img_size))
    mask_basis.flat[:M] = 1
    mask = sort_by_significance(mask_basis, Ord_rec, axis="flatten")
    if (verbose):
        im = plt.imshow(mask)
        plt.title("Acquisition in " + pattern_order + " order", fontsize=20)
        add_colorbar(im, "bottom", size="20%")
    

        # SIMULATE MEASUREMENTS
    x = x[0:1, :, :, :]
    X1 = x.detach().clone()
    if (verbose):
        plt.figure()
        imagesc(X1[0, 0, :, :], r"$x$")
    b, c, h, w = X1.shape
    alpha = alpha
    meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
    noise_op = noise.Poisson(meas_op, alpha)
    prep_op = prep.SplitPoisson(alpha, meas_op)
    torch.manual_seed(0)    # for reproducibility
    noise_op.alpha = alpha
    y = noise_op(X1)
    m = prep_op(y)
    f_stat = meas_op.pinv(m)

    if verbose:
        plt.figure()
        plt.imshow(f_stat.view(h, w).cpu().numpy(), cmap='gray')
        plt.title('Static reconstruction')
        plt.colorbar()
        plt.show()

    denoi_net = model_type()
    full_op = recon.PinvNet ( noise_op , prep_op, denoi_net)
    data_name = model_name
    #entrainé sur un ordre des basse freq et sur des images 128
    # recontruire avec un ordre de subsampling HF et réentrainé un modèle sur ça
    # prendre un peut de BF et un de 
    model_unet_path = os.path.join(model_folder_full, data_name)
    train.load_net(model_unet_path, full_op, device, False)


# %% 
test_model_on_data(model_name='right_noise_level_pinv-net_Unet_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order='70_lf',alpha=10,img_size=64,verbose=False)


# %%
