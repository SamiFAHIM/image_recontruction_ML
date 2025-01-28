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
from misc.pattern_order import choose_pattern_order
 #%% 
def test_model_on_data(model_name=None,model_type=nnet.Unet, pattern_order=None,alpha=10,und=4,img_size=64,verbose=False,model_path=None,nb_images=5):
    """
    Test a denoising model on a set of images.

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
    image_folder = 'data/images'       # images for simulated measurements
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
        batch_size=nb_images, 
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
    h=img_size
    meas_op = meas.HadamSplit(M, h, torch.from_numpy(Ord_rec))
    noise_op = noise.Poisson(meas_op, alpha)
    prep_op = prep.SplitPoisson(alpha, meas_op)
    torch.manual_seed(0)    # for reproducibility
    noise_op.alpha = alpha
    denoi_net = model_type()
    denoi_net.eval()
    # print("denoi",denoi_net.training)
    full_op = recon.PinvNet ( noise_op , prep_op, denoi_net)
    data_name = model_name
    model_unet_path = os.path.join(model_folder_full, data_name)
    train.load_net(model_unet_path, full_op, device, False)
    full_op.eval()
    psnr_tab=np.zeros((x.shape[0],1))
    ssim_tab=np.zeros((x.shape[0],1))
    psnr_tab_stat=np.zeros((x.shape[0],1))
    ssim_tab_stat=np.zeros((x.shape[0],1))
    if verbose:
        print ("valeur des pixels", x.min(), x.max())
    for i,image in enumerate(x):
        X1 = x[i:i+1, :, :, :].detach().clone()
        if (verbose):
            print("shape of X1 is ", X1.shape)
            plt.figure()
            imagesc(X1[0, 0, :, :],f'Original Image')
            # plt.title(f'Original image')
            plt.show()
        b, c, h, w = X1.shape
        y = noise_op(X1)
        m = prep_op(y)
        f_stat = meas_op.pinv(m)
        f_stat = meas_op.pinv(m)
        if verbose:
            plt.figure()
            plt.imshow(f_stat.view(h, w).cpu().numpy(), cmap='gray')
            plt.title('Psuedo inverse reconstruction')
            # plt.colorbar()
            plt.show()

        with torch.no_grad():
            
            x_rec = full_op.reconstruct(y)
            # print("fullop",full_op.training)
        if verbose:
            plt.figure()
            plt.imshow(x_rec.view(h, w).cpu().numpy(), cmap='gray')
            plt.title(f'Reconstructed image from model Variance (PSNR={round(psnr_(X1.view(h, h).cpu().numpy(), x_rec.view(h, h).cpu().numpy()),3)}, SSIM={round(ssim(X1.view(h, h).cpu().numpy(), x_rec.view(h, h).cpu().numpy()),3)})')           
            # plt.colorbar()
            plt.show()
        psnr_tab[i,0]=psnr_(X1.view(h, h).cpu().numpy(), x_rec.view(h, h).cpu().numpy())  
        ssim_tab[i,0]=ssim(X1.view(h, h).cpu().numpy(), x_rec.view(h, h).cpu().numpy())
        psnr_tab_stat[i,0]=psnr_(X1.view(h, h).cpu().numpy(), f_stat.view(h, h).cpu().numpy())  
        ssim_tab_stat[i,0]=ssim(X1.view(h, h).cpu().numpy(), f_stat.view(h, h).cpu().numpy())
    return psnr_tab,ssim_tab,psnr_tab_stat,ssim_tab_stat


#%%test 1 model
size_db=  100# number of images in the database
psnr, ssi, psnr_stat, ssim_stat = test_model_on_data(model_name='pinv-net_variance_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order="random",alpha=1000,img_size=64,verbose=False,nb_images=size_db)
print(f'random: \nmean psnr_avg {psnr.mean()}, ssim_avg {ssi.mean()}, psnr_stat_avg {psnr_stat.mean()}, ssim_stat_avg {ssim_stat.mean()}')
#%% BF

size_db=  100 # number of images in the database
psnr, ssi, psnr_stat, ssim_stat = test_model_on_data(model_name='pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order="low_freq",alpha=10,img_size=64,verbose=False,nb_images=size_db)
#%%
print(f'BF: \nmean psnr_avg {psnr.mean()}, ssim_avg {ssi.mean()}, psnr_stat_avg {psnr_stat.mean()}, ssim_stat_avg {ssim_stat.mean()},psnr_stat_std {psnr_stat.std()}, ssim_stat_std {ssim_stat.std()}')
 #%% 7°_lfcorr
 
size_db=  100 # number of images in the database
psnr, ssi, psnr_stat, ssim_stat = test_model_on_data(model_name='pinv-net_70_lfcorr_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order="70_lfcorr",alpha=10,img_size=64,verbose=False,nb_images=size_db)
print(f'70_lfcorr: \nmean psnr_avg {psnr.mean()}, ssim_avg {ssi.mean()}, psnr_stat_avg {psnr_stat.mean()}, ssim_stat_avg {ssim_stat.mean()}')
# %% 
# 1st Model
nb_models=5 # number of diff models to test
size_db=  100 # number of images in the database
model_names=["Low Frequency with WR","Variance with WR",'70_LFcorr with WR','70_LF with WR','70_LF without WR']
order_name="low_freq"
psnr_tab= np.zeros((nb_models,size_db)) # Stores the psnr for each model
ssim_tab= np.zeros((nb_models,size_db)) # Stores the ssim for each model
psnr,ssi,_,_ = test_model_on_data(model_name='pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=order_name,alpha=10,img_size=64,verbose=False,nb_images=size_db)
psnr_tab[0,:]=psnr.squeeze()
ssim_tab[0,:]=ssi.squeeze()


#2nd model
order_name="variance"
psnr,ssi,_,_= test_model_on_data(model_name='pinv-net_variance_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=order_name,alpha=10,img_size=64,verbose=False,nb_images=size_db)
psnr_tab[1,:]=psnr.squeeze()
ssim_tab[1,:]=ssi.squeeze()


#3rd model
order_name="70_lfcorr"
psnr,ssi,_,_= test_model_on_data(model_name='pinv-net_70_lfcorr_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=order_name,alpha=10,img_size=64,verbose=False,nb_images=size_db)
psnr_tab[2,:]=psnr.squeeze()
ssim_tab[2,:]=ssi.squeeze()
#%%
# 4th model
order_name="70_lf"
psnr,ssi,_,_= test_model_on_data(model_name='right_noise_level_pinv-net_Unet_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=order_name,alpha=10,img_size=64,verbose=False,nb_images=size_db)
psnr_tab[3,:]=psnr.squeeze()
ssim_tab[3,:]=ssi.squeeze()
#%%
#5th model
psnr,ssi,_,_= test_model_on_data(model_name='pinv-net_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=order_name,alpha=10,img_size=64,verbose=False,nb_images=size_db)
psnr_tab[4,:]=psnr.squeeze()
ssim_tab[4,:]=ssi.squeeze()
#%%
#  Plotting the results in an error bar plot

psnr_mean = np.mean(psnr_tab, axis=1)
psnr_std = np.std(psnr_tab, axis=1)

ssim_mean = np.mean(ssim_tab, axis=1)
ssim_std = np.std(ssim_tab, axis=1)

# Plotting the results
x = np.arange(psnr_tab.shape[0])  # Number of models vector

# plt.figure(figsize=(12, 6))
# plt.figure(figsize=(15, 6))

# Plot PSNR
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1) #Subplot for PSNR plotting
plt.errorbar(x, psnr_mean, yerr=psnr_std, fmt='o', capsize=5, label='PSNR', color='blue')
# plt.xticks(x, [f'Model {i+1}' for i in x])
plt.xticks(x, model_names,rotation=45,ha='right')
plt.title('PSNR Mean and Std with LF inference')
plt.xlabel('MODELS')
plt.ylabel('PSNR')
plt.ylim(10,28)
plt.ylim(10,28)
plt.grid(True)
plt.legend()
plt.tight_layout()
# Plot SSIM
plt.subplot(1, 2, 2)
# plt.figure()
plt.errorbar(x, ssim_mean, yerr=ssim_std, fmt='o', capsize=5, label='SSIM', color='green')
# plt.xticks(x, [f'Model {i+1}' for i in x])
plt.xticks(x, model_names,rotation=45,ha='right')
plt.title('SSIM Mean and Std with LF inference')
plt.xlabel('MODELS')
plt.ylabel('SSIM')
plt.ylim(0 ,1)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#%% 

psnr,ssi = test_model_on_data(model_name='pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07_retrained_light.pth',pattern_order='low_freq',alpha=10,img_size=128,verbose=False)



# %%
alpha_list=list(range(20, 0, -2))
psnr_mean=np.zeros((2,len(alpha_list)))
ssim_mean=np.zeros((2,len(alpha_list)))
psnr_std=np.zeros((2,len(alpha_list)))
ssim_std=np.zeros((2,len(alpha_list)))
for index,alpha in enumerate(alpha_list):
    psnr,ssi= test_model_on_data(model_name='pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order='70_lf',alpha=alpha,img_size=64,verbose=False)
    psnr_mean[0,index]=np.mean(psnr)
    ssim_mean[0,index]=np.mean(ssi)
    psnr_std[0,index]=np.std(psnr)
    ssim_std[0,index]=np.std(ssi)

    psnr,ssi= test_model_on_data(model_name='pinv-net_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order='lf_70',alpha=alpha,img_size=128,verbose=False)
    psnr_mean[1,index]=np.mean(psnr)
    ssim_mean[1,index]=np.mean(ssi)
    psnr_std[1,index]=np.std(psnr)
    ssim_std[1,index]=np.std(ssi)


#%% 
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2 rows, 2 columns
legend=["70_lf","low_freq"]
# Accessing each subplot
axes[0,0].plot(alpha_list, psnr_mean[0,:])  # Top-left
axes[0,0].plot(alpha_list, psnr_mean[1,:])  # Top-left
axes[0,0].set_title("PSNR Mean")
axes[0,0].set_xlabel("Noise level")
axes[0,0].set_ylim(10,20)
axes[0,0].legend(legend)

axes[0,1].plot(alpha_list, ssim_mean[0,:])  # Top-right
axes[0,1].plot(alpha_list, ssim_mean[1,:])  # Top-right
axes[0,1].set_title("SSIM Mean")
axes[0,1].set_xlabel("Noise level")
axes[0,1].set_ylim(0,1)
axes[0,1].legend(legend)


axes[1,0].plot(alpha_list, psnr_std[0,:])  # Bottom-left
axes[1,0].plot(alpha_list, psnr_std[1,:])  # Bottom-left
axes[1,0].set_title("PSNR STD")
axes[1,0].set_ylim(0,3)
axes[1,0].set_xlabel("Noise level")
axes[1,0].legend(legend)


axes[1,1].plot(alpha_list, ssim_std[0,:])  # Bottom-right
axes[1,1].plot(alpha_list, ssim_std[1,:])  # Bottom-right
axes[1,1].set_title("SSIM STD")
axes[1,1].set_xlabel("Noise level")
axes[1,1].set_ylim(0,1)
axes[1,1].legend(legend)
# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# %% 
alpha =10
number_of_orders=6
number_of_models=5
model_names=["Low Frequency with WR","Variance with WR",'70_lfcorr with WR','70_LF with WR','70_LF without WR']
psnr_mean_table=np.zeros((number_of_models,number_of_orders))
psnr_std_table=np.zeros((number_of_models,number_of_orders))
ssim_mean_table=np.zeros((number_of_models,number_of_orders))
ssim_std_table=np.zeros((number_of_models,number_of_orders))
pattern_list=['low_freq','70_lf','70_lfcorr','variance','random','high_freq']
#%%first training
for i in range(0,number_of_orders):
    psnr,ssi,_,_=test_model_on_data(model_name='pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=pattern_list[i],alpha=alpha,img_size=64,verbose=False, nb_images=100)
    psnr_mean_table[0,i]=np.mean(psnr)
    ssim_mean_table[0,i]=np.mean(ssi)
    psnr_std_table[0,i]=np.std(psnr)
    ssim_std_table[0,i]=np.std(ssi)
#%% Second training
for i in range(0,number_of_orders):
    psnr,ssi,_,_=test_model_on_data(model_name='pinv-net_variance_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=pattern_list[i],alpha=alpha,img_size=64,verbose=False, nb_images=100)
    psnr_mean_table[1,i]=np.mean(psnr)
    ssim_mean_table[1,i]=np.mean(ssi)
    psnr_std_table[1,i]=np.std(psnr)
    ssim_std_table[1,i]=np.std(ssi)
#%% Second training
for i in range(0,number_of_orders):
    psnr,ssi,_,_=test_model_on_data(model_name='pinv-net_70_lfcorr_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=pattern_list[i],alpha=alpha,img_size=64,verbose=False, nb_images=100)
    psnr_mean_table[2,i]=np.mean(psnr)
    ssim_mean_table[2,i]=np.mean(ssi)
    psnr_std_table[2,i]=np.std(psnr)
    ssim_std_table[2,i]=np.std(ssi)
#%% Fourth training
for i in range(0,number_of_orders):
    psnr,ssi,_,_=test_model_on_data(model_name='right_noise_level_pinv-net_Unet_stl10_N0_10_N_64_M_1024_epo_30_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=pattern_list[i],alpha=alpha,img_size=64,verbose=False, nb_images=100)
    psnr_mean_table[3,i]=np.mean(psnr)
    ssim_mean_table[3,i]=np.mean(ssi)
    psnr_std_table[3,i]=np.std(psnr)
    ssim_std_table[3,i]=np.std(ssi)
#%% Fifth training
for i in range(0,number_of_orders):
    psnr,ssi,_,_=test_model_on_data(model_name='pinv-net_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order=pattern_list[i],alpha=alpha,img_size=64,verbose=False, nb_images=100)
    psnr_mean_table[4,i]=np.mean(psnr)
    ssim_mean_table[4,i]=np.mean(ssi)
    psnr_std_table[4,i]=np.std(psnr)
    ssim_std_table[4,i]=np.std(ssi)
# %%
plt.figure(figsize=(8,6))
plt.subplot(1,2,1)
plt.plot(psnr_mean_table[0,:],'-o')
plt.plot(psnr_mean_table[1,:],'-o')
plt.plot(psnr_mean_table[2,:],'-o')
plt.plot(psnr_mean_table[3,:],'-o')
plt.plot(psnr_mean_table[4,:],'-o')

plt.xticks([0,1,2,3,4,5],pattern_list,rotation=45,ha='right')
plt.xlabel(r'Acquisition Order')
plt.ylabel('PSNR')
plt.ylim(10,28)
plt.legend(model_names)
plt.title('PSNR for different acquisition orders')
plt.grid()
plt.tight_layout()
# plt.show()

# plt.figure()
plt.subplot(1,2,2)
plt.plot(ssim_mean_table[0,:],'-o')
plt.plot(ssim_mean_table[1,:],'-o')
plt.plot(ssim_mean_table[2,:],'-o')
plt.plot(ssim_mean_table[3,:],'-o')
plt.plot(ssim_mean_table[4,:],'-o')
plt.xticks([0,1,2,3,4,5],pattern_list,rotation=45,ha='right')
plt.xlabel('Acquisition Order')
plt.ylabel('SSIM')
plt.ylim(0,1)
plt.legend(model_names)
plt.title('SSIM for different acquisition orders')
plt.grid()
plt.tight_layout()
plt.show()

# %% Reconstructed image
test_model_on_data(model_name='pinv-net_BF_Unet_weight_decay_stl10_N0_10_N_64_M_1024_epo_50_lr_0.001_sss_10_sdr_0.5_bs_256.pth',pattern_order='70_lfcorr',alpha=10,img_size=64,verbose=True, nb_images=1)
