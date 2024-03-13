# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:48:35 2022

This scripts reconstructs the images in Figure 8

NB (15-Sep-22): to debug needs to run
import collections
collections.Callable = collections.abc.Callable

"""

#%%
import os
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
from pathlib import Path
# get debug in spyder
import collections
collections.Callable = collections.abc.Callable

import torchvision

from spyrit.misc.statistics import Cov2Var
from spyrit.core.noise import Poisson 
from spyrit.core.meas import HadamSplit
from spyrit.core.prep import SplitPoisson
from spyrit.core.recon import DCNet, PinvNet, LearnedPGD
from spyrit.core.train import load_net
from spyrit.core.nnet import Unet, ConvNet, Identity
from spyrit.misc.sampling import reorder, Permutation_Matrix
from spyrit.misc.disp import add_colorbar, noaxis


from spas import read_metadata, spectral_slicing

torch.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Torch device: {device}')

#%% user-defined
# used for acquisition
N_acq = 64

# for reconstruction
N_rec = 128  # 128 or 64
M_list = [4096] #[4096, 1024, 512] # for N_rec = 128
#N_rec = 64
#M_list = [1024]

N0 = 10     # Check if we used 10 in the paper
stat_folder_rec = Path('../../stat/oe_paper/') # Path('../../stat/ILSVRC2012_v10102019/')

mode_sim = True # Reconstruct simulated images in addition to exp

net_arch    = 'lpgd'      # ['dc-net','pinv-net', 'lpgd']
net_denoi   = 'I'        # ['unet', 'cnn', 'drunet', 'P0', 'I']
net_data    = 'imagenet'    # 'imagenet'
bs = 256

# LPGD Variations 
log_fidelity = False
step_estimation = False
wls = True
lpgd_iter = 30

# limits for plotting images
vmin = -1
vmax = 1 

save_root = Path('../../recon/')

# Network paths
if net_arch == 'pinv-net':
    model_path = "../../model"    
    model_name = 'pinv-net_unet_imagenet_N0_10_m_hadam-split_N_128_M_4096_epo_30_lr_0.001_sss_10_sdr_0.5_bs_512_reg_1e-07'
elif net_arch == 'dc-net':
    # Load trained DC-Net
    model_path = '../../model/oe_paper/' 
    #model_name = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}' # Defined later
elif net_arch == 'drunet':
    model__path = "../../model"
    model_name = 'drunet_gray.pth'

# Name save
name_save_details = f'{net_arch}_{net_denoi}'
if net_arch == 'lpgd':
    name_save_details = name_save_details + f'_it{lpgd_iter}'
    if wls:
        name_save_details = name_save_details + '_wls'
# ---------------------------------------------------------
# Reconstruction functions
def init_denoi(net_denoi):
    if net_denoi == 'unet':
        denoi = Unet()
    elif net_denoi == 'cnn':
        denoi = ConvNet() 
    elif net_denoi == 'drunet':
        from drunet import DRUNet
        denoi = denoi.to(device)
        denoi = DRUNet()          
        denoi.load_state_dict(torch.load(os.path.join(model_path, model_name)), strict=False) 
        # load_net(os.path.join(model_path, model_name), denoi, device, strict = False)  
    elif net_denoi == 'P0':
        from spyrit.core.nnet import ProjectToZero
        denoi = ProjectToZero()
    elif net_denoi == 'I':
        denoi = Identity()
    return denoi    

def init_reconstruction_network(noise, prep, Cov_rec, net_arch, net_denoi = None):
    # Denoiser
    if net_denoi:
        denoi = init_denoi(net_denoi)

    # Reconstruction network
    if net_arch == 'dc-net':
        model = DCNet(noise, prep, Cov_rec, denoi)
        if net_denoi:
           load_net(os.path.join(model_path, model_name), model, device, strict = False)
    elif net_arch == 'pinv-net':
        model = PinvNet(noise, prep, denoi)
    elif net_arch == 'lpgd':
        model = LearnedPGD(noise, 
                              prep, 
                              iter_stop = lpgd_iter, 
                              wls=wls,
                              step_estimation=step_estimation,
                              gt=x_gt)
    model.eval()    # Mandantory when batchNorm is used
    model.to(device)
    return model

# Reconstruction: set attributes and reconstruct
def reconstruct(model, y, device, log_fidelity= False, step_estimation = False, wls = False):
    with torch.no_grad():
        # Not all models have these attributes
        if step_estimation:
            model.step_estimation = step_estimation
        if log_fidelity:
            model.log_fidelity = log_fidelity
        if wls:
            model.wls = wls

        rec_sim_gpu = model.reconstruct(y.to(device))
        
        if log_fidelity:
            data_fidelity = model.cost
        else:
            data_fidelity = None
        if hasattr(model, 'mse'):
            mse = model.mse
        else:
            mse = None
        
        rec_sim = rec_sim_gpu.cpu().detach().numpy().squeeze()
        rec_sim = rec_sim.reshape(N_rec, N_rec)
    return rec_sim, data_fidelity, mse

# ---------------------------------------------------------
#%% Parameters for simulated images 
def transform_gray_norm(img_size, crop_type = 'center'): 
    """ 
    Args:
        img_size=int, image size
    
    Create torchvision transform for natural images (stl10, imagenet):
    convert them to grayscale, then to tensor, and normalize between [-1, 1]
    """

    if crop_type=='center':
        transforms_resize = [torchvision.transforms.Resize(img_size), 
                            torchvision.transforms.CenterCrop(img_size)]                           

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.functional.to_grayscale,
        *transforms_resize,
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])])
    return transform

if mode_sim:
    path_natural_images = '../../images'

    # Create dataset and loader (expects class folder 'images/test/')
    #from spyrit.misc.statistics import transform_gray_norm
    transform = transform_gray_norm(N_rec)
    dataset = torchvision.datasets.ImageFolder(root=path_natural_images, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 7)

    x, _ = next(iter(dataloader))
    print(f'Shape of input images: {x.shape}')

    # Select image
    img_id = 4

    x = x[img_id:img_id+1,:,:,:]
    x = x.detach().clone()
    b,c,h,w = x.shape
    x_gt = np.copy(x)

    # plot
    x_plot = x.view(-1,h,h).cpu().numpy() 
    fig = plt.figure(figsize=(7,7))
    im = plt.imshow(x_plot[0,:,:], cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    add_colorbar(im, 'bottom')

    full_path = save_root / (f'sim{img_id}_{N_rec}' + '_gt.pdf')
    fig.savefig(full_path, bbox_inches='tight', dpi=600)

# ---------------------------------------------------------
#%% covariance matrix and network filnames
if N_rec==64:
    cov_rec_file= stat_folder_rec/ ('Cov_{}x{}'.format(N_rec, N_rec)+'.npy')
elif N_rec==128:
    cov_rec_file= stat_folder_rec/ ('Cov_8_{}x{}'.format(N_rec, N_rec)+'.npy')
    
#%% Networks
for M in M_list:    
    if (N_rec == 128) and (M == 4096):
        net_order   = 'rect'
    else:
        net_order   = 'var'

    net_suffix  = f'N0_{N0}_N_{N_rec}_M_{M}_epo_30_lr_0.001_sss_10_sdr_0.5_bs_{bs}_reg_1e-07_light'
    
    if net_arch == 'dc-net':
        model_name = f'{net_arch}_{net_denoi}_{net_data}_{net_order}_{net_suffix}'

    #%% Init and load trained network
    # Covariance in hadamard domain
    Cov_rec = np.load(cov_rec_file)
    
    # Sampling order
    if net_order == 'rect':
        Ord_rec = np.ones((N_rec, N_rec))
        n_sub = math.ceil(M**0.5)
        Ord_rec[:,n_sub:] = 0
        Ord_rec[n_sub:,:] = 0
        
    elif net_order == 'var':
        Ord_rec = Cov2Var(Cov_rec)

    name_save = f'sim{img_id}_{N_rec}_N0_{N0}_M_{M}_{net_order}'        
    # ---------------------------------------------------------
    # Init network  
    meas = HadamSplit(M, N_rec, Ord_rec)
    noise = Poisson(meas, N0) # could be replaced by anything here as we just need to recon
    prep  = SplitPoisson(N0, meas)    

    model = init_reconstruction_network(noise, prep, Cov_rec, net_arch, net_denoi)
    # ---------------------------------------------------------        
    #%% simulations
    if mode_sim:
        x = x.view(b * c, h * w)
        y = noise(x.to(device))
    
    if mode_sim:
        with torch.no_grad():
            rec_sim, data_fidelity, mse = reconstruct(model, y, device, log_fidelity, step_estimation, wls)
        
        fig , axs = plt.subplots(1,1)
        #im = axs.imshow(rec_sim, cmap='gray', vmin=vmin, vmax=vmax)
        im = axs.imshow(rec_sim, cmap='gray')
        noaxis(axs)
        add_colorbar(im, 'bottom')

        if 'name_save_details' in globals():
            name_save = name_save + '_' + name_save_details
        full_path = save_root / (name_save + '.pdf')
        fig.savefig(full_path, bbox_inches='tight', dpi=600)
        # 
        if log_fidelity:
            #np.linalg.norm(x_gt-rec_sim)/np.linalg.norm(x_gt)
            # Data fidelity
            fig=plt.figure(); plt.plot(data_fidelity, label='GD')
            plt.ylabel('Data fidelity')
            plt.xlabel('Iterations')
            full_path = save_root / (name_save + '_data_fidelity.png')
            fig.savefig(full_path, bbox_inches='tight', dpi=600)
        if hasattr(model, 'mse'):
            # MSE
            mse = np.array(mse)/np.linalg.norm(x_gt)
            fig=plt.figure(); plt.plot(mse, label='GD')
            plt.ylabel('NMSE')
            plt.xlabel('Iterations')
            # yaxis from 0 to 10
            plt.ylim(0,1)
            full_path = save_root / (name_save + '_nmse.png')
            fig.savefig(full_path, bbox_inches='tight', dpi=600)


    #%% Load expe data and unsplit
    data_root = Path('../../data/')

    data_file_prefix_list = ['zoom_x12_usaf_group5',
                             'zoom_x12_starsector',
                             'tomato_slice_2_zoomx2',
                             'tomato_slice_2_zoomx12',
                             ]
    
    
    #%% Load data
    for data_file_prefix in data_file_prefix_list:
        
        print(Path(data_file_prefix) / data_file_prefix)
        
        # meta data
        meta_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_metadata.json')
        _, acquisition_parameters, _, _ = read_metadata(meta_path)
        wavelengths = acquisition_parameters.wavelengths 
        
        # data
        full_path = data_root / Path(data_file_prefix) / (data_file_prefix + '_spectraldata.npz')
        raw = np.load(full_path)
        meas= raw['spectral_data']
        
        # reorder measurements to match with the reconstruction order
        Ord_acq = -np.array(acquisition_parameters.patterns)[::2]//2   # pattern order
        Ord_acq = np.reshape(Ord_acq, (N_acq,N_acq))                   # sampling map
        
        Perm_rec = Permutation_Matrix(Ord_rec)    # from natural order to reconstrcution order 
        Perm_acq = Permutation_Matrix(Ord_acq).T  # from acquisition to natural order
        meas = reorder(meas, Perm_acq, Perm_rec)
        
        #%% Reconstruct a single spectral slice from full reconstruction
        wav_min = 579 
        wav_max = 579.1
        wav_num = 1
        meas_slice, wavelengths_slice, _ = spectral_slicing(meas.T, 
                                                        wavelengths, 
                                                        wav_min, 
                                                        wav_max, 
                                                        wav_num)
        with torch.no_grad():
            m = torch.Tensor(meas_slice[:2*M,:]).to(device)
            
            if True:  # all methods?
                rec_gpu = model.reconstruct_expe(m)
                rec = rec_gpu.cpu().detach().numpy().squeeze()
            
                #%% Plot or save 
                # rotate
                #rec = np.rot90(rec,2)
                
                fig , axs = plt.subplots(1,1)
                im = axs.imshow(rec, cmap='gray')
                noaxis(axs)
                add_colorbar(im, 'bottom')
                
                full_path = save_root / (data_file_prefix + '_' + f'{M}_{N_rec}' + f'_{name_save_details}' + '.pdf')
                fig.savefig(full_path, bbox_inches='tight')  
        
            
    
            
        
