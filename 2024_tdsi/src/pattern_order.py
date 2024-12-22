from pathlib import Path
import numpy as np
from spyrit.misc.statistics import Cov2Var
import math
import os


image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics

# Full paths
image_folder_full = Path.cwd() / Path(image_folder)
model_folder_full = Path.cwd() / Path(model_folder)
stat_folder_full  = Path.cwd() / Path(stat_folder)

def choose_pattern_order(order_name, img_size):
    np.random.seed(seed=0)
    print('stat_folder_full',stat_folder_full)
    M = img_size ** 2 // 4  # Number of measurements (1/4 of the pixels)
    if order_name == 'low_freq':
        M_xy = math.ceil(M**0.5)
        Ord_rec = np.ones((img_size, img_size))

        Ord_rec[:,M_xy:] = 0
        Ord_rec[M_xy:,:] = 0

    elif order_name == 'naive':
        Ord_rec = np.ones((img_size, img_size))

    elif order_name == 'variance':
        if img_size == 128:
             cov_name = f'Cov_8_{img_size}x{img_size}.npy'
        else:
            cov_name = f'Cov_{img_size}x{img_size}.npy'

        #cov_path =  os.path.join(stat_folder_full, cov_name)
        #print(f"Loading covariance matrix from: {cov_path}")
        #Cov = np.load(cov_path)
        Cov = np.ones((img_size, img_size))
        print(f"Cov matrix {cov_name} loaded")
        print ("the size of the cov matrix", Cov.shape)
        Ord_rec = Cov2Var(Cov)

    elif order_name == 'random':
    #     set random pixel of the Ord_rec to 1
        # Initialize a 64x64 matrix with zeros
        matrix = np.zeros((img_size, img_size), dtype=int)
        
        # Flatten the matrix to work with indices
        flat_indices = np.arange(matrix.size)  # Create an array of indices [0, 1, ..., 4095]
        
        # Randomly choose M_xy**2 unique indices to be set to 1
        M_xy = math.ceil(M**0.5)
        random_indices = np.random.choice(flat_indices, size=M_xy**2, replace=False)
        
        # Set the chosen indices to 1
        matrix.flat[random_indices] = 1
        
        # Verify the result
        print("Matrix shape:", matrix.shape)
        print("Number of elements set to 1:", np.sum(matrix))        
        Ord_rec = matrix
    elif order_name == 'random_variance':
    #     # TODO 
        # set a recontruction in high frequencies
         # Initialize a 64x64 matrix with zeros
        matrix = np.zeros((img_size, img_size), dtype=int)
         # Flatten the matrix to work with indices
        flat_indices = np.arange(matrix.size)  # Create an array of indices [0, 1, ..., 4095]
        
    elif order_name=='high_freq':
        M_xy = math.ceil(M**0.5)
        Ord_rec = np.ones((img_size, img_size))

        Ord_rec[:,:M_xy] = 0
        Ord_rec[:M_xy,:] = 0



    
    elif order_name == '70_lf':
        # M_xy = math.ceil(M**0.5)
        quad_size=int(img_size/2)
        print("quad_size",quad_size)
        first_quadrant= np.zeros((quad_size,quad_size))
        second_quadrant= np.zeros((quad_size,quad_size))
        third_quadrant= np.zeros((quad_size,quad_size))
        fourth_quadrant= np.zeros((quad_size,quad_size))
        S=first_quadrant.size
        first_ones_to_keep=int(M*0.7)
        second_ones_to_keep=int(M*0.1)
        third_ones_to_keep=int(M*0.1)
        fourth_ones_to_keep=int(M*0.1)
        indices1 = np.random.choice(S, first_ones_to_keep, replace=False)
        indices2 = np.random.choice(S, second_ones_to_keep, replace=False)
        indices3 = np.random.choice(S, third_ones_to_keep, replace=False)
        indices4 = np.random.choice(S, fourth_ones_to_keep, replace=False)
        
        # first_quadrant.flatten()[indices1]=1

        # Modify the original array using its flat iterator
        first_quadrant.flat[indices1] = 1
        second_quadrant.flat[indices2] = 1
        third_quadrant.flat[indices3] = 1
        fourth_quadrant.flat[indices4] = 1

        # Initialize the full image
        Ord_rec = np.zeros((img_size, img_size))

        Ord_rec[:quad_size,:quad_size] = first_quadrant
        Ord_rec[:quad_size,quad_size:] = second_quadrant
        Ord_rec[quad_size:,:quad_size] = third_quadrant
        Ord_rec[quad_size:,quad_size:] = fourth_quadrant

    # elif order_name == 'random_variance_3':
    #     # TODO

    else:
        print('Order name is invalid')
        exit(1)

    return Ord_rec
