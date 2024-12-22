from pathlib import Path
"""
This file contains the configuration for the project.
the device used for computations, the paths to the data, models and statistics
it also contains the image settings
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

image_folder = 'data/images/'       # images for simulated measurements
model_folder = 'model/'             # reconstruction models
stat_folder  = 'stat/'              # statistics

# Full paths
image_folder_full = Path.cwd().parent.parent.parent / Path(image_folder)
model_folder_full = Path.cwd().parent.parent.parent / Path(model_folder)
stat_folder_full  = Path.cwd().parent.parent.parent / Path(stat_folder)

# Image settings
IMG_SIZE = 64
ALPHA = 10
#under_Sampling_factor = 4
#M = IMG_SIZE ** 2 // under_Sampling_factor  # Number of measurements (1/4 of the pixels)
