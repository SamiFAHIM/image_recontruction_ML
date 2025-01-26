# %%
import os
from pathlib import Path
import numpy as np
from PIL import Image

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

# %%
def read_all_images(path_to_data, num_images=None):
    """
    Reads a specified number of images from a binary dataset file.
    
    :param path_to_data: Path to the binary file containing the images
    :param num_images: Number of images to read (default is all images)
    :return: Array containing the selected images
    """
    with open(path_to_data, 'rb') as f:
        # Read the entire binary file
        everything = np.fromfile(f, dtype=np.uint8)

        # Reshape into images (channels, height, width)
        images = np.reshape(everything, (-1, 3, 96, 96))
        
        # Transpose to standard format (batch, height, width, channels)
        images = np.transpose(images, (0, 3, 2, 1))
        
        # If num_images is specified, limit the number of images
        if num_images is not None:
            images = images[:num_images]
        
        return images

# %%
def save_images(images, output_folder):
    """
    Saves the given images as PNG files in the specified directory.
    
    :param images: Array of images to save
    :param output_folder: Path to the directory where images will be saved
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    for idx, img_array in enumerate(images):
        img = Image.fromarray(img_array)
        img.save(output_folder / f"image_{idx:04d}.png")  # Save with a zero-padded filename

# %%
input_image_file = 'data_model_training/stl10_binary/test_X.bin'  # Path to the binary image file
output_image_folder = 'data/images/cropped'  # Folder where images will be saved

# Specify the number of images to read and save
num_images_to_save = 100  # Change this to the desired number of images

# Read and save images
input_path = Path.cwd() / input_image_file
output_path = Path.cwd() / output_image_folder

images = read_all_images(input_path, num_images=num_images_to_save)  # Read a subset of images
save_images(images, output_path)  # Save images to the specified folder

print(f"{num_images_to_save} images saved to: {output_path}")
