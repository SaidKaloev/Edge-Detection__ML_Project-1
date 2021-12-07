"""
Author: Said Kaloev
Exercise 5: Machine Learning Project
"""

import os
import glob
import numpy as np
import dill as pkl
import gzip

from random import randint
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

def Image_Reader(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    """Creates target, known and input array for model"""
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError("image_array must be a 2D numpy array")

    border_x_start, border_x_end = border_x
    border_y_start, border_y_end = border_y

    try:  # Check for conversion to int (would raise ValueError anyway but we will write a nice error message)
        border_x_start = int(border_x_start)
        border_x_end = int(border_x_end)
        border_y_start = int(border_y_start)
        border_y_end = int(border_y_end)
    except ValueError as e:
        raise ValueError(f"Could not convert entries in border_x and border_y ({border_x} and {border_y}) to int! "
                         f"Error: {e}")
    if border_x_start < 1 or border_x_end < 1:
        raise ValueError(f"Values of border_x must be greater than 0 but are {border_x_start, border_x_end}")
    if border_y_start < 1 or border_y_end < 1:
        raise ValueError(f"Values of border_y must be greater than 0 but are {border_y_start, border_y_end}")
    remaining_size_x = image_array.shape[0] - (border_x_start + border_x_end)
    remaining_size_y = image_array.shape[1] - (border_y_start + border_y_end)
    if remaining_size_x < 16 or remaining_size_y < 16:
        raise ValueError(f"the size of the remaining image after removing the border must be greater equal (16,16) "
                         f"but was ({remaining_size_x},{remaining_size_y})")
    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x_start:-border_x_end, border_y_start:-border_y_end] = 1
    # Create target_array - don't forget to use .copy(), otherwise target_array and image_array might point to the
    # same array!
    target_array = image_array[known_array == 0].copy()
    # Use image_array as input_array
    image_array[known_array == 0] = 0
    return image_array, known_array, target_array

def resize_transformer(filename):
    """This function gets an image, reshapes it with transforms and gives the output_image back"""
    im_shape = 90
    resize_transforms = transforms.Compose([transforms.Resize(size=im_shape),transforms.CenterCrop(size=(im_shape, im_shape)),])
    image = Image.open(filename)
    image = resize_transforms(image)
    return image

def data_preparator(input_path):
    """This function gets all jpg-images and transforms them to a pickel_file, to train later on"""
    image_files = sorted(glob.glob(os.path.join(input_path, "**", "*.jpg"), recursive=True))
    array = []
    with gzip.open(f"prep_files.pklz", "w") as fh:
        for i, images in tqdm(enumerate(image_files), desc="Preprocessing files", total=len(image_files), colour="green"):
            image = resize_transformer(images)
            image = np.array(image)
            x_border = randint(5,9)
            y_border = randint(5,9)
            copied_image = image.copy()
            img, known, target = Image_Reader(image, (x_border, 14-x_border), (y_border, 14-y_border))
            array.append((copied_image, img, known))

        print("Waiting to finish process.........")
        pkl.dump(dict(array=array), file=fh)
        print("Finished process!")
data_preparator("/home/said/Schreibtisch/2.Semester/Python/1.PythonChallenge/dataset")
