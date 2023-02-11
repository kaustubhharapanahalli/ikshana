import cv2
import numpy as np
import pydicom
import os


def read_image(image_path: str) -> np.ndarray:
    """
    read_image: Reads images which are encoded in JPG, JPEG or PNG format.

    Parameters
    ----------
    image_path : str
        The path of the image in the folder that needs to be read.

    Returns
    -------
    np.ndarray
        Image once read, it is returned as a numpy array data.
    """
    image = cv2.imread(image_path)
    return image
