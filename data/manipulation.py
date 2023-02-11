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


def read_dicom(image_path: str) -> pydicom.dataset.FileDataset:  # type: ignore
    """
    read_dicom: Function to read the DICOM image.

    Function reads the DICOM image and returns the data extracted in the form
    of a pydicom object. The object contains metadata of the DICOM image and
    also the image data representation in a pixelarray data. The pixelarray
    contains the image in the form of a numpy array.

    Parameters
    ----------
    image_path : str
        The path of the DICOM data in the folder that needs to be read.

    Returns
    -------
    _type_
        _description_
    """
    dicom_image = pydicom.dcmread(image_path)
    return dicom_image


def write_image(image_data: np.ndarray, path: str, name: str) -> None:
    """
    write_image: Writes a matrix of image data into a given folder location.

    Parameters
    ----------
    image_data : np.ndarray
        Image data represented in numpy array format.
    path : str
        Location where the image needs to be stored.
    name : str
        Image identifier that is associated with the data.
    """
    cv2.imwrite(os.path.join(path, name), image_data)
