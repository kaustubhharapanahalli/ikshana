import logging
import os

import cv2
import numpy as np
import pydicom

logger = logging.getLogger(__name__)


def read_image(image_path: str, image_name: str) -> np.ndarray:
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
    # read image using opencv imread function. The image path is provided using
    # os.path.join function to read the image from the path directly. The image
    # formats are expected to be the ones supported by opencv function. More
    # details can be found related to formats supported in the link provided
    # below:
    # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56
    image = cv2.imread(os.path.join(image_path, image_name))
    logger.debug(f"Image {image_name} read successfully")
    return image


def read_dicom(image_path: str, image_name: str) -> pydicom.dataset.FileDataset:  # type: ignore
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
    # read dicom image using pydicom.dcmread function. The module pydicom is
    # aimed to support working with dicom files. More details related to the
    # the pydicom library can be found here: https://pydicom.github.io/
    # The documentation for the function used - dcmread - can be found here:
    # https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.filereader.dcmread.html#pydicom.filereader.dcmread

    dicom_image = pydicom.dcmread(os.path.join(image_path, image_name))
    logger.debug(f"Image {image_name} read successfully")
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
    # Images are written into a file using opencv imwrite function. The
    # documentation of the function can be found here:
    # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
    cv2.imwrite(os.path.join(path, name), image_data)
    logger.debug(f"Image {name} written successfully")
