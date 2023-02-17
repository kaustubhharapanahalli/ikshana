#!bin/bash/python3
# Warm-up exercise 1

import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from ikshana.data.manipulation import read_dicom, read_image, write_image


def check_read_image(
    image_path: str, image_name: str, plot_path: str, plot_name: str
) -> np.ndarray:
    """
    check_read_image: Function check read image implementation.

    Parameters
    ----------
    image_path : str
        Path of the image folder
    image_name : str
        name of the image to read
    plot_path : str
        Path where the image should be stored after reading
    plot_name : str
        Name of the plot in which to be saved in the folder

    Returns
    -------
    np.ndarray
        Returns the image in the form of a numpy array
    """
    img_data = read_image(image_path=image_path, image_name=image_name)
    cv2.imwrite(os.path.join(plot_path, plot_name), img_data)

    return img_data


def check_write_image(
    image_data: np.ndarray, image_path: str, image_name: str
) -> None:
    """
    check_write_image: Function to write an image that is read.


    Parameters
    ----------
    image_data : np.ndarray
        Data of the PNG or JPG image read.
    image_path : str
        Path of the location where to write image.
    image_name : str
        Name of the image to be written in.
    """
    write_image(image_data=image_data, path=image_path, name=image_name)


def check_read_dicom(
    image_path: str, image_name: str, plot_path: str, plot_name: str
) -> pydicom.dataset.FileDataset:  # type: ignore
    img_data = read_dicom(image_path=image_path, image_name=image_name)
    plt.imsave(os.path.join(plot_path, plot_name), img_data.pixel_array)
    return img_data


if __name__ == "__main__":
    jpg_path = os.path.join("datasets", "Practice_PNGandJPG", "chestimage_JPG")
    jpg_name = "JPCLN001.jpg"
    jpg_img_data = check_read_image(
        jpg_path, jpg_name, "outputs", "read_jpg.jpg"
    )

    png_path = os.path.join("datasets", "Practice_PNGandJPG", "chestimage_PNG")
    png_name = "JPCLN001.png"
    png_img_data = check_read_image(
        png_path, png_name, "outputs", "read_png.png"
    )

    check_write_image(jpg_img_data, "outputs", "jpg_write.jpg")
    check_write_image(png_img_data, "outputs", "png_write.png")

    dicom_path = os.path.join(
        "datasets", "Practice_DICOM", "chestimages_DICOM"
    )
    dicom_name = "JPCLN001.dcm"
    check_read_dicom(dicom_path, dicom_name, "outputs", "dicom_read.png")
