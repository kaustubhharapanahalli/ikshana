#!bin/bash/python3
# Warm-up exercise 1

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.manipulation import read_dicom, read_image, write_image


def check_read_image(
    image_path: str, image_name: str, plot_path: str, plot_name: str
):
    img_data = read_image(image_path=image_path, image_name=image_name)
    cv2.imwrite(os.path.join(plot_path, plot_name), img_data)

    return img_data


def check_write_image(
    image_data: np.ndarray, image_path: str, image_name: str
):
    pass
    write_image(image_data=image_data, path=image_path, name=image_name)


def check_read_dicom(
    image_path: str, image_name: str, plot_path: str, plot_name: str
):
    img_data = read_dicom(image_path=image_path, image_name=image_name)
    print(img_data)


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
