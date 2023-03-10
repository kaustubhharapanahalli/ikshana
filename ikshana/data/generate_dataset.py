import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def walk_through_path(folder_path: Union[str, Path]):
    """
    walk_through_path: Walks through dir_path returning its contents.

    Parameters
    ----------
    folder_path : Union[str, Path]
        Location of the dataset.
    """
    for dirpath, dirnames, filenames in os.walk(folder_path):
        logger.debug(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.",
        )


def find_classification_classes(
    data_folder: Union[str, Path],
) -> Tuple[List[str], Dict[str, int]]:
    """
    find_classification_classes: Finds class names in the target data_folder.

    Assumes target data_folder is in standard image classification format.

    Parameters
    ----------
    data_folder : Union[str, Path]
        Location of the target data_folder.

    Returns
    -------
    Tuple[List[str], Dict[str, int]]
        Returns a tuple of list of class names and a dict containing the number
        of objects in each class.

    Raises
    ------
    FileNotFoundError
        If the folder does not have any files within it, then FileNotFoundError
        is raised.

    Example
    -------
        find_classification_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(
        entry.name for entry in os.scandir(data_folder) if entry.is_dir()
    )

    logger.info(f"Classes present: {classes}")

    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {data_folder}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class SingleLabelClassificationDataset(Dataset):
    """
    SingleLabelClassificationDataset: Classification Dataset generator using folders.

    The class generates unique dataset objects for different dataset folders
    containing images in the form of
        |- dataset_name
            |- train
                |- class_1
                |- class_2
                |- class_3
                .
                .
                .
            |- test
                |- class_1
                |- class_2
                |- class_3
                .
                .
                .
    """

    def __init__(self, folder_path: Union[str, Path], transform=None) -> None:
        """
        __init__: Initialize the dataset object

        Parameters
        ----------
        folder_path : Union[str, Path]
            Folder location of the dataset.
        transform : _type_, optional
            If the return object needs to be a transformed output, the
            transform function can be provided, by default None.
        """
        self.paths = list(Path(folder_path).glob("*/*.png"))
        self.transform = transform
        self.classes, self.class_to_index = find_classification_classes(
            folder_path
        )

    def load_image(self, index: int) -> Image.Image:
        """
        load_image: Opens an image in the Image.Image format

        Parameters
        ----------
        index : int
            Index of the image to be opened

        Returns
        -------
        Image.Image
            Opened image is returned as an Image.Image object.
        """
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        """
        __len__: Returns the length of the dataset

        Returns
        -------
        int
            Count of the dataset.
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        __getitem__: Provides individual items when requested as an iterator.

        Parameters
        ----------
        index : int
            Count of the item.

        Returns
        -------
        Tuple[torch.Tensor, int]
            The image as a tensor object and the class index it is associated
            with.
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_index = self.class_to_index[class_name]

        if self.transform:
            return self.transform(img), class_index
        else:
            return torch.tensor(img), class_index
