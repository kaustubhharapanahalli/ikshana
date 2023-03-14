import logging
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ikshana.data.manipulation import read_image

logger = logging.getLogger(__name__)


class Visualize:
    def __init__(
        self, dataset_path: str, dataset_name: str, type: str
    ) -> None:
        """
        __init__: Init function to handle data visualizations.

        Provides the statistics of the data distribution for a select dataset.
        The information provided is about the number of images present for each
        category, how classes are present in the dataset, and many more such
        minute information.

        Parameters
        ----------
        dataset_path : str
            Contains the folder path where the dataset is located at.
        dataset_name: str
            Contains the name of the dataset present in the dataset folder.
        type : str
            Indicates the type of operation being carried out. The
            implementation currently supports the dataset visualization for
            following types of datasets:
                1. Classification - Expects the dataset to have train and test
                   splits mandatorily, if val split is present, that would be
                   great as well. If not, val split will be handled ahead. In
                   each data split folder, the images should be present in
                   their category folders.

        """
        # Initialization of class variables
        self.type: str = type

        # The three variables are initialized for storing the names of the
        # folders where the dataset for each separation are present. This will
        # help in getting images for necessary dataset.
        self.train, self.test, self.val = "", "", ""

        # Logging of information
        logger.info(f"Dataset statistics:")

        # The dataset_location contains the details of the location and the
        # name of the dataset.
        self.dataset_location: str = os.path.join(dataset_path, dataset_name)

        # List of all the files and folder present in the folder.
        files: list = os.listdir(self.dataset_location)
        for file in files:
            # If dataset is for a classification problem, then the following
            # condition will be used to provide the dataset statistics for
            # each category.
            if type.lower() == "classification":
                # The condition is for getting details related to train set in
                # the dataset.
                if (
                    os.path.isdir(os.path.join(self.dataset_location, file))
                    and file.lower() == "train"
                ):
                    # Assigning the folder related to train set
                    self.train = file
                    # Logging of information
                    logger.info("Train dataset distribution")
                    # dictionary to store the details of category to count per
                    # category mapping.
                    categories_dict: Dict[str, int] = {}
                    categories: list = os.listdir(
                        os.path.join(self.dataset_location, file)
                    )
                    self.category_count = len(categories)

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(
                                    self.dataset_location, file, category
                                )
                            )
                        )

                        # Logging of information
                        categories_dict[category.capitalize()] = file_count

                    # The total images variable contains the count of the
                    # total number of images in the training set.
                    total_images = sum(list(categories_dict.values()))
                    logger.info(f"Total images in train: {total_images}")

                    # Logging statement as a string
                    category_separation: str = (
                        "Categorical separation for each category - "
                    )
                    for key, val in categories_dict.items():
                        category_separation += f"{key}: {val}; "
                    logger.info(category_separation)

                # repeat the same steps done for training, for test set.
                elif (
                    os.path.isdir(os.path.join(self.dataset_location, file))
                    and file.lower() == "test"
                ):
                    self.test = file
                    # Logging of information
                    logger.info("Test dataset distribution")
                    categories_dict: Dict[str, int] = {}
                    categories: list = os.listdir(
                        os.path.join(self.dataset_location, file)
                    )
                    self.category_count = len(categories)

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(
                                    self.dataset_location, file, category
                                )
                            )
                        )

                        # Logging of information
                        categories_dict[category.capitalize()] = file_count

                    total_images = sum(list(categories_dict.values()))
                    logger.info(f"Total images in test: {total_images}")
                    category_separation: str = (
                        "Categorical separation for each category - "
                    )
                    for key, val in categories_dict.items():
                        category_separation += f"{key}: {val}; "
                    logger.info(category_separation)

                # repeat the same steps done for training, for validation set,
                # if available.
                elif (
                    os.path.isdir(os.path.join(self.dataset_location, file))
                    and "val" in file.lower()
                ):
                    self.val = file
                    # Logging of information
                    logger.info("Validation dataset distribution")

                    categories_dict: Dict[str, int] = {}
                    categories: list = os.listdir(
                        os.path.join(self.dataset_location, file)
                    )
                    self.category_count = len(categories)

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(
                                    self.dataset_location, file, category
                                )
                            )
                        )

                        # Logging of information
                        categories_dict[category.capitalize()] = file_count

                    total_images = sum(list(categories_dict.values()))
                    logger.info(f"Total images in test: {total_images}")
                    category_separation: str = (
                        "Categorical separation for each category - "
                    )
                    for key, val in categories_dict.items():
                        category_separation += f"{key}: {val}; "
                    logger.info(category_separation)

    def visualize_category_data(
        self, samples_per_category: int, plot_name: str
    ) -> None:
        """
        visualize_category_data: Visualize samples of data for each category.

        Depending on the number of samples to be plotted per category, the
        function will save an image as output in the plots folder showcasing
        all the image samples per category. How many samples per category is
        indicated by the samples_per_category variable and the name of the
        saved plot will be decided based on the plot_names variable.

        Parameters
        ----------
        samples_per_category : int
            Indicates the number of samples to plot per category.
        plot_name : str
            Indicates the name in which the plot will be saved. This does not
            contain the format of the plot.
        """
        rows, columns = self.category_count, samples_per_category
        figure = plt.figure(
            figsize=(4 * samples_per_category, int(2.5 * self.category_count))
        )

        figure_location: int = 1

        for file in os.listdir(
            os.path.join(self.dataset_location, self.train)
        ):
            data_for_category: list = os.listdir(
                os.path.join(self.dataset_location, self.train, file)
            )
            count: int = len(data_for_category)

            choices: torch.Tensor = torch.randint(
                0, count, (samples_per_category,)
            )

            for choice in choices:
                figure.add_subplot(rows, columns, figure_location)
                plt.title(file)
                plt.axis("off")
                img_name: str = data_for_category[choice.item()]  # type: ignore
                imdata: np.ndarray = read_image(
                    os.path.join(
                        self.dataset_location,
                        self.train,
                        file,
                    ),
                    img_name,
                )
                plt.imshow(imdata)

                figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(os.path.join("plots", plot_name + ".png"))
        logger.info(f"Data sample visualization saved as {plot_name}.png")

    @staticmethod
    def visualize_individual_image(
        tensor_data: torch.Tensor,
        class_name: str,
        idx: int,
        grayscale: bool = False,
    ):
        """
        visualize_individual_image: Function to plot a single image output.

        Parameters
        ----------
        tensor_data : torch.Tensor
            Data provided in the format of a tensor from a dataloader.
        class_name : str
            name of the class to be displayed as title of the plot
        idx : int
            index of the image in the dataset to have unique plots for
            individual images.
        grayscale: bool
            Check for plotting the image as a grayscale or color image.
        """
        data = tensor_data.detach().cpu().permute(1, 2, 0).squeeze().numpy()
        plt.figure(figsize=(10, 7))
        if grayscale:
            plt.gray()
        plt.title(class_name)
        plt.plot(data)

    @staticmethod
    def save_model_plots(
        train_accuracy: List[Any],
        validation_accuracy: List[Any],
        train_loss: List[Any],
        valid_loss: List[Any],
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_accuracy,
            color="green",
            linestyle="-",
            label="Train Accuracy",
        )
        plt.plot(
            validation_accuracy,
            color="blue",
            linestyle="-",
            label="Validation Accuracy",
        )

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(os.path.join("plots", "accuracy.png"))

        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss,
            color="yellow",
            linestyle="-",
            label="Train Loss",
        )
        plt.plot(
            valid_loss,
            color="red",
            linestyle="-",
            label="Validation Loss",
        )

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join("plots", "loss.png"))

    @staticmethod
    def plot_correct_incorrect_classifications(dataset, predictions):
        pass
