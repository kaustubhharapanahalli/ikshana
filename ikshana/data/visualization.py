import logging
import os
import sys
from typing import Dict

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
        self.type: str = type
        self.train, self.test, self.val = "", "", ""

        # Logging of information
        logger.info(f"Dataset statistics:")
        self.dataset_location: str = os.path.join(dataset_path, dataset_name)

        files: list = os.listdir(self.dataset_location)
        for file in files:
            # If dataset is for a classification problem, then the following
            # condition will be used to provide the dataset statistics for
            # each category.
            if type.lower() == "classification":
                if (
                    os.path.isdir(os.path.join(self.dataset_location, file))
                    and file.lower() == "train"
                ):
                    self.train = file
                    # Logging of information
                    logger.info("Train dataset distribution")
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
                    logger.info(f"Total images in train: {total_images}")
                    category_separation: str = (
                        "Categorical separation for each category - "
                    )
                    for key, val in categories_dict.items():
                        category_separation += f"{key}: {val}; "
                    logger.info(category_separation)

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
