import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

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
        self.dataset_path: str = dataset_path
        self.dataset_name: str = dataset_name
        self.type: str = type

        # Logging of information
        logger.info(f"Dataset statistics:")
        dataset_location: str = os.path.join(dataset_path, dataset_name)

        files: list = os.listdir(dataset_location)
        for file in files:

            if type.lower() == "classification":
                if (
                    os.path.isdir(os.path.join(dataset_path, file))
                    and file.lower() == "train"
                ):
                    # Logging of information
                    logger.info("Train dataset distribution")

                    categories: list = os.listdir(
                        os.path.join(dataset_path, file)
                    )

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(dataset_path, file, category)
                            )
                        )

                        # Logging of information
                        logger.info(f"{category.capitalize()}: {file_count}")

                elif (
                    os.path.isdir(os.path.join(dataset_path, file))
                    and file.lower() == "test"
                ):
                    # Logging of information
                    logger.info("Test dataset distribution")
                    categories: list = os.listdir(
                        os.path.join(dataset_path, file)
                    )

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(dataset_path, file, category)
                            )
                        )

                        # Logging of information
                        logger.info(f"{category.capitalize()}: {file_count}")

                elif (
                    os.path.isdir(os.path.join(dataset_path, file))
                    and "val" in file.lower()
                ):
                    # Logging of information
                    logger.info("Validation dataset distribution")
                    categories: list = os.listdir(
                        os.path.join(dataset_path, file)
                    )

                    for category in categories:
                        file_count: int = len(
                            os.listdir(
                                os.path.join(dataset_path, file, category)
                            )
                        )

                        # Logging of information
                        logger.info(f"{category.capitalize()}: {file_count}")
