import logging
import os
import random
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Subset

from ikshana.data.manipulation import read_image
from ikshana.models.test import infer_image
from ikshana.utils.explain import compute_grad_cam

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
        plt.close()

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
        if grayscale:
            plt.gray()
        plt.imsave(
            os.path.join("plots", class_name + "_" + str(idx) + ".png"), data
        )

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
            color="black",
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
    def plot_correct_incorrect_classifications(
        dataset, model_path, model, device, data_split_type, target_layer
    ):
        checkpoint = torch.load(
            os.path.join("checkpoints", "best_model_directions.pth")
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        samples_per_class = 4
        correct_index, incorrect_index = [], []
        correct_plot_indices, incorrect_plot_indices = [], []

        if isinstance(dataset, Subset):
            num_classes = len(dataset.dataset.classes)  # type: ignore
            classes = dict()
            correct_class_check = dict()
            incorrect_class_check = dict()
            for i, cls in enumerate(dataset.dataset.classes):  # type: ignore
                classes[i] = cls
                correct_class_check[i] = 0
                incorrect_class_check[i] = 0
        else:
            num_classes = len(dataset.classes)
            classes = dict()
            correct_class_check = dict()
            incorrect_class_check = dict()
            for i, cls in enumerate(dataset.classes):  # type: ignore
                classes[i] = cls
                correct_class_check[i] = 0
                incorrect_class_check[i] = 0

        for i, data in enumerate(dataset):  # type: ignore
            image, label = data
            image = image.to(device).unsqueeze(dim=0)

            infer_result = infer_image(model.to(device), model_path, image)
            infer_result = infer_result.detach().cpu().squeeze().item()

            if infer_result != label:
                incorrect_index.append([i, label, infer_result])
                incorrect_class_check[label] += 1
            else:
                correct_index.append([i, label, infer_result])
                correct_class_check[label] += 1

        if len(correct_index) < samples_per_class * num_classes:
            correct_plot_indices = correct_index
        else:
            while len(correct_plot_indices) != samples_per_class * num_classes:
                pick = random.choices(correct_index)[0]
                if (
                    pick not in correct_plot_indices
                    and correct_class_check[pick[1]] != samples_per_class
                ):
                    correct_plot_indices.append(pick)
                    correct_index.remove(pick)
                    correct_class_check[pick[1]] += 1

        if len(incorrect_index) < samples_per_class * num_classes:
            incorrect_plot_indices = incorrect_index
        else:
            while (
                len(incorrect_plot_indices) != samples_per_class * num_classes
            ):
                pick = random.choices(incorrect_index)[0]
                if (
                    pick not in incorrect_plot_indices
                    and incorrect_class_check[pick[1]] != samples_per_class
                ):
                    incorrect_plot_indices.append(pick)
                    incorrect_class_check[pick[1]] += 1

        figure = plt.figure(figsize=(4 * samples_per_class, 4 * num_classes))
        figure_location: int = 1

        for data in correct_plot_indices:
            if isinstance(dataset, Subset):
                data_for_category = dataset.dataset[data[0]]
            else:
                data_for_category = dataset[data[0]]

            img, label = data_for_category
            figure.add_subplot(samples_per_class, num_classes, figure_location)
            plt.title(
                f": {classes[data[1]]}, Ground Truth: {classes[data[2]]}"
            )
            plt.axis("off")
            plt.imshow(
                data_for_category[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .squeeze()
                .numpy()
            )

            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join("plots", data_split_type + "_correct_predictions.png")
        )
        plt.close()

        figure = plt.figure(figsize=(4 * samples_per_class, 4 * num_classes))
        figure_location: int = 1
        for data in incorrect_plot_indices:
            if isinstance(dataset, Subset):
                data_for_category = dataset.dataset[data[0]]
            else:
                data_for_category = dataset[data[0]]

            img, label = data_for_category
            figure.add_subplot(samples_per_class, num_classes, figure_location)
            plt.title(
                f"Predicted: {classes[data[1]]}, Ground Truth: {classes[data[2]]}"
            )
            plt.axis("off")
            plt.imshow(
                data_for_category[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .squeeze()
                .numpy()
            )

            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join(
                "plots", data_split_type + "_incorrect_predictions.png"
            )
        )
        plt.close()

        figure = plt.figure(figsize=(7 * num_classes, 4 * samples_per_class))
        figure_location: int = 1

        for idx, label, pred in correct_plot_indices:
            image, label = dataset[idx]
            figure.add_subplot(
                samples_per_class * 2, num_classes * 2, figure_location
            )
            plt.title(
                f"Prediction: {classes[label]}, Ground Truth: {classes[pred]}"
            )
            plt.axis("off")
            plt.imshow(image.detach().cpu().permute(1, 2, 0).squeeze().numpy())

            figure_location += 1

            grad_cam_output = compute_grad_cam(
                model.to("cpu"), target_layer, image.to("cpu").unsqueeze(dim=0)
            )
            figure.add_subplot(
                samples_per_class * 2, num_classes * 2, figure_location
            )
            plt.title("Grad-CAM Output")
            plt.axis("off")
            plt.imshow(
                torch.tensor(grad_cam_output).squeeze().numpy(), cmap="plasma"
            )

            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join(
                "plots",
                data_split_type + "_separate_gradcam_correct_predictions.png",
            )
        )

        plt.close()

        figure = plt.figure(figsize=(7 * num_classes, 4 * samples_per_class))
        figure_location: int = 1

        for idx, label, pred in incorrect_plot_indices:
            image, label = dataset[idx]
            figure.add_subplot(
                samples_per_class * 2, num_classes * 2, figure_location
            )
            plt.title(
                f"Prediction: {classes[label]}, Ground Truth: {classes[pred]}"
            )
            plt.axis("off")
            plt.imshow(image.detach().cpu().permute(1, 2, 0).squeeze().numpy())

            figure_location += 1

            grad_cam_output = compute_grad_cam(
                model.to("cpu"), target_layer, image.to("cpu").unsqueeze(dim=0)
            )
            figure.add_subplot(
                samples_per_class * 2, num_classes * 2, figure_location
            )
            plt.title("Grad-CAM Output")
            plt.axis("off")
            plt.imshow(
                torch.tensor(grad_cam_output).squeeze().numpy(), cmap="plasma"
            )
            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join(
                "plots",
                data_split_type
                + "_separate_gradcam_incorrect_predictions.png",
            )
        )
        plt.close()

        figure = plt.figure(figsize=(4 * num_classes, 4 * samples_per_class))
        figure_location: int = 1

        for idx, label, pred in correct_plot_indices:
            image, label = dataset[idx]
            grad_cam_output = compute_grad_cam(
                model.to("cpu"), target_layer, image.to("cpu").unsqueeze(dim=0)
            )
            # Assuming 'gradcam_output' is the output from the Grad-CAM process
            grad_cam_output = np.squeeze(
                grad_cam_output
            )  # Remove the single channel dimension

            # Normalize the output to a range of 0-255 and convert to uint8
            grad_cam_output = cv2.normalize(
                grad_cam_output,
                None,  # type: ignore
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            grad_cam_output = cv2.cvtColor(
                cv2.applyColorMap(
                    grad_cam_output,
                    cv2.COLORMAP_JET,
                ),
                cv2.COLOR_BGR2RGB,
            )
            image = cv2.cvtColor(
                image.permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2RGB
            )

            image = cv2.normalize(
                image,
                None,  # type: ignore
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            masked_image = cv2.addWeighted(
                grad_cam_output,
                0.5,
                image,
                0.5,
                0,
            )
            figure.add_subplot(samples_per_class, num_classes, figure_location)
            plt.title(
                f"Predicted: {classes[label]}, Ground Truth: {classes[pred]}"
            )
            plt.axis("off")
            plt.imshow(masked_image)

            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join(
                "plots",
                data_split_type + "_gradcam_correct_predictions.png",
            )
        )
        plt.close()

        figure = plt.figure(figsize=(4 * num_classes, 4 * samples_per_class))
        figure_location: int = 1

        for idx, label, pred in incorrect_plot_indices:
            image, label = dataset[idx]
            grad_cam_output = compute_grad_cam(
                model.to("cpu"), target_layer, image.to("cpu").unsqueeze(dim=0)
            )
            # Assuming 'gradcam_output' is the output from the Grad-CAM process
            grad_cam_output = np.squeeze(
                grad_cam_output
            )  # Remove the single channel dimension

            # Normalize the output to a range of 0-255 and convert to uint8
            grad_cam_output = cv2.normalize(
                grad_cam_output,
                None,  # type: ignore
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            grad_cam_output = cv2.cvtColor(
                cv2.applyColorMap(
                    grad_cam_output,
                    cv2.COLORMAP_JET,
                ),
                cv2.COLOR_BGR2RGB,
            )
            image = cv2.cvtColor(
                image.permute(1, 2, 0).numpy(), cv2.COLOR_GRAY2RGB
            )

            image = cv2.normalize(
                image,
                None,  # type: ignore
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            masked_image = cv2.addWeighted(
                grad_cam_output,
                0.5,
                image,
                0.5,
                0,
            )
            figure.add_subplot(samples_per_class, num_classes, figure_location)
            plt.title(
                f"Predicted: {classes[label]}, Ground Truth: {classes[pred]}"
            )
            plt.axis("off")
            plt.imshow(masked_image)

            figure_location += 1

            if not os.path.exists("plots"):
                os.mkdir("plots")

        plt.savefig(
            os.path.join(
                "plots",
                data_split_type + "_gradcam_incorrect_predictions.png",
            )
        )
        plt.close()
