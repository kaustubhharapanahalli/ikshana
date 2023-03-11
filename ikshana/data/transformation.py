from typing import Any, List, Tuple, Type, Union

import torch
from torch import nn
from torchvision import transforms


class TransformBase:
    def __init__(
        self,
        resize_image_dimension: int,
        standard_deviation: Union[List[int] | Tuple[int], None] = None,
        mean: Union[List[int] | Tuple[int], None] = None,
        use_augmentation: bool = True,
        augmentation_stratgies: List[Any] | None = [],
    ) -> None:
        """
        __init__: Initialization of necessary parameters.

        The parameters that are essential for transformation of data, be it,
        train, test or validation, such variables are initialized and created
        as class variables.

        Parameters
        ----------
        resize_image_dimension : int
            Size of the image in pixel length to be resized into for easy
            training steps.
        standard_deviation : Union[List[int]  |  Tuple[int], None], optional
            Standard deviation of the dataset is provided which will be
            considered for the normalization of the dataset, by default None.
        mean : Union[List[int]  |  Tuple[int], None], optional
            Mean of the dataset is provided which will be considered for the
            normalization of the dataset, by default None.
        use_augmentation : bool, optional
            If augmentation needs to be used for training and validation is
            mentioned using the use_augmentation value. If true, augmentation
            is applied to train and validation, but not to test. If
            augmentation should be used for test, it is defined in the
            test_transform function, by default True.
        augmentation_strategies : List[Any] | None, optional
            List of augmentation strategies defined using albumentations or
            pytorch transforms, by default [].
        """
        # If augmentation needs to be used for training and validation is
        # mentioned using the use_augmentation value. If true, augmentation is
        # applied to train and validation, but not to test. If augmentation
        # should be used for test, it is defined in the test_transform
        # function.
        self.use_augmentation = use_augmentation
        self.augmentation_strategies = augmentation_stratgies

        # Create a resize transformation
        self.resize_image_dimension = transforms.Resize(
            (resize_image_dimension, resize_image_dimension)
        )

        # Create a normalize transformation depending on the mean and standard
        # deviation provided, if they are not provided, don't have a
        # normalization process
        if mean != None and standard_deviation != None:
            self.normalize = transforms.Normalize(mean, standard_deviation)
        else:
            self.normalize = None

        # Create a transformation to convert to tensor.
        self.to_tensor = transforms.ToTensor()

    def train_transform(self) -> transforms.Compose:
        """
        train_transform: Builds a transforms.Compose object for train set.

        Based on the provided augmentation strategies as a transforms.Compose
        object, the train dataset is built into a transforms and returned.

        Returns
        -------
        transforms.Compose
            Object of transforms.Compose to make it easy for utilization in the
            training of the model.
        """
        # If augmentation strategies should be used for training dataset
        # transformation, the below steps are applied, if not, the condition
        # else is executed.
        if self.use_augmentation:
            train_transforms_parameter = self.augmentation_strategies
            train_transforms_parameter.extend(  # type: ignore
                [
                    self.resize_image_dimension,
                    self.to_tensor,
                ]
            )
        else:
            train_transforms_parameter = [  # type: ignore
                self.resize_image_dimension,
                self.to_tensor,
            ]

        if self.normalize != None:
            train_transforms_parameter.append(self.normalize)  # type: ignore

        return transforms.Compose(train_transforms_parameter)  # type: ignore

    def validation_transform(self) -> transforms.Compose:
        """
        validation_transform: transforms.Compose object for validation set.

        Based on the provided augmentation strategies as a list, the validation
        dataset is built into a transforms and returned.

        Returns
        -------
        transforms.Compose
            Object of transforms.Compose to make it easy for utilization in the
            training of the model.
        """
        # If augmentation strategies should be used for validation dataset
        # transformation, the below steps are applied, if not, the condition
        # else is executed.
        if self.use_augmentation:
            validation_transforms_parameter = self.augmentation_strategies
            validation_transforms_parameter.extend(  # type: ignore
                [
                    self.resize_image_dimension,
                    self.to_tensor,
                ]
            )
        else:
            validation_transforms_parameter = [  # type: ignore
                self.resize_image_dimension,
                self.to_tensor,
            ]

        if self.normalize != None:
            validation_transforms_parameter.append(self.normalize)  # type: ignore

        return transforms.Compose(validation_transforms_parameter)  # type: ignore

    def test_transform(
        self, test_augmentation: bool = False
    ) -> transforms.Compose:
        """
        test_transform: transforms.Compose object for test set.

        Based on the provided augmentation strategies as a list, the test
        dataset is built into a transforms and returned.

        Returns
        -------
        transforms.Compose
            Object of transforms.Compose to make it easy for utilization in the
            training of the model.
        """
        # If augmentation strategies should be used for test dataset
        # transformation, the below steps are applied, if not, the condition
        # else is executed.
        if test_augmentation:
            test_transforms_parameter = self.augmentation_strategies
            test_transforms_parameter.extend(  # type: ignore
                [
                    self.resize_image_dimension,
                    self.to_tensor,
                ]
            )
        else:
            test_transforms_parameter = [  # type: ignore
                self.resize_image_dimension,
                self.to_tensor,
            ]

        if self.normalize != None:
            test_transforms_parameter.append(self.normalize)  # type: ignore

        return transforms.Compose(test_transforms_parameter)  # type: ignore
