from torch.utils.data import DataLoader, Dataset


class BaseDataLoader:
    def __init__(self, batch_size: int):
        """
        __init__: Initialization of requirements for defining data loaders.

        Parameters
        ----------
        batch_size : int
            Batch size that needs to be used for training of the models.
        """
        self.batch_size = batch_size

    def train_loader(self, train_dataset: Dataset) -> DataLoader:
        """
        train_loader: Generates a data loader for the training dataset.

        Parameters
        ----------
        train_dataset : Dataset
            Requires a pytorch dataset object related to the training dataset.

        Returns
        -------
        DataLoader
            Returns a data loader for the training dataset.
        """
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def validation_loader(self, validation_dataset: Dataset) -> DataLoader:
        """
        validation_loader: Generates a data loader for the validation dataset.

        Parameters
        ----------
        validation_dataset : Dataset
            Requires a pytorch dataset object related to the validation
            dataset.

        Returns
        -------
        DataLoader
            Returns a data loader for the validation dataset.
        """
        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def test_loader(self, test_dataset: Dataset) -> DataLoader:
        """
        test_loader: Generates a data loader for the test dataset.

        Parameters
        ----------
        test_dataset : Dataset
            Requires a pytorch dataset object related to the test dataset.

        Returns
        -------
        DataLoader
            Returns a data loader for the test dataset.
        """
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )
