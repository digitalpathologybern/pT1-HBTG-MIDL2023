# Utils
import logging
import os

from abc import abstractmethod

import torch_geometric


class BaseGraphDatasetSetup():
    """
    Base implementaton for a graph dataset with PyTorch Geometric.
    """

    @abstractmethod
    def get_class_weights(self):
        """
        Compute the class weights
        """
        raise NotImplementedError

    @abstractmethod
    def get_node_feature_names(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def get_edge_feature_names(self, dataset):
        raise NotImplementedError

    @abstractmethod
    def get_class_names(self, dataset):
        raise NotImplementedError

    # Dataloaders handling
    @classmethod
    def set_up_dataloaders(cls, **kwargs):
        """ Set up the dataloaders for the specified datasets.
        """
        logging.info('Loading {} from:{}'.format(
            os.path.basename(os.path.normpath(kwargs['input_folder'])),
            kwargs['input_folder'])
        )

        # Load the datasets
        train_ds, val_ds, test_ds = cls._get_datasets(**kwargs)

        # Setup transforms
        logging.info('Setting up transforms')
        cls.set_up_transforms(train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, **kwargs)

        # Get the dataloaders
        train_loader, val_loader, test_loader = cls._dataloaders_from_datasets(train_ds=train_ds,
                                                                               val_ds=val_ds,
                                                                               test_ds=test_ds,
                                                                               **kwargs)
        for i in {train_loader, val_loader, test_loader}:
            i.class_names = cls.get_class_names(i.dataset)

        logging.info("Dataset loaded successfully")

        # implement this?
        # verify_dataset_integrity(**kwargs)
        return train_loader, val_loader, test_loader, train_ds.num_classes, train_ds.num_features

    @classmethod
    def _dataloaders_from_datasets(cls, batch_size, train_ds, val_ds, test_ds, **kwargs):
        """
        This function creates (and returns) dataloader from datasets objects

        Parameters
        ----------
        batch_size : int
            The size of the mini batch
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        workers:
            Number of workers to use to load the data.

        Returns
        -------
        train_loader : torch_geometric.loader.DataLoader
        val_loader : torch_geometric.loader.DataLoader
        test_loader : torch_geometric.loader.DataLoader
            The dataloaders for each split passed
        """
        # Setup dataloaders
        logging.debug('Setting up dataloaders')
        train_loader = torch_geometric.loader.DataLoader(train_ds,
                                                         batch_size=batch_size,
                                                         shuffle=True)
        val_loader = torch_geometric.loader.DataLoader(val_ds,
                                                       batch_size=batch_size)
        test_loader = torch_geometric.loader.DataLoader(test_ds,
                                                        batch_size=batch_size)
        return train_loader, val_loader, test_loader

    @abstractmethod
    def _get_datasets(self, input_folder, **kwargs):
        """
        Loads the dataset from file system and provides the dataset splits for train validation and test

        Parameters
        ----------
        input_folder : string
            Path string that points to the dataset location

        Returns
        -------
        train_ds : data.Dataset
        val_ds : data.Dataset
        test_ds : data.Dataset
            Train, validation and test splits
        """
        raise NotImplementedError

    # Transforms handling
    @classmethod
    def set_up_transforms(cls, train_ds, val_ds, test_ds, **kwargs):
        """Set up the data transform"""
        # Assign the transform to splits
        train_ds.transform = cls.get_train_transform(num_node_attributes=train_ds.num_node_attributes,
                                                     num_edge_attributes=train_ds.num_edge_attributes, **kwargs)
        for ds in [val_ds, test_ds]:
            if ds is not None:
                ds.transform = cls.get_test_transform(num_node_attributes=train_ds.num_node_attributes,
                                                      num_edge_attributes=train_ds.num_edge_attributes, **kwargs)
        for ds in [train_ds, val_ds, test_ds]:
            if ds is not None:
                ds.target_transform = cls.get_target_transform(**kwargs)
