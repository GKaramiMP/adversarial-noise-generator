import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import Optional, Callable
from datetime import datetime
import numpy as np

class MNISTDataset:
    def __init__(self, batch_size: int = 64,
                 train_transform: Optional[Callable] = None,
                 valid_transform: Optional[Callable] = None,
                 train_subset_size: Optional[int] = None,
                 valid_subset_size: Optional[int] = None):  
        """
        Initializes the MNIST dataset loader.
        Args:
            batch_size (int): Batch size for data loading.
            train_transform (Callable, optional): Transformations for training data.
            valid_transform (Callable, optional): Transformations for validation data.
            train_subset_size (int, optional): Size of the training subset.
            valid_subset_size (int, optional): Size of the validation subset.
        """
        self.num_classes = 10
        self.image_size = 28
        self.image_pixels = self.image_size ** 2
        self.batch_size = batch_size
        self.train_transform = train_transform 
        self.valid_transform = valid_transform 
        
        # Prepare datasets
        self.full_train_dataset = datasets.MNIST(root='MNIST_data',
                                                  train=True,
                                                  transform=self.train_transform,
                                                  download=True
                                                  )
        self.full_valid_dataset = datasets.MNIST(root='MNIST_data',
                                                  train=False,
                                                  transform=self.valid_transform,
                                                  download=True
                                                  )

        # Create subsets if subset sizes are specified (not train all samples)
        if train_subset_size:
            train_indices = np.random.choice(len(self.full_train_dataset), train_subset_size, replace=False)
            self.train_dataset = Subset(self.full_train_dataset, train_indices)
        else:
            self.train_dataset = self.full_train_dataset

        if valid_subset_size:
            valid_indices = np.random.choice(len(self.full_valid_dataset), valid_subset_size, replace=False)
            self.valid_dataset = Subset(self.full_valid_dataset, valid_indices)
        else:
            self.valid_dataset = self.full_valid_dataset
            

    def get_data_loader(self, 
                        batch_size: Optional[int] = None,
                        train_transform: Optional[Callable] = None,
                        valid_transform: Optional[Callable] = None):
        """
        Prepares and returns the data loaders for training and testing.
        Args:
            batch_size (int, optional): Batch size for data loading. Defaults to the initialized batch size.
            train_transform (callable, optional): Transformations for training data.
            test_transform (callable, optional): Transformations for testing data.
        Returns:
            dict: A dictionary with 'train' and 'test' DataLoader objects.
        """
        
        print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

        # Use provided transforms or fall back to default
        if train_transform:
            self.train_dataset.transform = train_transform
        if valid_transform:
            self.valid_dataset.transform = valid_transform

        # Use provided batch_size or fallback to default
        batch_size = batch_size or self.batch_size

        data_loaders = {
          'train': DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True),
          'valid': DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
          }

        print('Found {} training examples'.format(len(self.train_dataset)))
        print('Found {} validation examples'.format(len(self.valid_dataset)))

        return data_loaders
