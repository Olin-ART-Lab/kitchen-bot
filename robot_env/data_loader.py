import tensorflow_datasets as tfds
from torch.utils.data.dataset import Dataset
import pandas as pd
import os

class OpenXDataset(Dataset):
    """Open X dataset."""

    def __init__(self):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        train.get_next()
