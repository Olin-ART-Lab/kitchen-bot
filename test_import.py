import torch
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset, DataLoader
import matplotlib.pyplot as plt
from typing import Protocol, Sequence, Any
from tqdm import tqdm
import os

class RandomAccessDataSource(Protocol):
  """Interface for datasources where storage supports efficient random access."""

  def __len__(self) -> int:
    """Number of records in the dataset."""

  def __getitem__(self, record_key: int) -> Sequence[Any]:
    """Retrieves records for the given record_keys."""

tf_dataset = tfds.load('bridge', split="train")

ds = tfds.as_numpy(tfds.load('bridge'))
ds_train = tfds.as_numpy(ds['train'])