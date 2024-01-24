import torch
import tensorflow_datasets as tfds
from torch.utils.data import Dataset, DataLoader

dataset = tfds.load("bridge")

class TFDatasetWrapper(Dataset):
    def __init__(self, tf_dataset):
        self.tf_dataset = tf_dataset

    def __len__(self):
        return len(self.tf_dataset)

    def __getitem__(self, idx):
        # TensorFlow datasets yield tuples of (image, label)
        image, label = self.tf_dataset[idx]
        # Convert to PyTorch tensors
        image = torch.from_numpy(image.numpy()).float()
        label = torch.tensor(label.numpy()).long()
        return image, label
    
pytorch_train_dataset = TFDatasetWrapper(dataset['train'])
print("Did the thing")