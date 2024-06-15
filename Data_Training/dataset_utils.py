"""
sources used:
Datasets & DataLoaders, https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#loading-a-dataset
"""

import os
from torch.utils.data import Dataset
from PIL import Image


class FacialExpressionDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_dataset(directory):
    img_paths = []
    labels = []
    class_names = ['Angry', 'Neutral', 'Focused', 'Happy']

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Directory {class_dir} does not exist.")
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_paths.append(img_path)
            labels.append(idx)

    return img_paths, labels, class_names
