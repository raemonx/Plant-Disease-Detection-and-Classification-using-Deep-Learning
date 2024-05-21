import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

warnings.filterwarnings("ignore")

# Define transformations for the training dataset for various datasets
TRANSFORMS={
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ]),
}

#  Class to apply transforms to train, validation and test datasets
#     as train and validation/test cannot have same transforms
#     except for transforms like Resize, ToTensor, Normalize.
class ApplyTransform(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)


"""##remove this
def imshow(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))

    plt.imshow(img)
    plt.show()
    img = img * std + mean
    plt.imshow(img)
    plt.show()


##remove this"""


def preprocess(dir, ds_name, val_size=0.1, test_size=0.1, shuffle=True):
    # Discover subdirectories within the dataset directory
    subdirectories = [
        d
        for d in os.listdir(f"{dir}/{ds_name}")
        if os.path.isdir(os.path.join(f"{dir}/{ds_name}", d))
    ]
    # Determine the number of classes
    no_classes = len(subdirectories)

    # Load the dataset using torchvision ImageFolder
    dataset = ImageFolder(f"{dir}/{ds_name}")


    # Calculate sizes for train, validation, and test splits
    n_val = int(val_size * len(dataset))
    n_test = int(test_size * len(dataset))
    n_train = len(dataset) - n_val - n_test

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )

    # Apply the specified transforms to each dataset split
    train_dataset = ApplyTransform(train_dataset, transform=TRANSFORMS['train'])
    val_dataset = ApplyTransform(val_dataset, transform=TRANSFORMS['val'])
    test_dataset = ApplyTransform(test_dataset, transform=TRANSFORMS['test'])

    """
    images, _ = next(iter(train_dataset))
    img_grid = torchvision.utils.make_grid(images)
    imshow(img_grid)
    ##remove this"""

    return train_dataset, val_dataset, test_dataset, no_classes
