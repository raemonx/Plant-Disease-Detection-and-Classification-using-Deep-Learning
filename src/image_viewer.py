import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torchvision.datasets import ImageFolder
import fire
from torchvision.utils import make_grid
warnings.filterwarnings("ignore")
ROOT = os.getcwd()


def run(dataset: str):
    TRAIN_TRANSFORMS = {
        "cassava": transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2),
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ]),
        "crop_disease": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        "plant_village": transforms.Compose([
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

    VAL_TRANSFORMS = {
        "cassava": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "crop_disease": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        "plant_village": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    TEST_TRANSFORMS = {
        "cassava": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        "crop_disease": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        "plant_village": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }
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

    def imshow(img, normalized=True):
        if isinstance(img, tuple):
            img = img[0]

        if normalized:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img.numpy().transpose((1, 2, 0))
        else:
            img = img.permute(1, 2, 0).numpy()
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def preprocess(dir, ds_name, val_size=0.1, test_size=0.1, shuffle=True):
        subdirectories = [
            d
            for d in os.listdir(f"{dir}/{ds_name}")
            if os.path.isdir(os.path.join(f"{dir}/{ds_name}", d))
        ]

        no_classes = len(subdirectories)

        dataset = ImageFolder(f"{dir}/{ds_name}")


        n_val = int(val_size * len(dataset))
        n_test = int(test_size * len(dataset))
        n_train = len(dataset) - n_val - n_test

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val, n_test]
        )
        original_image, _ = train_dataset[0]
        print("Image before applying transforms:")
        imshow(torchvision.transforms.ToTensor()(original_image), normalized=False)


        train_dataset = ApplyTransform(train_dataset, transform=TRAIN_TRANSFORMS[ds_name])
        val_dataset = ApplyTransform(val_dataset, transform=VAL_TRANSFORMS[ds_name])
        test_dataset = ApplyTransform(test_dataset, transform=TEST_TRANSFORMS[ds_name])

        transformed_image = train_dataset[0]
        print("Image after transformation:")
        imshow(transformed_image, normalized=True)

    preprocess(ROOT,ds_name=dataset)
if __name__ == "__main__":
    fire.Fire(run)

