import torch
import cv2
from glob import glob
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional

from albumentations import (
    Compose,
    Normalize,
    Resize,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    RandomRotate90,
    OneOf,
)
import albumentations as A
from glob import glob

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _get_transforms(use_augmentation: bool, img_size: int):
    if use_augmentation:
        return Compose(
            [
                Rotate(30, p=0.5),
                RandomRotate90(p=0.5),
                Resize(img_size, img_size),
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # Normalize(
                #     # mean=[0.485, 0.456, 0.406],
                #     # std=[0.229, 0.224, 0.225],
                #     mean=[0.4914, 0.4822, 0.4465],
                #     std=[0.2470, 0.2435, 0.2616]
                # ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                OneOf(
                    [
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),
                    ],
                    p=0.2,
                ),
                OneOf(
                    [
                        A.MotionBlur(blur_limit=3, p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                A.ShiftScaleRotate(rotate_limit=15),
                OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.1),
                        A.IAAPiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(p=0.3),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    return Compose(
        [
            Resize(img_size, img_size),
            # Normalize(
            #     # mean=[0.485, 0.456, 0.406],
            #     # std=[0.229, 0.224, 0.225],
            #     mean=[0.4914, 0.4822, 0.4465],
            #     std=[0.2470, 0.2435, 0.2616]
            # ),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        img_paths: List,
        labels: List,
        training: bool = True,
        img_size: int = 224,
        use_augmentation: bool = True,
    ):
        self.img_paths = img_paths
        self.labels = labels
        self.training = training
        self.img_size = img_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        img = cv2.imread(self.img_paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = _get_transforms(self.use_augmentation, self.img_size)(image=img)
        img = augmented["image"]

        if self.training:
            label = self.labels[item]

            return {
                "input": img,
                "target": torch.tensor(label, dtype=torch.long),
            }

        return {
            "input": img,
        }
