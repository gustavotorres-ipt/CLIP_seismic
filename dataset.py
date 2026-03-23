import json
import os
import random
import torch
# from torch.utils.data import random_split
from PIL import Image
from config import IMAGE_FOLDER_TRAIN, TEXT_FOLDER_TRAIN, IMAGE_FOLDER_VAL, TEXT_FOLDER_VAL
from torchvision import transforms
# from torchvision.transforms.v2 import GaussianNoise


class ImageNorm(object):
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)


def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return captions

def load_images(batch_size):
    image_paths = [os.path.join(IMAGE_FOLDER_TRAIN, filename)
                   for filename in sorted(os.listdir(IMAGE_FOLDER_TRAIN)) ]
    random.shuffle(image_paths)
    images = [Image.open(path).convert("RGB")
              for path in image_paths[:batch_size]]
    return images

def load_datasets():
    image_paths_train = [
        os.path.join(IMAGE_FOLDER_TRAIN, filename)
        for filename in sorted(os.listdir(IMAGE_FOLDER_TRAIN))
    ]

    image_paths_val = [
        os.path.join(IMAGE_FOLDER_VAL, filename)
        for filename in sorted(os.listdir(IMAGE_FOLDER_VAL))
    ]

    print("Loading images and captions...")

    text_paths_train = [os.path.join(TEXT_FOLDER_TRAIN, filename)
                        for filename in sorted(os.listdir(TEXT_FOLDER_TRAIN)) ]
    captions_train = [read_captions_json(path) for path in text_paths_train]

    text_paths_val = [os.path.join(TEXT_FOLDER_VAL, filename)
                      for filename in sorted(os.listdir(TEXT_FOLDER_VAL)) ]
    captions_val = [read_captions_json(path) for path in text_paths_val]

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        ImageNorm(),
    ])

    train_dataset = CustomDataset(
        image_paths=image_paths_train, caption_list_per_image=captions_train,
        transform=transformation
    )
    val_dataset = CustomDataset(
        image_paths=image_paths_val, caption_list_per_image=captions_val,
        transform=transformation
    )

    return train_dataset, val_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, caption_list_per_image, transform=None):
        self.image_paths = image_paths
        self.caption_list_per_image = caption_list_per_image
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = random.choice(self.caption_list_per_image[idx])
        return image, text
