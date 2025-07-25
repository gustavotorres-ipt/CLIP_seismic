import json
import os
import random
import torch
from torch.utils.data import random_split
from PIL import Image


IMAGE_FOLDER = "images"
TEXT_FOLDER = "captions"


def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return random.choice(captions)

def load_images(batch_size):
    image_paths = [os.path.join(IMAGE_FOLDER, filename)
                   for filename in os.listdir(IMAGE_FOLDER) ]
    random.shuffle(image_paths)
    images = [Image.open(path).convert("RGB") for path in image_paths[:batch_size]]
    return images

def load_datasets(preprocess):
    image_paths = [os.path.join(IMAGE_FOLDER, filename)
                   for filename in os.listdir(IMAGE_FOLDER) ]

    print("Loading images and captions...")
    text_paths = [os.path.join(TEXT_FOLDER, filename)
                  for filename in os.listdir(TEXT_FOLDER) ]
    captions = [read_captions_json(path) for path in text_paths]

    train_dataset = CustomDataset(
        image_paths=image_paths, texts=captions, transform=preprocess)

    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]
        return image, text
