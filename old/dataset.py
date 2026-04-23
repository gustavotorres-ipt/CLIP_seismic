import json
import os
import random
import torch
from torch.utils.data import random_split
from PIL import Image
from config import IMAGE_FOLDER_VALIDATION, IMAGE_FOLDER_TRAINING
from config import TEXT_FOLDER_VALIDATION, TEXT_FOLDER_TRAINING


def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return captions[0] #random.choice(captions)

# def load_images(batch_size):
#     image_paths = [os.path.join(IMAGE_FOLDER_VALIDATION, filename)
#                    for filename in os.listdir(IMAGE_FOLDER_VALIDATION) ]
#     random.shuffle(image_paths)
#     images = [Image.open(path).convert("RGB") for path in image_paths[:batch_size]]
#     return images

def load_datasets(preprocess):
    image_paths_train = [os.path.join(IMAGE_FOLDER_TRAINING, filename)
                         for filename in os.listdir(IMAGE_FOLDER_TRAINING) ]

    image_paths_val = [os.path.join(IMAGE_FOLDER_VALIDATION, filename)
                       for filename in os.listdir(IMAGE_FOLDER_VALIDATION) ]

    print("Loading images and captions...")
    text_paths_train = [os.path.join(TEXT_FOLDER_TRAINING, filename)
                        for filename in os.listdir(TEXT_FOLDER_TRAINING) ]
    captions_train = [read_captions_json(path) for path in text_paths_train]

    text_paths_val = [os.path.join(TEXT_FOLDER_VALIDATION, filename)
                      for filename in os.listdir(TEXT_FOLDER_VALIDATION) ]
    captions_val = [read_captions_json(path) for path in text_paths_val]

    train_dataset = CustomDataset(
        image_paths=image_paths_train, texts=captions_train, transform=preprocess
    )
    val_dataset = CustomDataset(
        image_paths=image_paths_val, texts=captions_val, transform=preprocess
    )

    return train_dataset, val_dataset, val_dataset


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
