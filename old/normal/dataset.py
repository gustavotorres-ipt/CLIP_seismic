import os
import random
from PIL import Image
import json
import torch

TEXT_FOLDER_TRAIN = "../data/captions/training"
TEXT_FOLDER_VAL = "../data/captions/validation"
IMAGE_FOLDER_TRAIN = "../data/images/training"
IMAGE_FOLDER_VAL = "../data/images/validation"


def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return random.choice(captions)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, processor):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        #if self.transform:
        #    image = self.transform(image)

        inputs = self.processor(
            text=self.texts[idx],
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=77
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}


def load_data(processor):
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

    train_dataset = CustomDataset(
        image_paths=image_paths_train, texts=captions_train, processor=processor
    )
    val_dataset = CustomDataset(
        image_paths=image_paths_val, texts=captions_val, processor=processor
    )

    return train_dataset, val_dataset
