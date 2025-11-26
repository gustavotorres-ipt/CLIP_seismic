import os
import random
from PIL import Image
import json
from torch.utils.data import random_split
import torch


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


def load_data(processor, image_dir, text_dir):
    image_paths = sorted([
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
    ])

    print("Loading images and captions...")
    text_paths = sorted([
        os.path.join(text_dir, filename)
        for filename in os.listdir(text_dir)
    ])

    captions = [read_captions_json(path) for path in text_paths]

    train_dataset = CustomDataset(
        image_paths=image_paths, texts=captions, processor=processor)

    # return train_dataset, val_dataset
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size])

    return train_dataset, val_dataset, test_dataset

    image_files = sorted([f for f in os.listdir(image_dir)])
    data = []

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        txt_path = os.path.join(text_dir, f'{filename[:-4]}.json')

        if os.path.exists(txt_path):

            with open(txt_path, "r") as t:
                caption = json.load(t)['captions'][0]

            data.append({"image": img_path, "text": caption})

    dataset = Dataset.from_list(data)
    return dataset
