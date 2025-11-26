import os
from config import BATCH_SIZE, LEARNING_RATE
import torch
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
# from datasets import load_dataset
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import load_data
from torch import nn
from config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY

NUM_EPOCHS = 30


def preprocess(batch, processor):
    inputs = processor(
        text=batch["text"],
        images=batch["image"],
        return_tensors="pt",
        padding=True
    )
    return inputs


def finetune_model(clip_model, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(clip_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    clip_model.to(device)
    clip_model.train()
    for epoch in range(NUM_EPOCHS):
        loss_epoch_train = 0

        for batch in tqdm(train_loader):

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = clip_model(**batch, return_loss=True)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch_train += loss.item()

        avg_train_loss = loss_epoch_train / len(train_loader)
        avg_val_loss = validation_step(clip_model, val_loader)

        print(f"Epoch {epoch} | Training loss: {avg_train_loss} | Validation loss: {avg_val_loss}")


def validation_step(clip_model, val_loader):
    # Validation Step
    clip_model.eval()
    loss_epoch_val = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = clip_model(**batch, return_loss=True)

            # Compute loss with learnable temperature
            loss = outputs.loss
            loss_epoch_val += loss.item()

    avg_val_loss = loss_epoch_val / len(val_loader)

    return avg_val_loss


def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset, val_dataset, _ = load_data(
        processor,
        "./imagens_janelas_sismofacies/training",
        "./legendas_sismofacies/training"
    )

    finetune_model(model, train_dataset, val_dataset)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    main()
