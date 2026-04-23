import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import load_data
from config import BATCH_SIZE, LEARNING_RATES, WEIGHT_DECAY
from model_clip_openai import CLIP_Model

NUM_EPOCHS = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def preprocess(batch, processor):
#     inputs = processor(
#         text=batch["text"],
#         images=batch["image"],
#         return_tensors="pt",
#         padding=True
#     )
#     return inputs
# 

def finetune_model(clip_model, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW([
        {'params': clip_model.model.vision_model.parameters(),
         'lr': LEARNING_RATES['image_encoder']},
        {'params': clip_model.model.text_model.parameters(),
         'lr': LEARNING_RATES['text_encoder']},
        {'params': clip_model.model.visual_projection.parameters(),
         'lr': LEARNING_RATES['image_proj']},
        {'params': clip_model.model.text_projection.parameters(),
         'lr': LEARNING_RATES['text_proj']},
        {'params': clip_model.model.logit_scale,
         'lr': LEARNING_RATES['logit_scale']},
    ], weight_decay=WEIGHT_DECAY)

    clip_model.train()
    for epoch in range(NUM_EPOCHS):
        loss_epoch_train = 0

        for batch in tqdm(train_loader):

            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = clip_model(batch)

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
            outputs = clip_model(batch)

            # Compute loss with learnable temperature
            loss = outputs.loss
            loss_epoch_val += loss.item()

    avg_val_loss = loss_epoch_val / len(val_loader)

    return avg_val_loss


def main():
    clip_model = CLIP_Model()
    clip_model.to(device)

    train_dataset, val_dataset = load_data(clip_model.processor)

    finetune_model(clip_model, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
