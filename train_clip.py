import os
import sys
import argparse
import torch
import torch.nn.functional as F
import copy
from model import CLIP_DistilBert_ResNet
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset import load_datasets
from tqdm import tqdm
from config import EPOCHS, LEARNING_RATES, BATCH_SIZE, PATIENCE, device

STEPS_SCHEDULER = 2


def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def main(args):
    while os.path.exists(args.output_model):
        user_input = input(
            f"A model {args.output_model} already exists. Do you want to override it? (y/n): ")
        if user_input[0] == 'n':
            sys.exit(0)
        elif user_input[0] == 'y':
            break
        else:
            print("Invalid option.")


    custom_clip_model = CLIP_DistilBert_ResNet().to(device)

    optimizer = Adam([
        {'params': custom_clip_model.image_encoder.parameters(),
         'lr': LEARNING_RATES['image_encoder']},
        {'params': custom_clip_model.text_encoder.parameters(),
         'lr': LEARNING_RATES['text_encoder']},
        {'params': custom_clip_model.image_proj.parameters(),
         'lr': LEARNING_RATES['image_proj']},
        {'params': custom_clip_model.text_proj.parameters(),
         'lr': LEARNING_RATES['text_proj']},
        {'params': custom_clip_model.logit_scale,
         'lr': LEARNING_RATES['logit_scale']},
    ])

    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Number of warmup steps and number of cosine scheduling steps.

    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(custom_clip_model.state_dict())

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEPS_SCHEDULER,
        gamma=0.5      # multiply LR by 0.5
    )

    for epoch in tqdm(range(EPOCHS)):  # Number of epochs
        ######### Training ###########
        print(f"\nEpoch {epoch + 1}...")

        custom_clip_model.train()
        total_train_loss, total_val_loss = 0, 0

        for step, (images, texts, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            optimizer.zero_grad()
            logits_per_image, logits_per_text = custom_clip_model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                custom_clip_model.logit_scale.clamp_(0, 4.6)

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataset)

        ######### Validation ###########
        custom_clip_model.eval()

        with torch.no_grad():
            for images, texts, _ in tqdm(val_loader):
                images = images.to(device)
                # texts = list of caption strings
                
                logits_per_image, logits_per_text = custom_clip_model(images, texts)
                loss = clip_loss(logits_per_image, logits_per_text)
                
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataset)

        current_lr = scheduler.get_last_lr()
        scheduler.step()

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}",
              f"Logit Scale: {custom_clip_model.logit_scale.item()}")
        print("Current LR: ", current_lr)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(custom_clip_model.state_dict())
            epochs_no_improve = 0
            torch.save(custom_clip_model.state_dict(), args.output_model)
            print(args.output_model, "saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping on epoch {epoch}...')
                break

    custom_clip_model.load_state_dict(best_model_wts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a ResNet model.')

    parser.add_argument('-o', '--output_model', type=str, required=True,
                        help='Name of output pth model.')
    args = parser.parse_args()

    main(args)
