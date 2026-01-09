import torch
import math
import torch.nn.functional as F
import copy
from model import CLIP_DistilBert_ResNet
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import load_datasets
from tqdm import tqdm
from config import EPOCHS, LEARNING_RATES, BATCH_SIZE, PATIENCE, OUTPUT_MODEL, WARMUP_STEPS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def main():
    custom_clip_model = CLIP_DistilBert_ResNet().to(device)

    optimizer = AdamW([
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
    ], weight_decay=0.2)

    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    gr_acc_steps = 4 # Gradient accumulation_steps

    # Number of warmup steps and number of cosine scheduling steps.

    steps_per_epoch = math.ceil(len(train_loader) / gr_acc_steps)
    total_steps = max(steps_per_epoch * EPOCHS, 3000)
    cosine_steps = total_steps - WARMUP_STEPS

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_steps, eta_min=1e-6,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_STEPS,
    )

    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(custom_clip_model.state_dict())
    global_step = 0 # CLIP schedulers are defined according to steps not epochs

    for epoch in tqdm(range(EPOCHS)):  # Number of epochs
        ######### Training ###########
        print(f"\nEpoch {epoch + 1}...")

        custom_clip_model.train()
        total_train_loss, total_val_loss = 0, 0

        optimizer.zero_grad()

        for step, (images, texts) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            # texts = list of caption strings
            
            logits_per_image, logits_per_text = custom_clip_model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)

            loss = loss / gr_acc_steps
            loss.backward()

            if (step + 1) % gr_acc_steps == 0 or (step + 1) == len(train_loader):
                with torch.no_grad():
                    custom_clip_model.logit_scale.clamp_(0, 4.6)

                optimizer.step()
                optimizer.zero_grad()

                # Step scheduler
                if global_step < WARMUP_STEPS:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()
                global_step += 1
                if global_step % 100 == 0:

                    print(f"\nStep {global_step}, LR: {optimizer.param_groups[0]['lr']:.8f}")

            total_train_loss += loss.item() * gr_acc_steps

        ######### Validation ###########
        custom_clip_model.eval()

        with torch.no_grad():
            for images, texts in tqdm(val_loader):
                images = images.to(device)
                # texts = list of caption strings
                
                logits_per_image, logits_per_text = custom_clip_model(images, texts)
                loss = clip_loss(logits_per_image, logits_per_text)
                
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)


        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}",
              f"Logit Scale: {custom_clip_model.logit_scale.exp()}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(custom_clip_model.state_dict())
            epochs_no_improve = 0
            torch.save(custom_clip_model.state_dict(), OUTPUT_MODEL)
            print(OUTPUT_MODEL, "saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping on epoch {epoch}...')
                break

    custom_clip_model.load_state_dict(best_model_wts)
            

if __name__ == "__main__":
    main()
