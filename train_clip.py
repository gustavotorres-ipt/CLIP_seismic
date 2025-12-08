import torch.nn.functional as F
import torch
import copy
from model import CLIP_DistilBert_ResNet
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import load_datasets
from tqdm import tqdm
from config import EPOCHS, LEARNING_RATES, BATCH_SIZE, PATIENCE

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

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )


    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(custom_clip_model.state_dict())

    for epoch in tqdm(range(EPOCHS)):  # Number of epochs
        print(f"\nEpoch {epoch}...")

        custom_clip_model.train()
        total_train_loss, total_val_loss = 0, 0

        for images, texts in tqdm(train_loader):
            images = images.to(device)
            # texts = list of caption strings
            
            logits_per_image, logits_per_text = custom_clip_model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

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

        lr_scheduler.step()

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}",
              f"Logit Scale: {custom_clip_model.logit_scale.exp()}")

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(custom_clip_model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f'Early stopping on epoch {epoch}...')
                break

    custom_clip_model.load_state_dict(best_model_wts)

    output_model = "clip_janelas_seismic_faces.pth"

    torch.save(custom_clip_model.state_dict(), output_model)
    print(output_model, "saved.")
            

if __name__ == "__main__":
    main()
