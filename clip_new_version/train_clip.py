from model import CLIP_DistilBert_ResNet34
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from dataset import load_datasets
from tqdm import tqdm
from config import EPOCHS, LEARNING_RATE


def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def main():
    custom_clip_model = CLIP_DistilBert_ResNet34(embed_dim=512).to(device)

    optimizer = AdamW([
        {'params': custom_clip_model.image_encoder.parameters(), 'lr': 1e-5},
        {'params': custom_clip_model.text_encoder.parameters(), 'lr': 1e-5},
        {'params': custom_clip_model.text_proj.parameters(), 'lr': 1e-4},
        # {'params': custom_clip_model.image_proj.parameters(), 'lr': 1e-4},
        {'params': custom_clip_model.logit_scale, 'lr': 1e-5},
    ], weight_decay=0.2)
    # Build scheduler
    # scheduler = cosine_decay_schedule(
    #     optimizer,
    #     num_epochs=EPOCHS,
    #     num_steps_per_epoch=len(train_loader)
    # )

    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    for epoch in tqdm(range(EPOCHS)):  # Number of epochs
        custom_clip_model.train()
        total_train_loss, total_val_loss = 0, 0

        for images, texts in train_loader:
            images = images.to(device)
            # texts = list of caption strings
            
            logits_per_image, logits_per_text = custom_clip_model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        with torch.no_grad():
            for images, texts in val_loader:
                images = images.to(device)
                # texts = list of caption strings
                
                logits_per_image, logits_per_text = custom_clip_model(images, texts)
                loss = clip_loss(logits_per_image, logits_per_text)
                
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
