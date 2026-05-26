import numpy as np
import copy
from model_loader import load_clip_model
import open_clip
from torchvision.transforms.functional import to_pil_image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from dataset import load_datasets
from torchvision import transforms
from config import BATCH_SIZE, EPOCHS

# TODO fazer pares de imagens e legendas


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.stop_training = False

    def check_early_stopping(self, val_loss, clip_model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            clip_model.best_model_wts = copy.deepcopy( clip_model.state_dict())
            print("Model improved")
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.stop_training = True


def contrastive_loss(logits_per_image, logits_per_text):

    # Compute cosine similarity
    # logits_per_image = image_features @ text_features.T
    # logits_per_text = text_features @ image_features.T

    # Use learned temperature
    # logit_scale = logit_scale.exp().clamp(max=100)
    # logits_per_image *= logit_scale
    # logits_per_text *= logit_scale

    # Define labels (diagonal entries)
    # batch_size = image_features.size(0)
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    # Compute contrastive loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # Return average loss
    return (loss_i2t + loss_t2i) / 2

class LearnableTemperatureContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        # Initialize logit_scale (log(1/temperature))
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature)),
            requires_grad=True
        )

    def forward(self, image_features, text_features):
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T

        # Use learned temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image *= logit_scale
        logits_per_text *= logit_scale

        # Define labels (diagonal entries)
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Compute contrastive loss
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Return average loss
        return (loss_i2t + loss_t2i) / 2


def train_epoch():
    custom_clip_model.train()
    total_train_loss = 0
    for images, texts in tqdm(train_loader):
        images = images.to(device)
        texts = [t for t in texts]

        # images = images.clamp(0, 1).cpu()
        # # Convert to PIL image
        # final_image = to_pil_image(images[0])
        # final_image.show()

        tok_texts = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt",
            max_length=128
        ).to(device)

        # Forward pass: compute image and text features
        # image_features = custom_clip_model.encode_image(images)
        # text_features = custom_clip_model.encode_text(tok_texts)
        logits_per_image, logits_per_text = custom_clip_model(images, tok_texts)

        # Compute loss with learnable temperature
        loss = loss_fn(logits_per_image, logits_per_text)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # print("param (logit_scale):", loss_fn.logit_scale.item())
        # print("exp(logit_scale):", loss_fn.logit_scale.exp().item())
        # print("grad (logit_scale):", loss_fn.logit_scale.grad)
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss


def validation_step():
    # Validation Step
    custom_clip_model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            tok_texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt",
                max_length=128
            ).to(device)

            # Forward pass: compute image and text features
            # image_features = custom_clip_model.encode_image(images)
            # text_features = custom_clip_model.encode_text(tok_texts)
            logits_per_image, logits_per_text = custom_clip_model(images, tok_texts)

            # Compute loss with learnable temperature
            loss = loss_fn(logits_per_image, logits_per_text)
            total_val_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n",
          f"Logit scale: {custom_clip_model.logit_scale.exp()}")
    return avg_val_loss


def test_step(custom_clip_model):
    custom_clip_model.eval()  # Set the model to evaluation mode
    total_test_loss = 0  # Initialize variable to track test loss

    # Disable gradient calculation (since we're not training during testing)
    with torch.no_grad():
        # Iterate over the test dataset
        for images, texts in test_loader:
            images = images.to(device)  # Move images to the device

            tok_texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt",
                max_length=128
            ).to(device)
            # Forward pass: Compute image and text features
            # image_features = custom_clip_model.encode_image(images)
            # text_features = custom_clip_model.encode_text(tok_texts)
            logits_per_image, logits_per_text = custom_clip_model(images, tok_texts)

            # Compute loss with learnable temperature
            loss = loss_fn(logits_per_image, logits_per_text)
            total_test_loss += loss.item()  # Accumulate test loss

    # Calculate the average test loss
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model, _, preprocess = open_clip.create_model_and_transforms(
    #     'ViT-B-32', pretrained='laion2b_s34b_b79k'
    # )

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    custom_clip_model = load_clip_model()

    # loss_fn = LearnableTemperatureContrastiveLoss()
    # loss_fn = loss_fn.to(device)
    loss_fn = contrastive_loss

    optimizer = optim.AdamW([
        {'params': custom_clip_model.image_encoder.parameters(), 'lr': 3e-4},  # ← this is the real fix
        {'params': custom_clip_model.text_encoder.parameters(),  'lr': 1e-5},
        {'params': custom_clip_model.image_proj.parameters(),    'lr': 5e-5},
        {'params': custom_clip_model.text_proj.parameters(),     'lr': 5e-5},
        {'params': [custom_clip_model.logit_scale],              'lr': 1e-5},
    ], weight_decay=1e-3)

    early_stopping = EarlyStopping(patience=50, min_delta=0.001)

    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6,
    )

    train_dataset, val_dataset, test_dataset = load_datasets(preprocess)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    custom_clip_model = custom_clip_model.to(device)
    # best_model = custom_clip_model COPY HERE

    print("Starting training...")

    for epoch in tqdm(range(EPOCHS)):  # Number of epochs
        total_train_loss = train_epoch()

        # Validation Step
        avg_val_loss = validation_step()

        early_stopping.check_early_stopping(avg_val_loss, custom_clip_model)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # Adjust the learning rate using the scheduler
        lr_scheduler.step()

    custom_clip_model.load_state_dict(custom_clip_model.best_model_weights)
    print("exp(logit_scale):", custom_clip_model.logit_scale.exp())

    test_step(custom_clip_model)

    output_model = "clip_janelas_seismic_faces.pth"

    torch.save(custom_clip_model.state_dict(), output_model)
    print(output_model, "saved.")
