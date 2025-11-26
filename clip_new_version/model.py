import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast
from torchvision.models import resnet34
from torchvision import models

class CLIP_DistilBert_ResNet34(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        embed_dim: int = 512,
        image_pretrained: bool = True,
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. TEXT ENCODER (DistilBERT)
        # -----------------------------
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_name)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)

        text_hidden_dim = self.text_encoder.config.dim  # usually 768

        # Text projection into shared space
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)

        # -----------------------------
        # 2. IMAGE ENCODER (ResNet-34)
        # -----------------------------
        self.image_encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.image_encoder.fc = nn.Identity()  # remove classification head

        image_hidden_dim = 512  # ResNet-18 output dimension

        # Image projection into shared space
        self.image_proj = nn.Linear(image_hidden_dim, embed_dim)

        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / 0.07))
        )

    # ------------------------------------------------------------------
    # Encode text using DistilBERT → pooled embedding → projection → norm
    # ------------------------------------------------------------------
    def encode_text(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        # move to same device as module params
        tokens = {k: v.to(self.text_proj.weight.device) for k, v in tokens.items()}

        outputs = self.text_encoder(**tokens)

        # Use CLS token embedding (DistilBERT has CLS at index 0)
        text_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

        text_emb = self.text_proj(text_emb)  # project
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize
        return text_emb

    # --------------------------------------------------------------
    # Encode image using ResNet-18 → projection → norm
    # --------------------------------------------------------------
    def encode_image(self, images):
        img_feat = self.image_encoder(images)  # (batch, 512)
        img_emb = self.image_proj(img_feat)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        return img_emb

    # --------------------------------------------------------------
    # Forward pass: compute contrastive logits
    # --------------------------------------------------------------
    def forward(self, images, texts):
        image_emb = self.encode_image(images)  # (B, D)
        text_emb = self.encode_text(texts)  # (B, D)

        # Compute pairwise similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * (image_emb @ text_emb.T)
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text


