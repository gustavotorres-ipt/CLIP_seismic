import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

class CLIP_Model(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. TEXT ENCODER (DistilBERT)
        # -----------------------------
        # self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    # --------------------------------------------------------------
    # Forward pass: compute contrastive logits
    # --------------------------------------------------------------
    def forward(self, batch):
        return self.model(**batch, return_loss=True)
