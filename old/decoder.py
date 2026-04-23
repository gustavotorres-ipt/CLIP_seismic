import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math
from model_loader import load_clip_model
from transformers import AutoTokenizer, AutoModel
from clip_training import load_datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_image(image, title=None):
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, ch)
        self.norm2 = nn.GroupNorm(8, ch)
        self.dense = nn.Linear(emb_dim, ch)

    def forward(self, x, emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.dense(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h

class GlideDecoder256(nn.Module):
    def __init__(self, emb_dim=512, base_ch=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, base_ch * 16 * 16),
            nn.SiLU()
        )
        self.init_conv = nn.Conv2d(base_ch, base_ch, 3, padding=1)

        # encode
        self.res1 = ResBlock(base_ch, emb_dim)
        self.res2 = ResBlock(base_ch, emb_dim)
        self.res_mid = ResBlock(base_ch, emb_dim)

        # decode
        self.res_up1 = ResBlock(base_ch, emb_dim)
        self.res_up2 = ResBlock(base_ch, emb_dim)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)


    def forward(self, batch_size, t, text_emb):
        # noise: (B,3,256,256)
        # t: (B,)
        # text_emb: (B,512)
        t_emb = timestep_embedding(t, text_emb.size(-1)).to(device)

        cond = text_emb + t_emb

        # project to spatial and merge with noise
        # batch_size = noise.size(0)

        h = self.proj(text_emb).view(batch_size, -1, 16, 16)

        h = F.interpolate(h, scale_factor=16, mode='bilinear', align_corners=False)
        h = self.init_conv(h)

        # encoder
        h = self.res1(h, cond)
        h = F.avg_pool2d(h, 2)
        h = self.res2(h, cond)
        h = F.avg_pool2d(h, 2)
        h = self.res_mid(h, cond)

        # decoder
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.res_up1(h, cond)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.res_up2(h, cond)

        return torch.sigmoid(self.out_conv(h))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    _, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    clip_model = load_clip_model()
    clip_model.to(device)

    train_dataset, val_dataset, test_dataset = load_datasets(preprocess)

    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = GlideDecoder256()
    model.to(device)

    for images, texts in test_loader:
        with torch.no_grad():
            tok_texts = tokenizer(
                texts, padding=True, truncation=True,
                return_tensors="pt", max_length=128
            ).to(device)

            # noise = torch.randn(batch_size, 128, 256, 256)
            text_embeddings = clip_model.encode_text(tok_texts)

            t = torch.randint(0, 1000, (batch_size,))
            text_emb = torch.randn(batch_size, 512)
            out_imgs = model(batch_size, t, text_embeddings)

            out_imgs = out_imgs.cpu().numpy()

            for img in out_imgs:
                img_pil = np.transpose(img, (1, 2, 0))
                plot_image(img_pil)
