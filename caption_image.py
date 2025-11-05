from model_loader import CustomCLIPModel, load_custom_encoders
from dataset import IMAGE_FOLDER, TEXT_FOLDER, read_captions_json
import os
import random
import numpy as np
import torch
import open_clip
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

N_CAPTIONS_REF = 10000
CAPTIONS_FOLDER = "captions"

def plot_image(image, caption=None):
    plt.imshow(image, cmap='gray')
    if caption is not None:
        plt.title(caption)
    plt.show()
    plt.close()

def load_captions():
    captions_files = [os.path.join(TEXT_FOLDER, f) for f in os.listdir(TEXT_FOLDER)]
    random.shuffle(captions_files)

    captions = [read_captions_json(path) for path in captions_files[:N_CAPTIONS_REF]]
    return captions

def load_image():
    selected_image = random.choice(os.listdir(IMAGE_FOLDER))
    img_path = os.path.join(IMAGE_FOLDER, selected_image)

    image = Image.open(img_path).convert("RGB")
    return image

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)

print("Tokenizing words...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
image_encoder, text_encoder = load_custom_encoders()
model = CustomCLIPModel(image_encoder, text_encoder)
model.load_state_dict(torch.load("customized_clip.pth"))

model.to(device)

captions = load_captions()
original_image = load_image()
image = preprocess(original_image).unsqueeze(0)

with torch.no_grad():
    tok_captions = tokenizer(
        captions, padding=True, truncation=True, return_tensors="pt",
        max_length=128
    ).to(device)

    image = image.to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(tok_captions)

    logits_per_image, logits_per_text = model(image, tok_captions)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

best_match = np.argmax(probs)

best_caption = captions[best_match]
plot_image(original_image, best_caption)
