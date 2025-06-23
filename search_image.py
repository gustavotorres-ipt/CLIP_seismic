from clip_training import CustomCLIPModel, load_custom_encoders, IMAGE_FOLDER, TEXT_FOLDER, read_captions_json
import os
import random
import numpy as np
import torch
import open_clip
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt

N_CANDIDATE_IMAGES = 2000
CAPTIONS_FOLDER = "captions"

def read_user_input():
    caption = input("Search for a seismic image: ")
    return caption

def plot_image(image, caption=None):
    plt.imshow(image, cmap='gray')
    if caption is not None:
        plt.title(caption)
    plt.show()
    plt.close()


def load_images():
    images = []

    images_files = os.listdir(IMAGE_FOLDER)
    random.shuffle(images_files)
    for img in images_files[:N_CANDIDATE_IMAGES]:
        img_path = os.path.join(IMAGE_FOLDER, img)

        image = Image.open(img_path).convert("RGB")
        images.append(image)
    return images

print("Loading model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)

image_encoder, text_encoder = load_custom_encoders()
model = CustomCLIPModel(image_encoder, text_encoder)
model.load_state_dict(torch.load("customized_clip.pth"))

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)


print("Loading images...")
original_images = load_images()
images = np.array([preprocess(img) for img in original_images])
images = torch.from_numpy(images)

with torch.no_grad():
    images = images.to(device)
    image_features = model.encode_image(images)

while(True):
    with torch.no_grad():
        caption = read_user_input()

        print("Encoding text...")
        tok_caption = tokenizer(
            caption, padding=True, truncation=True, return_tensors="pt",
            max_length=128
        ).to(device)

        text_features = model.encode_text(tok_caption)

        print("Searching for best match...")
        logits_per_image, logits_per_text = model(images, tok_caption)
        probs = logits_per_text.softmax(dim=-1).cpu().numpy()

    best_match = np.argmax(probs)

    best_image = original_images[best_match]
    print("Image found!")
    plot_image(best_image, caption)
