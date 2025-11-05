from model_loader import CustomCLIPModel, load_custom_encoders
from dataset import IMAGE_FOLDER
import os
import random
import numpy as np
import torch
import open_clip
from PIL import Image
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

N_CANDIDATE_IMAGES = 2000
CAPTIONS_FOLDER = "captions"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def load_clip_model():
    print("Loading model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )

    image_encoder, text_encoder = load_custom_encoders()
    model = CustomCLIPModel(image_encoder, text_encoder)
    model.load_state_dict(torch.load("customized_clip.pth"))

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.to(device)
    return model, preprocess, tokenizer

def load_and_encode_images(preprocess):
    print("Loading images...")
    original_images = load_images()
    images = np.array([preprocess(img) for img in original_images])
    images = torch.from_numpy(images)

    with torch.no_grad():
        images = images.to(device)
    return images, original_images

def search_image(model, caption, images_torch, original_images, tokenizer):
    with torch.no_grad():
        print("Encoding text...")
        tok_caption = tokenizer(
            caption, padding=True, truncation=True, return_tensors="pt",
            max_length=128
        ).to(device)

        print("Searching for best match...")
        logits_per_image, logits_per_text = model(images_torch, tok_caption)
        # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
        probs = logits_per_image.softmax(dim=0).cpu().numpy()

        idx_best_match = np.argmax(probs)

        best_match = original_images[idx_best_match]
        print("Image found!")
        return best_match

def main():
    model, preprocess, tokenizer = load_clip_model()
    images, original_images = load_and_encode_images(preprocess)

    while(True):
        caption = read_user_input()
        best_match = search_image(model, caption, images, original_images, tokenizer)
        plot_image(best_match, caption)

if __name__ == "__main__":
    main()
