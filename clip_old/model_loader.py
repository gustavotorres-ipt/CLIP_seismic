import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LANGUAGE_MODEL, VISION_MODEL
from transformers import AutoTokenizer, AutoModel
from torchvision import models

class CustomCLIPModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, init_temperature=0.1):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor([torch.log(torch.tensor(1.0 / init_temperature))]))
        self.image_encoder = image_encoder  # Your custom image encoder
        self.text_encoder = text_encoder  # Your custom text encoder
        self.projection_layer = nn.Linear(in_features=768, out_features=512) 

    def encode_image(self, images):
        features_image = self.image_encoder(images)[:,:,0,0]
        return features_image
        # return self.projection_layer(features_image)  # Project to CLIP space

    def encode_text(self, tokenized_texts):
        output_llm = self.text_encoder(**tokenized_texts)
        features_text = output_llm.last_hidden_state[:, 0, :]
        features_proj = self.projection_layer(features_text)   # Custom text encoder
        return features_proj
        # return self.projection_layer(text_features)  # Project to CLIP space

    def forward(self, image, tokenized_text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(tokenized_text)
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T

        # Use learned temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image *= logit_scale
        logits_per_text *= logit_scale

        return logits_per_image, logits_per_text

def load_custom_encoders():
    # image_encoder

    if "resnet50" in VISION_MODEL:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        image_encoder = nn.Sequential(*list(model.children())[:-1])
    elif "resnet18" in VISION_MODEL:
        resnet18 = models.resnet18(pretrained=False)
        model = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        ) 
        image_encoder = model
    else:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        image_encoder = nn.Sequential(*list(model.children())[:-1])

    text_encoder = AutoModel.from_pretrained(LANGUAGE_MODEL)

    image_encoder.load_state_dict(torch.load(VISION_MODEL))  # Custom image encoder
    image_encoder.eval()

    return image_encoder, text_encoder

def load_clip_model():
    image_encoder, text_encoder = load_custom_encoders()
    custom_clip_model = CustomCLIPModel(image_encoder, text_encoder)

    return custom_clip_model
