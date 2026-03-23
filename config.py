import torch

BATCH_SIZE = 64
PATIENCE = 10
EPOCHS = 30
PROJECTION_SIZE = 512

VISION_MODEL = 'checkpoints/resnet18_synthetic_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/seismic_distilbert.pt'

IMAGE_FOLDER_TRAIN = 'data/images/training'
TEXT_FOLDER_TRAIN = 'data/captions/training'

IMAGE_FOLDER_VAL = 'data/images/validation'
TEXT_FOLDER_VAL = 'data/captions/validation'

LEARNING_RATES = {
    'image_encoder': 1e-5,
    'text_encoder': 1e-5,
    'text_proj': 1e-3,
    'image_proj': 1e-3,
    'logit_scale': 1e-3,
}
OUTPUT_MODEL = "clip_sismico_sintetico.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
