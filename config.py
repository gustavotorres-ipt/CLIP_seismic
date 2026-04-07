import torch

BATCH_SIZE = 128
PATIENCE = 10
EPOCHS = 30
PROJECTION_SIZE = 512
IMG_SIZE = 96

VISION_MODEL   = 'checkpoints/resnet18_f3_pari_peno_FAN_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/mlm_f3_pari_peno_FAN.pt'

IMAGE_FOLDER_TRAIN = 'data/imagens_f3_pari_peno/training'
TEXT_FOLDER_TRAIN = 'data/legendas_f3_pari_peno/training'
 
IMAGE_FOLDER_VAL = 'data/imagens_f3_pari_peno/validation'
TEXT_FOLDER_VAL = 'data/legendas_f3_pari_peno/validation'


LEARNING_RATES = {
    'image_encoder': 1e-5,
    'text_encoder':  1e-5,
    'image_proj':    1e-4,
    'text_proj':     1e-4,
    'logit_scale':   1e-4
}
CLIP_FILE = 'clip_f3_pari_peno_FAN.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
