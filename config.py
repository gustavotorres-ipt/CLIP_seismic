import torch

BATCH_SIZE = 192
PATIENCE = 10
EPOCHS = 30
PROJECTION_SIZE = 512
IMG_SIZE = 96

VISION_MODEL = 'checkpoints/resnet18_parihaka_f3_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/lang_ckpt_parihaka_f3.pt'

IMAGE_FOLDER_TRAIN = 'data/janelas_parihaka_f3_balanceado/training'
TEXT_FOLDER_TRAIN = 'data/legendas_parihaka_f3_balanceado/training'
 
IMAGE_FOLDER_VAL = 'data/janelas_parihaka_f3_balanceado/validation'
TEXT_FOLDER_VAL = 'data/legendas_parihaka_f3_balanceado/validation'

STEPS_SCHEDULER = 4

LEARNING_RATES = {
    'image_encoder': 1e-5,
    'text_encoder':  1e-5,
    'image_proj':    1e-4,
    'text_proj':     1e-4,
    'logit_scale':   1e-4
}
# CLIP_FILE = 'clip_parihaka_f3.pth'
CLIP_FILE = 'aaa.pth'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
