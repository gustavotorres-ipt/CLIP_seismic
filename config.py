# VISION_MODEL = 'resnet18_image_encoder.pth'
# LANGUAGE_MODEL = 'seismic_distilbert.pt'
# 
# IMAGE_FOLDER_TRAIN = '../images/training'
# TEXT_FOLDER_TRAIN = '../captions/training'
# 
# IMAGE_FOLDER_VAL = '../images/validation'
# TEXT_FOLDER_VAL = '../captions/validation'
# 
# LEARNING_RATES = {
#     'image_encoder': 1e-5,
#     'text_encoder': 1e-5,
#     'image_proj': 1e-4,
#     'text_proj': 1e-4,
#     'logit_scale': 1e-5
# }

VISION_MODEL = 'checkpoints/resnet18_inline_xline_todos_tams_balanceado_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'

IMAGE_FOLDER_TRAIN = 'data/janelas_iline_xline_32_40_64_balanceado/training'
TEXT_FOLDER_TRAIN = 'data/legendas_iline_xline_32_40_64_balanceado/training'

IMAGE_FOLDER_VAL = 'data/janelas_iline_xline_32_40_64_balanceado/validation'
TEXT_FOLDER_VAL = 'data/legendas_iline_xline_32_40_64_balanceado/validation'

LEARNING_RATES = {
    'image_encoder': 3e-4,
    'text_encoder':  1e-5,
    'image_proj':    5e-5,
    'text_proj':     5e-5,
    'logit_scale':   1e-5
}

BATCH_SIZE = 64
PATIENCE = 30
EPOCHS = 50
