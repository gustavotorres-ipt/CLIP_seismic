VISION_MODEL = 'checkpoints/resnet18_image_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/seismic_distilbert.pt'

IMAGE_FOLDER_TRAIN = 'data/images/training'
TEXT_FOLDER_TRAIN = 'data/captions/training'

IMAGE_FOLDER_VAL = 'data/images/validation'
TEXT_FOLDER_VAL = 'data/captions/validation'

LEARNING_RATES = {
    'image_encoder': 1e-5,
    'text_encoder': 1e-5,
    'image_proj': 1e-3,
    'text_proj': 1e-3,
    'logit_scale': 1e-4
}

# VISION_MODEL = 'checkpoints/resnet18_inline_xline_todos_tams_balanceado_encoder.pth'
# LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'
# 
# IMAGE_FOLDER_TRAIN = 'data/janelas_iline_xline_32_40_64_balanceado/training'
# TEXT_FOLDER_TRAIN = 'data/legendas_iline_xline_32_40_64_balanceado/training'
# 
# IMAGE_FOLDER_VAL = 'data/janelas_iline_xline_32_40_64_balanceado/validation'
# TEXT_FOLDER_VAL = 'data/legendas_iline_xline_32_40_64_balanceado/validation'
# 
# LEARNING_RATES = {
#     'image_encoder': 3e-4,
#     'text_encoder':  1e-5,
#     'image_proj':    5e-5,
#     'text_proj':     5e-5,
#     'logit_scale':   1e-5
# }

BATCH_SIZE = 32
PATIENCE = 30
EPOCHS = 50
PROJECTION_SIZE = 512

OUTPUT_MODEL = "clip_sismico_sintetico_com_preprocess.pth"
