BATCH_SIZE = 64
PATIENCE = 30
EPOCHS = 32
PROJECTION_SIZE = 512
WARMUP_STEPS = 100

# VISION_MODEL = 'checkpoints/resnet18_image_encoder.pth'
# LANGUAGE_MODEL = 'checkpoints/seismic_distilbert.pt'
# 
# IMAGE_FOLDER_TRAIN = 'data/images/training'
# TEXT_FOLDER_TRAIN = 'data/captions/training'
# 
# IMAGE_FOLDER_VAL = 'data/images/validation'
# TEXT_FOLDER_VAL = 'data/captions/validation'
# 
# LEARNING_RATES = {
#     'image_encoder': 5e-5,
#     'text_encoder': 5e-5,
#     'image_proj': 5e-4,
#     'text_proj': 5e-4,
#     'logit_scale': 5e-4
# }
# OUTPUT_MODEL = "clip_sismico_sintetico.pth"


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
#     'image_encoder': 1e-5,
#     'text_encoder':  1e-5,
#     'image_proj':    1e-4,
#     'text_proj':     1e-4,
#     'logit_scale':   1e-4
# }
# OUTPUT_MODEL = "clip_il_xl_32_40_64_balanceado.pth"

VISION_MODEL = 'checkpoints/resnet18_inline_xline_64_balanceado_v2_encoder.pth'
LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'

IMAGE_FOLDER_TRAIN = 'data/janelas_inline_xline_64_balanceado/training'
TEXT_FOLDER_TRAIN = 'data/legendas_inline_xline_64_balanceado/training'
 
IMAGE_FOLDER_VAL = 'data/janelas_inline_xline_64_balanceado/validation'
TEXT_FOLDER_VAL = 'data/legendas_inline_xline_64_balanceado/validation'

LEARNING_RATES = {
    'image_encoder': 5e-5,
    'text_encoder':  5e-5,
    'image_proj':    5e-4,
    'text_proj':     5e-4,
    'logit_scale':   5e-4
}
OUTPUT_MODEL = "clip_il_xl_64_balanceado.pth"
