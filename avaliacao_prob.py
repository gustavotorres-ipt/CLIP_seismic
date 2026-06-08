import torch
import torch.nn.functional as F
from model import CLIP_DistilBert_ResNet


model = CLIP_DistilBert_ResNet().to(device)

# 1. Carregue o modelo com o bbb.pth salvo na época 11
model.load_state_dict(torch.load("bbb.pth"))
model.eval()

# 2. Crie um "banco" com 3 textos com conceitos sísmicos opostos
prompts = [
    "this is a parallel seismic image with low amplitude",
    "this is a chaotic seismic image with high amplitude",
    "this is a faulted seismic image with continuous reflectors"
]

# 3. Pegue UMA imagem de validação que você sabe que é paralela/baixa amplitude
# image_tensor shape: [1, 3, 96, 96]

with torch.no_grad():
    # Extraia os embeddings
    img_emb = F.normalize(model.encode_image(image_tensor.to(device)), dim=-1)
    text_emb = F.normalize(model.encode_text(prompts), dim=-1)
    
    # Calcule a probabilidade zero-shot (semelhante ao artigo original do CLIP)
    similarity = (img_emb @ text_emb.T) * 100
    probs = similarity.softmax(dim=-1)

for i, prompt in enumerate(prompts):
    print(f"Prompt: '{prompt}' | Probabilidade: {probs[0][i].item()*100:.2f}%")
