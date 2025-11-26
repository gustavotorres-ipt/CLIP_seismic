from model import CLIP_DistilBert_ResNet34
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

def clip_loss(logits_per_image, logits_per_text):
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def load_datasets():
    image_paths = [os.path.join(IMAGE_FOLDER, filename)
                   for filename in os.listdir(IMAGE_FOLDER) ]

    print("Loading images and captions...")
    text_paths = [os.path.join(TEXT_FOLDER, filename)
                  for filename in os.listdir(TEXT_FOLDER) ]
    captions = [read_captions_json(path) for path in text_paths]

    train_dataset = CustomDataset(
        image_paths=image_paths, texts=captions)

    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.2 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        train_dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset


def main():
    model = CLIP_DistilBert_ResNet34(embed_dim=512).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_dataset, val_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for images, texts in dataloader:
        images = images.cuda()
        # texts = list of caption strings
        
        logits_per_image, logits_per_text = model(images, texts)
        loss = clip_loss(logits_per_image, logits_per_text)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("loss:", loss.item())
            

if __name__ == "__main__":
    main()
