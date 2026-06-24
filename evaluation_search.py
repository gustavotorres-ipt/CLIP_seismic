import numpy as np
import torch
import torch.nn.functional as F
from numpy._typing import NDArray
from tqdm import tqdm
from model import CLIP_DistilBert_ResNet
from config import CLIP_FILE, BATCH_SIZE, device
from torch.utils.data import DataLoader
from dataset import load_datasets


NUMBERS_FACIES = {
    '1': 'chaotic',   '2': 'divergent', '3': 'parallel',  '4': 'sigmoid',
}


def calc_cosine_similarities(
    dataloader: DataLoader, clip_encoder: CLIP_DistilBert_ResNet,
    prompt_embeds: torch.Tensor
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:

    cos_similarities_torch = []
    labels = []

    print("Calculating most similar images...")

    with torch.no_grad():
        # Calc cosine distance between all images
        for img_batch, _, label_batch in tqdm(dataloader):
            img_batch = img_batch.to(device)
            labels += list(label_batch)

            image_embeds = clip_encoder.encode_image(img_batch)

            prompt_embeds = F.normalize(prompt_embeds, dim=1)
            image_embeds = F.normalize(image_embeds, dim=1)

            similarities_batch = image_embeds @ prompt_embeds.T

            cos_similarities_torch.append(similarities_batch.squeeze(1))

        cos_similarities_torch = torch.concat(cos_similarities_torch)
        cos_similarities = cos_similarities_torch.detach().cpu().numpy()
        labels = np.array(labels)

        return cos_similarities, labels


def evaluate_classification_face(
        target_face, clip_encoder, dataloader, num_images):

    prompt = f'{target_face} seismic facies.'

    with torch.no_grad():
        text_embeds = clip_encoder.encode_text(prompt)

    cos_similarities, labels = calc_cosine_similarities(
        dataloader, clip_encoder, text_embeds)

    print("Calculating most similar images...")
    most_similar_imgs = np.argsort(cos_similarities)[::-1][:num_images]
    labels_top_images = labels[most_similar_imgs]

    valid_labels = [str(label) for label in labels_top_images
                    if label == target_face]

    accuracy = len(valid_labels) / len(labels_top_images)
    print("Accuracy:", accuracy)
    print()


def evaluate_recall_k(clip_encoder: CLIP_DistilBert_ResNet,
                      dataloader: DataLoader, k: int):
    """ Calculate the distances between all images and all texts and
    calculate the average Recall@k metric.

    We iterate through the images in batches and convert them into embeddings.
    For each image, we compute the distance between its embeddings
    and all text embeddings.

    Then we calculate recall@k, which measures how many times the text
    corresponding to an image was in the top k most similar embeddings).
    """
    clip_encoder.eval()

    with torch.no_grad():
        n_correct = 0
        n_images_already_checked = 0

        all_text_embeds = []
        all_captions = []

        # Encode all text embeddings first
        for _, text_batch, _ in dataloader:
            all_captions.extend(text_batch)

            text_embeds = clip_encoder.encode_text(text_batch)
            text_embeds = F.normalize(text_embeds, dim=1)

            all_text_embeds.append(text_embeds.cpu())

        all_text_embeds = torch.cat(all_text_embeds, dim=0).to(device)

        for img_batch, _, _ in tqdm(dataloader):
            # Encode the batch of images
            img_batch = img_batch.to(device)
            image_embeds = clip_encoder.encode_image(img_batch)
            image_embeds = image_embeds.to(device)
            image_embeds = F.normalize(image_embeds, dim=1)

            # We measure the similarity between embeddings of a batch of images
            # and all text embeddings
            similarities_batch = image_embeds @ all_text_embeds.T
            _, indices = torch.topk(similarities_batch, k=k, dim=1)

            # We count the number of times that the correct text is in the
            # top k only for this batch of images.
            for idx, topk in enumerate(indices):
                current_image = idx + n_images_already_checked

                retrieved_captions = [all_captions[i] for i in topk]
                current_caption = all_captions[current_image]

                if current_caption in retrieved_captions:
                    n_correct += 1
                    print("CURRENT CAPTION:", current_caption)
                    print("RETRIEVED:", retrieved_captions)

                # if (topk == current_image).any():
                #     n_correct += 1
                # gt_sim = similarities_batch[idx, current_image]
                # best_sim = similarities_batch[idx].max()

                # print(current_image, gt_sim.item(), best_sim.item())


            n_images_already_checked += len(img_batch)

        recall_k = n_correct / len(dataloader.dataset)
        return recall_k


def main():
    _, val_dataset = load_datasets(deterministic_captions=True)

    dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the CLIP model
    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    clip_encoder.load_state_dict(torch.load(CLIP_FILE))

    clip_encoder.eval()

    while True:
        selected_num = input(
            "What is the label?\n" +
            "1 - chaotic\n" +
            "2 - divergent\n" +
            "3 - parallel\n"+
            "4 - sigmoid\n",
        )
        if selected_num not in NUMBERS_FACIES:
            print("Invalid number.")
        else:
            break

    target_face = NUMBERS_FACIES[selected_num]
    evaluate_classification_face(
        target_face, clip_encoder, dataloader, num_images=200)
    # print(evaluate_recall_k(clip_encoder, dataloader, k=10))


if __name__ == "__main__":
    main()
