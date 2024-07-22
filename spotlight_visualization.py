import os
import platform
import pandas as pd
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from dataset import BirdDataset
import ast
import numpy as np

def get_device():
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_predictions(data_loader, model, device, label_map):
    data_to_save = []
    valid_labels = set()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing images"):
            if batch is None:
                continue
            images, labels, image_paths = batch
            images = [img for img in images if img is not None]
            labels = [lbl for lbl in labels if lbl is not None]
            image_paths = [path for path in image_paths if path is not None]

            if not images or not labels or not image_paths:
                continue

            images = torch.stack(images).to(device)
            outputs = model.forward_features(images)
            embeddings = model.forward_head(outputs, pre_logits=True)
            preds = model.forward_head(outputs)
            top2_probabilities, top1_class_indices = torch.topk(preds.softmax(dim=1), k=2)

            for img, label, path, embed, top_probs, top_indices in zip(
                images, labels, image_paths, embeddings, top2_probabilities, top1_class_indices
            ):
                label_id = label_map.get(label, -1)
                if label_id == -1:
                    continue
                valid_labels.add(label)
                data_to_save.append({
                    'image': path,
                    'label_str': label,
                    'label': label_id,
                    'prediction': top_indices[0].item(),
                    'embeddings': embed.cpu().flatten().tolist(),
                    'probs': top_probs.cpu().numpy().tolist()
                })
    return data_to_save, list(valid_labels)

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:  # If the batch is empty after filtering
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    device = get_device()
    print(f"----------------------")
    print(f"Using device: {device}")
    print(f"----------------------\n")

    # Load the model
    model = timm.create_model('fastvit_sa12', pretrained=True, in_chans=3) # using fastvit_sa12
    model.to(device)
    model.eval()
    input_size = model.default_cfg['input_size'][1:]
    input_mean = list(model.default_cfg['mean'])
    input_std = list(model.default_cfg['std'])

    # Dataset path
    dataset_path = "./KaggleBirds/"
    csv_file = os.path.join(dataset_path, "birds.csv")

    # Read the CSV in chunks
    chunksize = 100000
    chunked_birds_df = pd.read_csv(csv_file, chunksize=chunksize)
    birds_df = pd.concat(chunked_birds_df)

    # Sample the dataset
    sample_size = 5000
    birds_df = birds_df.sample(n=sample_size, random_state=42)

    # Create label map
    unique_labels = birds_df['labels'].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=input_mean, std=input_std),
    ])

    # Create Dataset & DataLoader (using dataLoader for faster processing)
    bird_dataset = BirdDataset(birds_df, dataset_path, transform=transform)
    batch_size = 1
    data_loader = DataLoader(bird_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)

    # Get embeddings and other data
    data_to_save, valid_labels = get_predictions(data_loader, model, device, label_map)

    # Create DataFrame
    output_df = pd.DataFrame(data_to_save)

    # Save to CSV
    output_csv = './bird_dataset_predictions.csv'
    output_df.to_csv(output_csv, index=False)

    # Display with spotlight
    from renumics import spotlight
    spotlight.show(output_df, layout="https://spotlight.renumics.com/resources/image_classification_v1.0.json")

if __name__ == "__main__":
    main()